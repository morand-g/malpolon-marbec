"""NVIDIA DALI-backed DataModule for GPU-accelerated data loading.

The entire pipeline -- tar I/O, reshape, normalize, random flip -- runs
on GPU via DALI, eliminating Python per-sample overhead.  CPU threads
only handle sequential tar reads; everything else is a GPU kernel.

Standalone: no dependency on malpolon -- only needs Lightning, DALI, numpy.

Shard layout (created by ``create_shards.py``)::

    fold0-000000.tar, fold0-000001.tar, ...
    fold1-000000.tar, ...
    ...
    meta.json
    normalize.json

At training time, *fold* selects which groups are test / val / train:
    fold=k -> test=fold_k, val=fold_{k+1}, train=remaining folds
"""
import os
import time

import json
import tarfile
from pathlib import Path

import numpy as np
import lightning.pytorch as pl

from canonical_split import split_sample_ids, validate_shard_folds
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class _DALIWrapper:
    def __init__(self, dali_iter, split):
        self._dali_iter = dali_iter
        self._split = split
        self._epoch = 0

    def __iter__(self):
        start = time.perf_counter()
        previous = start

        for batch_idx, batch_list in enumerate(self._dali_iter):
            now = time.perf_counter()
            wait = now - previous

            if batch_idx % 20 == 0 or wait > 2:
                print(
                    f"[DALI WAIT] split={self._split} "
                    f"epoch={self._epoch} "
                    f"batch={batch_idx} "
                    f"wait={wait:.3f}s"
                )

            d = batch_list[0]
            yield d["tile"], d["label"]

            previous = time.perf_counter()

        elapsed = time.perf_counter() - start
        print(
            f"[DALI EPOCH] split={self._split} "
            f"epoch={self._epoch} "
            f"time={elapsed:.1f}s"
        )
        self._epoch += 1

    def __len__(self):
        return len(self._dali_iter)


class DALIWebDatasetModule(pl.LightningDataModule):
    """Lightning DataModule using NVIDIA DALI for GPU-accelerated data loading.

    Reads from fold-based uint16 WebDataset shards (created by
    ``create_shards.py``).  The GPU pipeline is::

        reinterpret(uint16) -> reshape(24,H,W) -> cast(float32)
        -> normalize -> D4 augmentation (train only)

    Shards store the 6 reflectance bands as uint16 = reflectance * 10000.
    This is equivalent to the requested Albumentations Normalize with
    max_pixel_value=0.0001, so DALI applies (uint16 - mean) / std.

    Parameters
    ----------
    wds_dir : str
        Directory containing ``fold*-*.tar``, ``meta.json``, and
        ``normalize.json`` (produced by ``create_shards.py``).
    fold : int
        Which fold to hold out as test (val = fold+1, train = rest).
    n_folds : int
        Total number of folds.  Must match the shards on disk.
    train_batch_size, inference_batch_size, num_workers, device_id
        Standard DataModule / DALI parameters.
    """

    def __init__(
        self,
        wds_dir: str,
        fold: int = 0,
        n_folds: int = 5,
        canonical_fold_ids: dict[str, list[str]] = None,
        train_batch_size: int = 32,
        inference_batch_size: int = 16,
        num_workers: int = 4,
        device_id: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.wds_dir = Path(wds_dir).resolve()  # DALI needs absolute paths
        self.device_id = device_id
        self.fold = fold

        # Load metadata
        self.meta = json.load(open(self.wds_dir / "meta.json"))
        self._shape = list(self.meta["shape"])  # [24, 224, 224]
        n_channels = self._shape[0]
        n_bands = self.meta["n_bands"]  # 6
        n_trimesters = self.meta.get("n_trimesters", 1)  # 4

        # Map split -> list of fold indices
        self._split_folds = {
            "test":  [fold % n_folds],
            "val":   [(fold + 1) % n_folds],
            "train": [(fold + i) % n_folds for i in range(2, n_folds)],
        }
        if canonical_fold_ids is not None:
            validate_shard_folds(self.wds_dir, canonical_fold_ids)
            self.split_sample_ids = split_sample_ids(canonical_fold_ids, fold)

        # Load 6-band normalization constants.
        # Albumentations Normalize(mean, std, max_pixel_value=0.0001) applied
        # to reflectance is equivalent here to normalizing stored uint16 values
        # as (uint16 - mean) / std.
        mean_6 = np.array(
            [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
            dtype=np.float32,
        )

        std_6 = np.array(
            [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
            dtype=np.float32,
        )
        if len(mean_6) != n_bands or len(std_6) != n_bands:
            raise ValueError(
                f"normalize.json has {len(mean_6)} mean values and {len(std_6)} std values, "
                f"but meta.json declares n_bands={n_bands}"
            )

        # Tile 6-band stats to 24 channels (4 trimesters x 6 bands)
        self._mean = np.tile(mean_6, n_trimesters).reshape(n_channels, 1, 1)
        self._std = np.tile(std_6, n_trimesters).reshape(n_channels, 1, 1)

        # Flat lists for fn.constant
        self._mean_list = self._mean.ravel().tolist()
        self._std_list = self._std.ravel().tolist()

    def _get_shard_paths(self, split):
        """Collect tar shard paths and corresponding index paths for *split*."""
        shard_paths = []
    
        for k in self._split_folds[split]:
            shard_paths.extend(sorted(self.wds_dir.glob(f"fold{k}-*.tar")))
    
        if not shard_paths:
            raise FileNotFoundError(
                f"No shards for split '{split}' "
                f"(folds {self._split_folds[split]}) "
                f"in {self.wds_dir}"
            )
    
        index_paths = [p.with_suffix(".idx") for p in shard_paths]
    
        missing = [str(p) for p in index_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing WebDataset index files:\n" + "\n".join(missing)
            )
    
        return (
            [str(p) for p in shard_paths],
            [str(p) for p in index_paths],
        )
        

    def _build_pipeline(self, split, batch_size, is_train):

        shard_paths, index_paths = self._get_shard_paths(split)
        shape = self._shape
        n_channels = shape[0]
        mean_list = self._mean_list
        std_list = self._std_list
        num_threads = max(self.num_workers, 1)
        device_id = self.device_id

        @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
        def pipe():
            raw_tile, raw_label = fn.readers.webdataset(
                paths=shard_paths,
                index_paths=index_paths,
                ext=["input", "target"],
                random_shuffle=is_train,
                initial_fill=512 if is_train else 1,
                seed=42,
                prefetch_queue_depth=1,
                read_ahead=False,
                dont_use_mmap=True,
                name="reader",
            )

            # GPU decode: raw bytes -> uint16 -> reshape -> float32
            tile = fn.reinterpret(raw_tile.gpu(), dtype=types.UINT16)
            tile = fn.reshape(tile, shape=shape)
            tile = fn.cast(tile, dtype=types.FLOAT)

            # Normalize like A.Normalize(..., max_pixel_value=0.0001) on stored reflectance uint16
            mean = fn.constant(fdata=mean_list, shape=[n_channels, 1, 1],
                               dtype=types.FLOAT, device="gpu")
            std = fn.constant(fdata=std_list, shape=[n_channels, 1, 1],
                              dtype=types.FLOAT, device="gpu")

            tile = (tile - mean) / std

            if is_train:
                tile = fn.flip(
                    tile,
                    horizontal=fn.random.coin_flip(),
                    vertical=fn.random.coin_flip(),
                )

            # Label
            label = fn.reinterpret(raw_label.gpu(), dtype=types.FLOAT)
            label = fn.reshape(label, shape=[1])

            return tile, label

        p = pipe()
        p.build()
        return p

    def _make_loader(self, split, batch_size, is_train):
        pipe = self._build_pipeline(split, batch_size, is_train)
        return _DALIWrapper(
            DALIGenericIterator(
                pipe,
                ["tile", "label"],
                reader_name="reader",
                auto_reset=True,
                last_batch_policy=LastBatchPolicy.PARTIAL,
            ), split = split
        )

    def _ordered_fold_sample_ids(self, fold: int) -> list[str]:
        sample_ids = []
    
        shard_paths = sorted(
            Path(self.wds_dir).glob(f"fold{fold}-*.tar")
        )
    
        for shard_path in shard_paths:
            with tarfile.open(shard_path) as shard:
                for member in shard:
                    if member.isfile() and member.name.endswith(".input"):
                        sample_ids.append(
                            member.name.removesuffix(".input")
                        )
    
        return sample_ids

    def train_dataloader(self):
        return self._make_loader("train", self.train_batch_size, is_train=True)

    def val_dataloader(self):
        return self._make_loader("val", self.inference_batch_size, is_train=False)

    def test_dataloader(self):
        return self._make_loader("test", self.inference_batch_size, is_train=False)

    def test_sample_ids(self) -> list[str]:
        return self._ordered_fold_sample_ids(self.fold)
