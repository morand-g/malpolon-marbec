from pathlib import Path
import sys
import yaml

POVERTY_DIR = Path("/lustre/fsn1/projects/rech/cvn/udm31lu/malpolon-marbec/examples/poverty")

TIFF_ROOT = Path("/lustre/fsn1/projects/rech/cvn/udm31lu/images/seasonal")
DALI_ROOT = Path("/lustre/fsn1/projects/rech/cvn/udm31lu/webdataset_mada_tempov")

sys.path.insert(0, str(POVERTY_DIR))

from poverty_dataset import MSDataModule
from dali_datamodule import DALIWebDatasetModule

CFG = yaml.safe_load(
    (POVERTY_DIR / "config" / "cnn_on_ms_torchgeo_config.yaml")
    .read_text(encoding="utf-8")
)


def summarize_loader(name, loader):
    batch_sizes = []
    total_samples = 0
    total_batches = 0

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch["tile"]
        else:
            raise TypeError(type(batch))

        bs = int(x.shape[0])

        batch_sizes.append(bs)
        total_samples += bs
        total_batches += 1

    print(f"\n{name}")
    print(f"  number of batches: {total_batches}")
    print(f"  first batch size: {batch_sizes[0]}")
    print(f"  last batch size: {batch_sizes[-1]}")
    print(f"  unique batch sizes: {sorted(set(batch_sizes))}")
    print(f"  total samples yielded: {total_samples}")


if __name__=="__main__":
    # TIFF
    tiff_dm = MSDataModule(
        dataset_path=str(TIFF_ROOT),
        labels_name=CFG["data"]["labels_name"],
        train_batch_size=CFG["data"]["train_batch_size"],
        inference_batch_size=CFG["data"]["inference_batch_size"],
        num_workers=CFG["data"]["num_workers"],
        fold="A",
        fold_path=str(POVERTY_DIR / CFG["data"]["fold_path"]),
        nature=CFG["data"]["nature"],
        nightlight=CFG["data"]["nightlight"],
        dict_normalize=str(POVERTY_DIR / CFG["data"]["dict_normalize"]),
    )
    
    tiff_dm.setup("fit")
    summarize_loader(
        "TIFF train",
        tiff_dm.train_dataloader(),
    )
    
    
    # DALI
    dali_dm = DALIWebDatasetModule(
        wds_dir=str(DALI_ROOT),
        fold=0,
        n_folds=5,
        train_batch_size=CFG["data"]["train_batch_size"],
        inference_batch_size=CFG["data"]["inference_batch_size"],
        num_workers=CFG["data"]["num_workers"],
    )
    
    summarize_loader(
        "DALI train",
        dali_dm.train_dataloader(),
    )

    import pandas as pd

    # CSV canonique corrigé
    CSV = "outputs/data_corrections/africa_v2/common_seasonal_composite_folds.csv"
    
    canonical = pd.read_csv(CSV)
    
    # Madagascar uniquement, train du fold A = C + D + E
    expected = canonical[
        (canonical["country"].str.lower() == "madagascar")
        & (canonical["spatial_fold"].isin(["C", "D", "E"]))
    ].copy()
    
    # Clés attendues
    expected_keys = {
        f"{str(row.country).lower()}_{int(row.year)}_{int(row.cluster_id)}"
        for row in expected.itertuples()
    }
    
    # Clés réellement présentes dans le Dataset TIFF
    tiff_keys = {
        f"{str(row.country).lower()}_{int(row.year)}_{int(row.cluster_id)}"
        for row in ds.dataframe.itertuples()
    }
    
    print("Expected train:", len(expected_keys))
    print("TIFF train:", len(tiff_keys))
    
    print("\nExpected but missing from TIFF:")
    print(sorted(expected_keys - tiff_keys))
    
    print("\nUnexpected in TIFF:")
    print(sorted(tiff_keys - expected_keys))