import numpy as np
import rasterio
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig

# Import their existing module
from poverty_dataset import MSDataModule 

# --- CONFIGURATION ---
REF_BANDS = ['red', 'green', 'blue', 'nir08', 'swir16', 'swir22']
LWIR_BAND = ['lwir', 'lwir11']

# Quantization Constants
REF_OFFSET = 0.2
REF_SCALE = 0.0000275
TEMP_SCALE = 100.0 

# TARGET SIZE (From Paper & ResNet requirements)
TARGET_H, TARGET_W = 224, 224

def center_crop_array(img, target_h, target_w):
    """
    Crops a (C, H, W) or (H, W) numpy array to the center.
    """
    # handle 2D or 3D case
    if len(img.shape) == 3:
        h, w = img.shape[1], img.shape[2]
    else:
        h, w = img.shape[0], img.shape[1]
    
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    if start_h < 0 or start_w < 0:
        # If image is smaller than target, pad it (edge case)
        # But you said images are ~350, so this shouldn't happen.
        raise ValueError(f"Image too small: {h}x{w} < {target_h}x{target_w}")

    if len(img.shape) == 3:
        return img[:, start_h:start_h+target_h, start_w:start_w+target_w]
    else:
        return img[start_h:start_h+target_h, start_w:start_w+target_w]

def create_processed_memmap_composite(config):
    config.data.nature = 'composite' 
    
    dm = MSDataModule(**config.data, fold=config.run.fold)
    ds_raw = dm.get_dataset(split='all', transform=None)
    dataframe = ds_raw.dataframe
    root_dir = ds_raw.root_dir
    
    N = len(dataframe)
    
    print(f"Preprocessing {N} samples.")
    print(f"Operation: Center Crop ({TARGET_H}x{TARGET_W}) -> Quantize")

    # Create Memmaps with FIXED 224x224 size
    fp_ref = np.memmap("composite_reflectance.dat", dtype='uint16', mode='w+', shape=(N, 6, TARGET_H, TARGET_W))
    fp_temp = np.memmap("composite_temperature.dat", dtype='uint16', mode='w+', shape=(N, 1, TARGET_H, TARGET_W))

    for i in tqdm(range(N)):
        row = dataframe.iloc[i]
        
        ref_buffer = [] 
        lwir_buffer = []
        
        filename = f"{row.cluster_id}.tif"
        tile_path = os.path.join(root_dir, str(row.country.lower()), str(row.year), filename)
        
        try:
            with rasterio.open(tile_path) as src:
                desc_map = {desc: idx + 1 for idx, desc in enumerate(src.descriptions)}
                
                # Read Reflectance
                season_ref = []
                valid_ref = True
                for b in REF_BANDS:
                    if b in desc_map:
                        season_ref.append(src.read(desc_map[b]))
                    else:
                        valid_ref = False; break
                
                if valid_ref:
                    ref_stack = np.stack(season_ref) 
                    # CROP HERE before buffering (saves memory)
                    ref_stack = center_crop_array(ref_stack, TARGET_H, TARGET_W)
                    ref_buffer.append(ref_stack)

                # Read LWIR
                if LWIR_BAND[0] in desc_map:
                    lwir_data = src.read(desc_map[LWIR_BAND[0]])
                    lwir_data = center_crop_array(lwir_data, TARGET_H, TARGET_W)
                    lwir_buffer.append(lwir_data)
                elif LWIR_BAND[1] in desc_map:
                    lwir_data = src.read(desc_map[LWIR_BAND[1]])
                    lwir_data = center_crop_array(lwir_data, TARGET_H, TARGET_W)
                    lwir_buffer.append(lwir_data)

                else:
                    print(f"Warning: LWIR band not found in {tile_path}")
                    
        except Exception as e:
            print(f"Error processing {tile_path}: {e}")
            continue
        
        # --- AGGREGATE & SAVE ---
        
        # Reflectance
        if len(ref_buffer) > 0:
            ref_all = np.stack(ref_buffer, axis=0)
            ref_all = np.nan_to_num(ref_all)
        else:
            ref_all = np.zeros((6, TARGET_H, TARGET_W), dtype=np.float32)
            
        ref_quant = (ref_all + REF_OFFSET) / REF_SCALE
        fp_ref[i] = np.clip(ref_quant, 0, 65535).astype(np.uint16)

        # Temperature
        if len(lwir_buffer) > 0:
            lwir_all = np.stack(lwir_buffer, axis=0)
            lwir_all = np.nan_to_num(lwir_all)
        else:
            lwir_all = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)
            
        temp_quant = (lwir_all * TEMP_SCALE)
        fp_temp[i] = np.clip(temp_quant, 0, 65535).astype(np.uint16)[None, :, :]
        
        if i % 500 == 0:
            fp_ref.flush(); fp_temp.flush()

    fp_ref.flush()
    fp_temp.flush()
    
    np.save("composite_meta.npy", {
        "shape_ref": (N, 6, TARGET_H, TARGET_W),
        "shape_temp": (N, 1, TARGET_H, TARGET_W),
        "ref_offset": REF_OFFSET, "ref_scale": REF_SCALE, "temp_scale": TEMP_SCALE
    })
    print("Done.")

def create_processed_memmap_composite(config):
    config.data.nature = 'composite' 
    
    dm = MSDataModule(**config.data, fold=config.run.fold)
    ds_raw = dm.get_dataset(split='all', transform=None)
    dataframe = ds_raw.dataframe
    root_dir = ds_raw.root_dir
    
    N = len(dataframe)
    
    print(f"Preprocessing {N} samples.")
    print(f"Operation: Center Crop ({TARGET_H}x{TARGET_W}) -> Quantize")

    # Create Memmaps with FIXED 224x224 size
    fp_ref = np.memmap("composite_reflectance.dat", dtype='uint16', mode='w+', shape=(N, 6, TARGET_H, TARGET_W))
    fp_temp = np.memmap("composite_temperature.dat", dtype='uint16', mode='w+', shape=(N, 1, TARGET_H, TARGET_W))

    for i in tqdm(range(N)):
        row = dataframe.iloc[i]
        
        ref_buffer = [] 
        lwir_buffer = []
        
        filename = f"{row.cluster_id}.tif"
        tile_path = os.path.join(root_dir, str(row.country.lower()), str(row.year), filename)
        
        try:
            with rasterio.open(tile_path) as src:
                desc_map = {desc: idx + 1 for idx, desc in enumerate(src.descriptions)}
                
                # Read Reflectance
                season_ref = []
                valid_ref = True
                for b in REF_BANDS:
                    if b in desc_map:
                        season_ref.append(src.read(desc_map[b]))
                    else:
                        valid_ref = False; break
                
                if valid_ref:
                    ref_stack = np.stack(season_ref) 
                    # CROP HERE before buffering (saves memory)
                    ref_stack = center_crop_array(ref_stack, TARGET_H, TARGET_W)
                    ref_buffer.append(ref_stack)

                # Read LWIR
                if LWIR_BAND[0] in desc_map:
                    lwir_data = src.read(desc_map[LWIR_BAND[0]])
                    lwir_data = center_crop_array(lwir_data, TARGET_H, TARGET_W)
                    lwir_buffer.append(lwir_data)
                elif LWIR_BAND[1] in desc_map:
                    lwir_data = src.read(desc_map[LWIR_BAND[1]])
                    lwir_data = center_crop_array(lwir_data, TARGET_H, TARGET_W)
                    lwir_buffer.append(lwir_data)

                else:
                    print(f"Warning: LWIR band not found in {tile_path}")
                    
        except Exception as e:
            print(f"Error processing {tile_path}: {e}")
            continue
        
        # --- AGGREGATE & SAVE ---
        
        # Reflectance
        if len(ref_buffer) > 0:
            ref_all = np.stack(ref_buffer, axis=0)
            ref_all = np.nan_to_num(ref_all)
        else:
            ref_all = np.zeros((6, TARGET_H, TARGET_W), dtype=np.float32)
            
        ref_quant = (ref_all + REF_OFFSET) / REF_SCALE
        fp_ref[i] = np.clip(ref_quant, 0, 65535).astype(np.uint16)

        # Temperature
        if len(lwir_buffer) > 0:
            lwir_all = np.stack(lwir_buffer, axis=0)
            lwir_all = np.nan_to_num(lwir_all)
        else:
            lwir_all = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)
            
        temp_quant = (lwir_all * TEMP_SCALE)
        fp_temp[i] = np.clip(temp_quant, 0, 65535).astype(np.uint16)[None, :, :]
        
        if i % 500 == 0:
            fp_ref.flush(); fp_temp.flush()

    fp_ref.flush()
    fp_temp.flush()
    
    np.save("composite_meta.npy", {
        "shape_ref": (N, 6, TARGET_H, TARGET_W),
        "shape_temp": (N, 1, TARGET_H, TARGET_W),
        "ref_offset": REF_OFFSET, "ref_scale": REF_SCALE, "temp_scale": TEMP_SCALE
    })
    print("Done.")

def create_processed_memmap_seasonal(config):
    config.data.nature = 'seasonal' 
    
    dm = MSDataModule(**config.data, fold=config.run.fold)
    ds_raw = dm.get_dataset(split='all', transform=None)
    dataframe = ds_raw.dataframe
    root_dir = ds_raw.root_dir
    
    N = len(dataframe)
    
    print(f"Preprocessing {N} samples.")
    print(f"Operation: Center Crop ({TARGET_H}x{TARGET_W}) -> Quantize")

    # Create Memmaps with FIXED 224x224 size
    fp_ref = np.memmap("seasonal_reflectance.dat", dtype='uint16', mode='w+', shape=(N, 4, 6, TARGET_H, TARGET_W))
    fp_temp = np.memmap("seasonal_temperature.dat", dtype='uint16', mode='w+', shape=(N, 4, 1, TARGET_H, TARGET_W))

    for i in tqdm(range(N)):
        row = dataframe.iloc[i]
        
        ref_buffer = [] 
        lwir_buffer = []
        
        for season in range(1, 5):
            filename = f"{row.cluster_id}_{season}.tif"
            tile_path = os.path.join(root_dir, str(row.country.lower()), str(row.year), filename)
            
            try:
                with rasterio.open(tile_path) as src:
                    desc_map = {desc: idx + 1 for idx, desc in enumerate(src.descriptions)}
                    
                    # Read Reflectance
                    season_ref = []
                    valid_ref = True
                    for b in REF_BANDS:
                        if b in desc_map:
                            season_ref.append(src.read(desc_map[b]))
                        else:
                            valid_ref = False; break
                    
                    if valid_ref:
                        ref_stack = np.stack(season_ref) 
                        # CROP HERE before buffering (saves memory)
                        ref_stack = center_crop_array(ref_stack, TARGET_H, TARGET_W)
                        ref_buffer.append(ref_stack)

                    # Read LWIR
                    if LWIR_BAND[0] in desc_map:
                        lwir_data = src.read(desc_map[LWIR_BAND[0]])
                        lwir_data = center_crop_array(lwir_data, TARGET_H, TARGET_W)
                        lwir_buffer.append(lwir_data)
                    elif LWIR_BAND[1] in desc_map:
                        lwir_data = src.read(desc_map[LWIR_BAND[1]])
                        lwir_data = center_crop_array(lwir_data, TARGET_H, TARGET_W)
                        lwir_buffer.append(lwir_data)

                    else:
                        print(f"Warning: LWIR band not found in {tile_path}")
                        
            except Exception as e:
                print(f"Error processing {tile_path}: {e}")
                continue
        
        # --- AGGREGATE & SAVE ---

        T = 4

        # Reflectance
        ref_stack = np.zeros((T, 6, TARGET_H, TARGET_W), dtype=np.float32)
        
        if len(ref_buffer) > 0:
            t = min(len(ref_buffer), T)
            ref_stack[:t] = np.stack(ref_buffer[:t], axis=0)
            ref_stack = np.nan_to_num(ref_stack)
        
        ref_quant = (ref_stack + REF_OFFSET) / REF_SCALE
        fp_ref[i] = np.clip(ref_quant, 0, 65535).astype(np.uint16)

        # Temperature
        temp_stack = np.zeros((T, TARGET_H, TARGET_W), dtype=np.float32)
        
        if len(lwir_buffer) > 0:
            t = min(len(lwir_buffer), T)
            temp_stack[:t] = np.stack(lwir_buffer[:t], axis=0)
            temp_stack = np.nan_to_num(temp_stack)
        
        temp_quant = temp_stack * TEMP_SCALE
        fp_temp[i] = np.clip(temp_quant, 0, 65535).astype(np.uint16)[:, None, :, :]
        
        if i % 500 == 0:
            fp_ref.flush()
            fp_temp.flush()

        np.save("seasonal_meta.npy", {
                "shape_ref":  (N, 4, 6, TARGET_H, TARGET_W),
                "shape_temp": (N, 4, 1, TARGET_H, TARGET_W),
                "ref_offset": REF_OFFSET,
                "ref_scale":  REF_SCALE,
                "temp_scale": TEMP_SCALE,
            })

@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def main(cfg: DictConfig) -> None:
    create_processed_memmap_composite(cfg)

if __name__ == "__main__":
    main()
