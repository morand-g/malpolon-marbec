import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize_scalar
from captum.attr import IntegratedGradients, Saliency

import rasterio
from rasterio.transform import from_origin
from rasterio.plot import show as rioshow
import matplotlib.pyplot as plt
from matplotlib import colormaps


def save_integrated_gradients(model, dataset, best_species, class_indices, output_dir):

     #################### To rewrite completely ####################

    model.eval()
    integrated_gradients = Saliency(model)
    os.makedirs(output_dir, exist_ok=True)

    #df = dataset._load_observation_data()
    #alltargets = df[dataset.species]

    for i in range(len(dataset)):
        # Get the i-th sample from the dataset

        ind = dataset.survey_ids[i]
        inputs, _ = dataset[i]
        #targets = torch.from_numpy(alltargets.loc[ind].values)

        inputs = {k:v.to(model.device).unsqueeze(0).requires_grad_() for k, v in inputs.items()}
        #targets = targets.to(model.device).float().requires_grad_()

        for j in range(len(best_species)):
            class_idx = class_indices[i]

            os.makedirs(output_dir / str(best_species[j]), exist_ok=True)

            target = torch.nn.functional.one_hot(torch.tensor(class_idx), num_classes=len(dataset.species)).to(model.device).float().requires_grad_()
            negativetarget = 1 - target
            fulltarget = torch.stack([negativetarget, target], dim=-1).unsqueeze(0)
            attributions = integrated_gradients.attribute(tuple(inputs.values()), target=(class_idx,1))
            attributions_np = attributions[0].cpu().detach().numpy()
            np.save(output_dir / str(best_species[j]) / f'ig_{ind}.npy', attributions_np)



def sr_distance(thres, df, target_sr):
    sr = (df > thres).astype(int).sum(axis=1)
    
    return np.abs(sr.mean() - target_sr.mean())




def export_f1_scores(cfg):

    ######### Calculate THRESHOLD ##########

    output_path = Path(cfg.run.checkpoint_path).parent
    tv_predictions = pd.read_csv(output_path / 'predictions-probs-trainval.csv', index_col='survey_id')

    # Load targets

    p = Path(cfg.data.inputs_path) / cfg.data.dataset_name

    fulldf = pd.read_csv(p, index_col='survey_id',
                        dtype = {22:str, 24:str, 25:str})
    targets = fulldf.loc[fulldf['subset'] == 'val', tv_predictions.columns]
    tv_predictions = tv_predictions.loc[targets.index]

    # Test dataset summary
    targets_sr = (targets > 0).sum(axis=1)

    THRESHOLD = minimize_scalar(sr_distance, args=(tv_predictions, targets_sr),method='Bounded', bounds=(0,1))['x']


    ####### Calculate Test F1 scores #########

    predictions = pd.read_csv(output_path / 'predictions-probs.csv', index_col='survey_id')
    targets = fulldf.loc[fulldf['subset'] == 'test', predictions.columns]


    dic = {}
    for s in predictions.columns:
        preds = (predictions[s] > THRESHOLD).astype(float).to_numpy().flatten()
        targ = (targets[s] != 0).astype(float).to_numpy().flatten()
        f1 = f1_score(targ, preds)
        dic[s] = {'f1': f1}
    scores = pd.DataFrame(dic).T


    scores.sort_values(ascending=False, by='f1', inplace = True)
    scores.to_csv(output_path / f"testF1--TH={THRESHOLD:.3f}--.4rank={len(scores[scores['f1']>=0.4])}.csv")



def export_correlation_scores(cfg, classif = False, ordering = 'pearsonr'):

    # Load predictions
    output_path = Path(cfg.run.checkpoint_path).parent
    
    if classif:
        predictions = pd.read_csv(output_path / 'predictions-probs.csv', index_col='survey_id')
    else:
        predictions = pd.read_csv(output_path / 'predictions-biomass.csv', index_col='survey_id')
    
    # Load targets

    p = Path(cfg.data.inputs_path) / cfg.data.dataset_name
    
    try:
        fulldf = pd.read_csv(p, index_col='survey_id', dtype = {22:str, 24:str, 25:str}).loc[predictions.index]
    except IndexError:
        fulldf = pd.read_csv(p, index_col='survey_id').loc[predictions.index]
        
    
    dic = {}
    for s in predictions.columns:
        nonzero = (fulldf[s] > 0) * (predictions[s] > 0)
        dic[s] = {'pearsonr': pearsonr(fulldf[s], predictions[s])[0],
                  'spearmanr': spearmanr(fulldf[s], predictions[s])[0],
                 }
        
        if nonzero.sum() > 1:
            dic[s]['pearsonr_nz'] =  pearsonr(fulldf.loc[nonzero,s], predictions.loc[nonzero,s])[0]
            dic[s]['spearmanr_nz'] =  spearmanr(fulldf.loc[nonzero,s], predictions.loc[nonzero,s])[0]
        
    scores = pd.DataFrame(dic).T
    scores.sort_values(ascending=False, by=ordering, inplace = True)
    scores.to_csv(output_path / f"testR2--{ordering}.4rank={len(scores[scores[ordering]>=0.4])}.csv")

    if classif:

        with open(Path(cfg.data.inputs_path) / cfg.data.dataset_name.replace("binned.csv","bins.txt"), "r") as f:
            medians = f.readlines()
            medians = [float(m.strip()) for m in medians]

        median_dic = {i: medians[i] for i in range(len(medians))}
        median_predictions = predictions.astype(int).replace(median_dic)

        # Load targets

        p = Path(cfg.data.inputs_path) / cfg.data.dataset_name.replace(f"_{cfg.model.num_bins}binned","")
        fulldf = pd.read_csv(p, index_col='survey_id', dtype = {22:str, 24:str, 25:str}).loc[predictions.index]

        dic = {}
        for s in median_predictions.columns:
            dic[s] = {'pearsonr': pearsonr(fulldf[s], median_predictions[s])[0],
                      'spearmanr': spearmanr(fulldf[s], median_predictions[s])[0]}
            
        scores = pd.DataFrame(dic).T
        scores.sort_values(ascending=False, by='pearsonr', inplace = True)
        scores.to_csv(output_path / f"testR2median--.4rank={len(scores[scores['pearsonr']>=0.4])}.csv")



def export_map(predictions_mean, predictions_ci, var_name, out_path, filename):

    RES = 0.1

    min_lat, max_lat = predictions_mean['latitude'].min() - 1 - 0.5*RES, predictions_mean['latitude'].max() + 1 - 0.5*RES
    min_lon, max_lon = predictions_mean['longitude'].min() - 1 - 0.5*RES, predictions_mean['longitude'].max() + 1 - 0.5*RES

    # Create a regular grid over the extent
    grid_lon = np.linspace(min_lon, max_lon, 1+int((max_lon - min_lon) / RES), endpoint=True)
    grid_lat = np.linspace(min_lat, max_lat, 1+int((max_lat - min_lat) / RES), endpoint=True)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    grid_values, grid_ci = np.full(grid_lon.shape, np.nan), np.full(grid_lon.shape, np.nan)

    # Convert point data to grid coordinates
    x = (np.floor((predictions_mean['longitude'].values - min_lon) / RES).astype(int))
    y = (np.ceil((predictions_mean['latitude'].values - min_lat) / RES).astype(int))
    grid_values[y, x] = predictions_mean[var_name].to_numpy()
    grid_ci[y, x] = predictions_ci[var_name].to_numpy()


    # Write raster
    transform = from_origin(min_lon, max_lat, RES, RES)
    dst = rasterio.open(Path(out_path,f'{filename}.tif'), 'w', driver='GTiff',
                        height = grid_values.shape[0], width = grid_values.shape[1],
                        dtype=str(grid_values.dtype),
                        count=2,
                        crs='epsg:4326',
                        transform=transform,
                        nodata=np.nan,
                        compress='lzw')

    dst.write_band(1, np.flipud(grid_values))
    dst.set_band_description(1, f'{var_name}_mean')
    dst.write_band(2, np.flipud(grid_ci))
    dst.set_band_description(2, f'{var_name}_ci')
    dst.close()



def interpolate_color(a, b, color_a: str, color_b: str):

    # Pick a color from a 2d space varying between color a and color b, where a controls the hue interpolation and b controls the saturation.

    def hex_to_hsl(hex_color):
        r, g, bl = (int(hex_color.lstrip('#')[i:i+2], 16) / 255 for i in (0, 2, 4))
        cmax, cmin = max(r, g, bl), min(r, g, bl)
        l = (cmax + cmin) / 2
        d = cmax - cmin
        s = 0 if d == 0 else d / (1 - abs(2*l - 1))
        if d == 0: h = 0
        elif cmax == r: h = 60 * (((g - bl) / d) % 6)
        elif cmax == g: h = 60 * ((bl - r) / d + 2)
        else:           h = 60 * ((r - g)  / d + 4)
        return h, s * 100, l * 100

    hA, sA, lA = hex_to_hsl(color_a)
    hB, sB, lB = hex_to_hsl(color_b)

    dh = ((hB - hA + 540) % 360) - 180
    h = (hA + dh * a) % 360
    s = sA + (sB - sA) * b
    l = lA + (lB - lA) * b

    h, s, l = np.asarray(h), np.asarray(s) / 100, np.asarray(l) / 100
    c = (1 - np.abs(2*l - 1)) * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = l - c / 2
    h6 = h / 60
    r = np.select([h6<1, h6<2, h6<3, h6<4, h6<5], [c, x, 0, 0, x], c)
    g = np.select([h6<1, h6<2, h6<3, h6<4, h6<5], [x, c, c, x, 0], 0)
    b = np.select([h6<1, h6<2, h6<3, h6<4, h6<5], [0, 0, x, c, c], x)

    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    return np.clip(rgb, 0, 1)




def convert_to_png(input_dir, input_file):

    date = input_file.split('_')[-1].replace('.tif', '')

    with rasterio.open('/home/gaetan/Downloads/oceans50m.tiff') as src:

        oceans = src.read(1)
        ocean_transform = src.transform

    # Open raster
    with rasterio.open(Path(input_dir) / input_file) as src:

        means = src.read(1).astype(float)
        ci = src.read(2).astype(float)
        transform = src.transform

        mask = np.isnan(means)

         # Normalize each band to [0, 1]
        def norm(arr, mask):
            valid = arr[~mask]
            mn, mx = valid.min(), valid.max()
            out = (arr - mn) / (mx - mn + 1e-10)
            out[mask] = 0
            return out
        
        means_norm = norm(means, mask)

        #rgba = np.dstack((interpolate_color(means_norm, ci_norm, "#ff381a", "#041a00"), (~mask).astype(float)))
        rgba = np.dstack((colormaps.get_cmap('turbo')(means_norm)[:,:,:3],  (~mask).astype(float)[:,:,np.newaxis]))
        fig = plt.figure(frameon=False, figsize=(40, 40 * oceans.shape[0] / oceans.shape[1]))
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_axis_off()

        # Plot background
        rioshow(oceans, transform=ocean_transform, vmin=0, vmax = 1, cmap='gray', ax = ax)

        # Plot raster
        h, w = means.shape
        left   = transform.c
        top    = transform.f
        right  = left + transform.a * w
        bottom = top  + transform.e * h
        ax.imshow(rgba, extent=[left, right, bottom, top],
                origin='upper', aspect='auto', interpolation='nearest')


        # Plot legend
        legend_size = 32
        legend_img = np.zeros((legend_size, legend_size, 3))
        xv, yv = np.meshgrid(np.linspace(0, 1, legend_size),
                            np.linspace(0, 1, legend_size))
        legend_img = interpolate_color(xv, yv, "#ff381a", "#041a00")

        legend_ax = fig.add_axes([0.01, 0.01, 0.08, 0.08 * oceans.shape[1] / oceans.shape[0]])
        legend_ax.imshow(legend_img, origin='lower', aspect='auto')
        legend_ax.set_xlabel('Mean', fontsize=20, color='white')
        legend_ax.set_ylabel('Confidence interval', fontsize=20, color='white')
        legend_ax.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
        for spine in legend_ax.spines.values():
            spine.set_edgecolor('white')

        
        fig.text(   0.05, 0.95,
                    date,
                    ha='left', va='top',
                    fontsize=40
                )
        
        plt.savefig(Path(input_dir) / input_file.replace('.tif', '.png'), bbox_inches='tight', pad_inches=0)


def load_bootstrap_metrics(output_dir, cp_name, reindex = True):

    ### Load multiple output metrics and return average and confidence intervals

    if 'pa' in cp_name:
        metric = 'f1'
        metric_key = 'testF1'
    elif 'reg' in cp_name:
        metric = 'pearsonr_nz'
        metric_key = 'testR2--pearsonr_nz.4'
    else:
        metric = 'pearsonr'
        metric_key = 'testR2--pearsonr.4'

    if 'xgb' in str(output_dir):
        metric_key = "xgb_best_"
    
    parent_folder = Path(output_dir) / cp_name

    df_list = []
    for fo in parent_folder.glob("Seed*"):
        df = pd.read_csv(next(fo.glob(metric_key + "*.csv")), index_col = 0)
        if 'eco' not in cp_name and reindex:
            dg = pd.Series(df.iloc[:400][metric], name = fo.name)
            dg = dg.reset_index(drop=True)
        else:
            dg = pd.Series(df[metric], name = fo.name)
        df_list.append(dg)


    all_df = pd.concat(df_list, axis = 1).T

    y_mean = all_df.mean(axis=0)
    y_std  = all_df.std(axis=0)
    y_sem  = y_std / np.sqrt(len(df_list))  # standard error

    # 95% confidence interval (normal approx): mean ± 1.96 * SEM 
    ci = 1.96 * y_sem

    return(y_mean, ci)



def load_bootstrap_sr_metrics(output_dir, cp_name, uicn = False):

    # Calculate SR from P-A predictions and return average and confidence intervals

    parent_folder = Path(output_dir) / cp_name

    if uicn:
        traits = pd.read_csv('/data/data/RLS/traits.csv', index_col=0)
        

    df_list = []
    for fo in parent_folder.glob("Seed*"):
        filename = next(fo.glob("presences*.csv"))
        df = pd.read_csv(filename, index_col = 0)
        THRESHOLD = float(filename.stem.split('--TH=')[-1])
        sr = (df > THRESHOLD).astype(int).sum(axis=1)
        if uicn:
            species= df.columns
            uicnL_status_dict = {s: traits['IUCN_inferred_Loiseau23'].get(s, 'No Status') for s in species}
            uicnL_threatened = pd.Series([(uicnL_status_dict[s] == 'Threatened')*1 for s in species], index = species, name = 'Threatened')
            df = df * uicnL_threatened
            sr = (df > THRESHOLD).astype(int).sum(axis=1)
        df_list.append(np.log(1+sr))

    
    all_df = pd.concat(df_list, axis = 1).T

    y_mean = all_df.mean(axis=0)
    y_std  = all_df.std(axis=0)
    y_sem  = y_std / np.sqrt(len(df_list))  # standard error

    # 95% confidence interval (normal approx): mean ± 1.96 * SEM 
    ci = 1.96 * y_sem

    return(y_mean, ci)
    


def load_bootstrap_bm_metrics(output_dir, cp_name):

    ### Calculate total biomass from predictions and return average and confidence intervals

    bm_maxima = pd.read_csv('/marbec-data/RLS-Australia/malpolon/inputs/australia/biomass_lognorm_maxima.csv', index_col=0)['0']
    parent_folder = Path(output_dir) / cp_name

    df_list = []
    for fo in parent_folder.glob("Seed*"):
        df = pd.read_csv(fo / "predictions-biomass.csv", index_col = 0)
        actual_bm = np.exp(bm_maxima*df) - 1
        log_total_bm = np.log(1 + actual_bm.sum(axis=1))
        df_list.append(log_total_bm)

    all_df = pd.concat(df_list, axis = 1).T

    y_mean = all_df.mean(axis=0)
    y_std  = all_df.std(axis=0)
    y_sem  = y_std / np.sqrt(len(df_list))  # standard error

    # 95% confidence interval (normal approx): mean ± 1.96 * SEM 
    ci = 1.96 * y_sem

    return(y_mean, ci)





def load_bootstrap_eco(output_dir, pred_name, cp_name):

    # Load ecological indicators predictions and return average and confidence intervals

    parent_folder = Path(output_dir) / pred_name

    eco_maxima = pd.read_csv('/marbec-data/RLS-Australia/malpolon/inputs/australia/eco_indic_maxima.csv', index_col=0)

    df_list = []
    for fo in parent_folder.glob("Seed*"):
        df = pd.read_csv(fo / "predictions-biomass.csv", index_col = 0)
        actual_eco = np.exp(eco_maxima.loc[df.columns].T.values * df) - 1

        corrections = pd.read_csv(Path(output_dir) / cp_name / fo.name / "corrections.csv", index_col = 0)
        corrected_eco = actual_eco * corrections['slope'] + corrections['intercept']
        df_list.append(corrected_eco.clip(0, None))

    all_df = np.stack([df.values for df in df_list])

    y_mean = pd.DataFrame(all_df.mean(axis=0), index=df_list[0].index, columns=df_list[0].columns)
    y_std  = pd.DataFrame(all_df.std(axis=0), index=df_list[0].index, columns=df_list[0].columns)
    y_sem  = y_std / np.sqrt(len(df_list))  # standard error

    # 95% confidence interval (normal approx): mean ± 1.96 * SEM 
    ci = 1.96 * y_sem

    return(y_mean, ci)