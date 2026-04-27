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
from matplotlib import colors
from matplotlib.patches import Patch

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


def convert_to_png(input_dir, input_file, colormap, species = False,
                   uncertainty_threshold = 0.05, relative_threshold = False):

    date = input_file.split('_')[-1].replace('.tif', '')

    with rasterio.open('/home/gaetan/Downloads/oceans50m.tiff') as src:

        oceans = src.read(1)
        ocean_transform = src.transform

    # Open raster
    with rasterio.open(Path(input_dir) / input_file) as src:

        means = src.read(1).astype(float)
        ci = src.read(2).astype(float)
        transform = src.transform

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
        ret = ax.imshow(means, extent=[left, right, bottom, top], cmap = colormap, norm=colors.Normalize(vmin=np.nanmin(means), vmax=np.nanmax(means)),
                origin='upper', aspect='auto', interpolation='nearest')
        
        # Add uncertainty hatching
        if relative_threshold:
            threshold_name = f"{100*uncertainty_threshold:.0f}%"
            relative_ci = ci / (means + 1e-8)
            ys, xs = np.where(relative_ci > uncertainty_threshold)
        else:
            threshold_name = f"{uncertainty_threshold:.2f}"
            ys, xs = np.where(ci > uncertainty_threshold)


        # Add title and legend
        if species:
            title = r"$\bfit{" + input_file.split('_')[0].replace(' ', '\ ') + "}$"
            
        else:
            title_mapping = {
                'sr': 'Species Richness',
                'bm': 'Biomass',
                'uicn': 'Threatened Species Richness'
            }
            title = r"$\bf{" + title_mapping[input_file.split('_')[1]].replace(' ', '\ ') + "}$"   

        lons = transform.c + (xs + 0.5) * transform.a
        lats = transform.f + (ys + 0.5) * transform.e

        ax.scatter(lons, lats, s=12, c='white',
                marker='.', linewidths=0, zorder=5)


        # Plot colorbar
        cax = fig.add_axes([0, 0, 0.1, 1])
        cax.set_axis_off()  
        cbar = fig.colorbar(ret, ax=cax)
        cbar.ax.tick_params(labelsize=40)
        cbar.ax.tick_params(length=10, width=2)

        # Plot legend
        legendlabel = f"High uncertainty (CI > {threshold_name})"
        handle_certain = Patch(facecolor='steelblue', edgecolor='none', label='Low uncertainty')
        handle_uncertain = Patch(facecolor='steelblue', edgecolor='white',label=legendlabel, hatch='..')

        legend = ax.legend(
            handles=[handle_certain, handle_uncertain],
            loc='lower center',
            fontsize=30,
            framealpha=0.8,
            handleheight=1.2
        )

        # Add date annotation

        

        fig.text(   0.05, 0.95,
                    f"Predictions for\n" + title + f"\non {date}",
                    ha='left', va='top',
                    fontsize=40
                )
        
        

        plt.savefig(Path(input_dir) / input_file.replace('.tif', '.png'), bbox_inches='tight', pad_inches=0)


def load_bootstrap_metrics(output_dir, cp_name, reindex = True, pa_metric = 'F1'):

    ### Load multiple output metrics and return average and confidence intervals

    if 'pa' in cp_name:
        if pa_metric == 'TSS':
            metric = 'TSS'
            metric_key = 'tss'
        else:
            metric = 'f1'
            metric_key = 'testF1'
    elif 'reg' in cp_name:
        metric = 'pearsonr_nz'
        metric_key = 'testR2--pearsonr_nz.4'
    else:
        metric = 'pearsonr'
        metric_key = 'testR2--pearsonr.4'

    if 'xgb' in str(output_dir) and pa_metric == 'F1':
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



def load_bootstrap_pa(output_dir, pred_name, usecols = None):

    # Load P-A predictions and return average and confidence intervals

    parent_folder = Path(output_dir) / pred_name

    df_list = []
    for fo in parent_folder.glob("Seed*"):
        df = pd.read_csv(fo / "predictions-probs.csv", index_col = 0, usecols=usecols)
        df_list.append(df)

    all_df = np.stack([df.values for df in df_list])

    y_mean = pd.DataFrame(all_df.mean(axis=0), index=df_list[0].index, columns=df_list[0].columns)
    y_std  = pd.DataFrame(all_df.std(axis=0), index=df_list[0].index, columns=df_list[0].columns)
    y_sem  = y_std / np.sqrt(len(df_list))  # standard error

    # 95% confidence interval (normal approx): mean ± 1.96 * SEM 
    ci = 1.96 * y_sem

    return(y_mean, ci)