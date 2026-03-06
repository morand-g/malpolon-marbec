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



def export_correlation_scores(cfg, classif = False):

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
    scores.sort_values(ascending=False, by='pearsonr', inplace = True)
    scores.to_csv(output_path / f"testR2--.4rank={len(scores[scores['pearsonr']>=0.4])}.csv")

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



def export_map(predictions, var_name, out_path, filename):

    RES = 0.1

    min_lat, max_lat = predictions['latitude'].min() - 1 - 0.5*RES, predictions['latitude'].max() + 1 - 0.5*RES
    min_lon, max_lon = predictions['longitude'].min() - 1 - 0.5*RES, predictions['longitude'].max() + 1 - 0.5*RES

    # Create a regular grid over the extent
    grid_lon = np.linspace(min_lon, max_lon, 1+int((max_lon - min_lon) / RES), endpoint=True)
    grid_lat = np.linspace(min_lat, max_lat, 1+int((max_lat - min_lat) / RES), endpoint=True)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

    grid_values = np.full(grid_lon.shape, np.nan) 

    # Convert point data to grid coordinates
    x = (np.floor((predictions['longitude'].values - min_lon) / RES).astype(int))
    y = (np.ceil((predictions['latitude'].values - min_lat) / RES).astype(int))
    grid_values[y, x] = predictions[var_name].to_numpy()
    data = np.floor(255*np.flipud(grid_values))


    # Write raster
    transform = from_origin(min_lon, max_lat, RES, RES)
    dst = rasterio.open(Path(out_path,f'{filename}.tif'), 'w', driver='GTiff',
                        height = data.shape[0], width = data.shape[1],
                        dtype=str(data.dtype),
                        count=1,
                        crs='epsg:4326',
                        transform=transform,
                        nodata=np.nan,
                        compress='lzw')

    dst.write(data, indexes=1)
    dst.close()



def convert_to_png(input_dir, input_file):

    date = input_file.split('_')[-1].replace('.tif', '')

    with rasterio.open('/home/gaetan/Downloads/oceans50m.tiff') as src:

        oceans = src.read(1)
        ocean_transform = src.transform

    # Open raster
    with rasterio.open(Path(input_dir) / input_file) as src:

        fig = plt.figure(frameon=False, figsize=(40, 40 * oceans.shape[0] / oceans.shape[1]))
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_axis_off()

        # Plot background
        rioshow(oceans, transform=ocean_transform, vmin=0, vmax = 1, cmap='gray', ax = ax)

        # Plot raster
        ret = rioshow(src, ax=ax,cmap='turbo')

        im = ret.get_images()[-1]
        cax = fig.add_axes([0, 0, 0.1, 1])
        cax.set_axis_off()  
        cbar = fig.colorbar(im, ax=cax)
        cbar.ax.tick_params(labelsize=40)
        cbar.ax.tick_params(length=10, width=2)
        
        fig.text(   0.05, 0.95,
                    date,
                    ha='left', va='top',
                    fontsize=40
                )
        
        plt.savefig(Path(input_dir) / input_file.replace('.tif', '.png'), bbox_inches='tight', pad_inches=0)
