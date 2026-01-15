import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize_scalar
from captum.attr import IntegratedGradients, Saliency


def save_integrated_gradients(model, dataset, best_species, class_indices, output_dir):

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

        for i in range(len(best_species)):
            class_idx = class_indices[i]

            os.makedirs(output_dir / str(best_species[i]), exist_ok=True)

            target = torch.nn.functional.one_hot(torch.tensor(class_idx), num_classes=len(dataset.species)).to(model.device).float().requires_grad_()
            negativetarget = 1 - target
            fulltarget = torch.stack([negativetarget, target], dim=-1).unsqueeze(0)
            attributions = integrated_gradients.attribute(tuple(inputs.values()), target=(class_idx,1))
            attributions_np = attributions[0].cpu().detach().numpy()
            np.save(output_dir / str(best_species[i]) / f'ig_{ind}.npy', attributions_np)



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
        dic[s] = {'pearsonr': pearsonr(fulldf[s], predictions[s])[0],
                  'spearmanr': spearmanr(fulldf[s], predictions[s])[0]}
        
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
