import numpy as np
import pandas as pd
from pathlib import Path

import rasterio
from rasterio.transform import from_origin
from rasterio.plot import show as rioshow
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
import plotly.express as px

import geopandas as gpd

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
                   uncertainty_threshold = 0.05, relative_threshold = False, crop = None, hatch_color='black', hatch_size=20, title_coords = (0.05, 0.95)):

    date = input_file.split('_')[-1].replace('.tif', '')

    with rasterio.open('/home/gaetan/Downloads/oceans50m.tiff') as src:

        oceans = src.read(1)
        ocean_transform = src.transform

    # Open raster
    with rasterio.open(Path(input_dir) / input_file) as src:

        means = src.read(1).astype(float)
        ci = src.read(2).astype(float)
        transform = src.transform

        # Calculate extent of the raster
        h, w = means.shape
        left   = transform.c
        top    = transform.f
        right  = left + transform.a * w
        bottom = top  + transform.e * h

        # Apply percentage crop if provided
        if crop is not None:
            crop_left   = left   + crop['left'] * (right - left)
            crop_right  = left   + crop['right'] * (right - left)
            crop_bottom = bottom + crop['bottom'] * (top - bottom)
            crop_top    = bottom + crop['top'] * (top - bottom)
        else:
            crop_left, crop_right, crop_bottom, crop_top = left, right, bottom, top
            
        crop_w = crop_right - crop_left
        crop_h = crop_top - crop_bottom
        fig = plt.figure(frameon=False, figsize=(40, 40 * crop_h / crop_w))
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_axis_off()

        # Plot background
        rioshow(oceans, transform=ocean_transform, vmin=0, vmax = 1, cmap='gray', ax = ax)

        # Plot raster
        ret = ax.imshow(means, extent=[left, right, bottom, top], cmap = colormap, norm=colors.Normalize(vmin=np.nanmin(means), vmax=np.nanmax(means)),
                origin='upper', aspect='auto', interpolation='nearest')
        
        # Add uncertainty hatching
        if relative_threshold:
            threshold_name = f"{100*uncertainty_threshold:.0f}%"
            relative_ci = ci / (means + 1e-8)
            ys, xs = np.where(relative_ci > uncertainty_threshold)
        else:
            threshold_name = f"{uncertainty_threshold:.3f}"
            ys, xs = np.where(ci > uncertainty_threshold)

        lons = transform.c + (xs + 0.5) * transform.a
        lats = transform.f + (ys + 0.5) * transform.e

        ax.scatter(lons, lats, s=hatch_size, c=hatch_color,
                marker='.', linewidths=0, zorder=5)

        # Crop the plot to the specified extent
        ax.set_xlim(crop_left, crop_right)
        ax.set_ylim(crop_bottom, crop_top)

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


        # Plot colorbar
        cax = fig.add_axes([0, 0, 0.1, 1])
        cax.set_axis_off()  
        cbar = fig.colorbar(ret, ax=cax)
        cbar.ax.tick_params(labelsize=40)
        cbar.ax.tick_params(length=10, width=2)

        # Plot legend
        legendlabel = f"High uncertainty (CI > {threshold_name})"
        handle_certain = Patch(facecolor='steelblue', edgecolor='none', label='Low uncertainty')
        handle_uncertain = Patch(facecolor='steelblue', edgecolor=hatch_color,label=legendlabel, hatch='..')

        legend = ax.legend(
            handles=[handle_certain, handle_uncertain],
            loc='lower center',
            fontsize=30,
            framealpha=0.8,
            handleheight=1.2
        )

        # Add date annotation

        fig.text(   title_coords[0], title_coords[1],
                    f"Predictions for\n" + title + f"\non {date}",
                    ha='left', va='top',
                    fontsize=40
                )
        
        

        plt.savefig(Path(input_dir) / input_file.replace('.tif', '.png'), bbox_inches='tight', pad_inches=0)





def blindspots_map(input_dir, input_file, crop = None, cmap = 'turbo'):

    date = input_file.split('_')[-1].replace('.tif', '')

    def treat_color(c, opacity = 1):
        return px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) + (opacity,)

    with rasterio.open('/home/gaetan/Downloads/oceans50m.tiff') as src:

        oceans = src.read(1)
        ocean_transform = src.transform

    # Open raster
    with rasterio.open(Path(input_dir) / input_file) as src:

        means = np.floor(src.read(1).astype(float))
        transform = src.transform

        # Calculate extent of the raster
        h, w = means.shape
        left   = transform.c
        top    = transform.f
        right  = left + transform.a * w
        bottom = top  + transform.e * h

        crop_left   = left   + crop['left'] * (right - left)
        crop_right  = left   + crop['right'] * (right - left)
        crop_bottom = bottom + crop['bottom'] * (top - bottom)
        crop_top    = bottom + crop['top'] * (top - bottom)
        crop_w = crop_right - crop_left
        crop_h = crop_top - crop_bottom

        print(f"Crop extent: left={crop_left}, right={crop_right}, bottom={crop_bottom}, top={crop_top}")
        fig = plt.figure(frameon=False, figsize=(10, 10 * crop_h / crop_w))
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_axis_off()

        # Plot background
        rioshow(oceans, transform=ocean_transform, vmin=0, vmax = 1, cmap='gray', ax = ax)

        masked = np.ma.masked_where(means <= 2, means)

        custom_cmap = colors.ListedColormap([treat_color(px.colors.qualitative.Pastel[2])])
        custom_cmap.set_bad(alpha=0)

        # Plot raster
        ret = ax.imshow(masked, extent=[left, right, bottom, top], cmap = custom_cmap, norm=colors.Normalize(vmin=np.nanmin(means), vmax=np.nanmax(means)),
                origin='upper', aspect='auto', interpolation='nearest')
    


        gdf = gpd.read_file("/home/gaetan/Downloads/aus_highly_protected.gpkg")
        gdf.plot(
            ax=ax,
            facecolor=treat_color(px.colors.qualitative.Pastel[0], opacity=0.5),
            edgecolor='black',
            linewidth=0.5,
            zorder=10
        )

        ax.set_xlim(crop_left, crop_right)
        ax.set_ylim(crop_bottom, crop_top)
    

        # Plot legend
    
        handles = [Patch(facecolor=treat_color(px.colors.qualitative.Pastel[0], opacity=0.5), edgecolor='black', label="Highly protected areas"),
                   Patch(facecolor=treat_color(px.colors.qualitative.Pastel[2]), edgecolor='none', label="Blindspots (2+ threatened species)")]


        legend = ax.legend(
            handles=handles,
            title="",
            loc="lower left",
            fontsize=18,
            title_fontsize=20,
            framealpha=0.9
        )
        
        
        plt.show()



def load_bootstrap_metrics(output_dir, cp_name, reindex = True, pa_metric = 'F1'):

    ### Load multiple output metrics and return average and confidence intervals

    if 'pa' in cp_name:
        if pa_metric == 'TSS':
            metric = 'TSS'
            metric_key = 'tss'
        elif pa_metric == 'Boyce':
            metric = 'boyce'
            metric_key = 'boyce'
        elif pa_metric == 'AUC':
            metric = 'auc'
            metric_key = 'auc'
        elif pa_metric == 'maxTSS':
            metric = 'maxTSS'
            metric_key = 'maxtss'
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


def get_traits(species, traits_path = Path('/data/data/RLS/')):

    output = pd.DataFrame(index = species)

    ####################### Usual traits #############################

    traits = pd.read_csv(traits_path / 'traits.csv', index_col=0)

    class_dict = {s: traits['class'].get(s, 'Unknown') for s in species}
    output['class'] = pd.Series(class_dict, index = species, name = 'class').fillna('Unknown')

    troph_level_dict = {s: traits['Troph'].get(s, np.nan) for s in species}
    output['troph_level'] = pd.Series(troph_level_dict, index = species, name = 'troph_level').fillna(np.nan)
    output['troph_level_cat'] = pd.cut(output['troph_level'], bins=[1, 2, 3, 4, 5], labels=['1-2', '2-3', '3-4', '4-5'])

    troph_guild_dict = {s: traits['trophic_guild'].get(s, 'Unknown') for s in species}
    output['trophic_guild'] = pd.Series(troph_guild_dict, index = species, name = 'trophic_guild').fillna('Unknown')

    importance_dict = {s: traits['Importance'].get(s, np.nan) for s in species}
    output['importance'] = pd.Series(importance_dict, index = species, name = 'importance').fillna('Unknown')

    dempel_dict = {s: traits['DemersPelag'].get(s, np.nan) for s in species}
    output['dempel'] = pd.Series(dempel_dict, index = species, name = 'dempel').fillna('Unknown')

    climvuln_dict = {s: traits['ClimVuln_SSP585'].get(s, np.nan) for s in species}
    climvuln = pd.Series(climvuln_dict, index = species, name = 'climvuln').fillna(np.nan)
    output['climvuln'] = pd.cut(climvuln, bins=[0.3, 0.45, 0.5, 0.55, 7], labels=['0.3-0.45', '0.45-0.5', '0.5-0.55', '0.55-7'])

    range_dict = {s: traits['geographic_range_Albouy19'].get(s, np.nan) for s in species}
    range_series = pd.Series(range_dict, index = species, name = 'range')
    output['range'] = pd.cut(range_series, bins=[0, 250, 500, 1000, 1500, 2000, 3500], labels=['0-250', '250-500', '500-1000', '1000-1500', '1500-2000', '2000+'])

    depth_dict = {s: traits['depth_max'].get(s, np.nan) for s in species}
    depth_series = pd.Series(depth_dict, index = species, name = 'depth_max')
    output['depth_max'] = pd.cut(depth_series, bins=[0, 50, 100, 200, 500, 5000], labels=['0-50', '50-100', '100-200', '200-500', '500+'])

    uicn_status_dict = {s: traits['IUCN_category'].get(s, 'No Status') for s in species}
    output['uicn_status'] = pd.Series(uicn_status_dict, index = species, name = 'uicn_status').fillna('No Status')

    uicnL_status_dict = {s: traits['IUCN_inferred_Loiseau23'].get(s, 'No Status') for s in species}
    output['uicnL_status'] = pd.Series(uicnL_status_dict, index = species, name = 'uicnL_status').fillna('No Status')
    output['uicnL_threatened'] = pd.Series([(uicnL_status_dict[s] == 'Threatened')*1 for s in species], index = species, name = 'Threatened')



    ##################### Edgar 2023 vulnerability ##########################

    edgar_traits = pd.read_csv(traits_path / 'especes_menacees_edgar.csv', index_col='species_name')
    edgar_dict = {s: edgar_traits['category_edgar'].get(s, 'Non Threatened') for s in species}
    output['edgar_cat'] = pd.Series(edgar_dict, index = species, name = 'edgar_category').fillna('Non Threatened')

    output['edgar_cat_threatened'] = output['edgar_cat'].copy()
    output.loc[output['edgar_cat_threatened'] != 'Non Threatened', 'edgar_cat_threatened'] = 'Threatened'

    #################### Boyce 2022 vulnerability ##########################

    boyce_vuln = pd.read_csv(traits_path / 'Boyce_etal_2022_NATCC.csv', index_col='SPname')
    boyce_vuln585 = boyce_vuln[boyce_vuln['Experiment'] == 'SSP585']
    boyce_dict = {s: boyce_vuln585['ClimRisk'].get(s, 'Unknown') for s in species}
    output['boyce_climvuln'] = pd.Series(boyce_dict, index = species, name = 'boyce_climvuln').fillna('Unknown')


    return output