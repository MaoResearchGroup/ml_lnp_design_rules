import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from utilities import get_mean_shap
import os
from collections import defaultdict
from scipy import interpolate
# Function to create bump plots
def bumpplot(dataframe, lw, save_path, feature_order = None):

    
    #sort feature order
    if feature_order == None:
        sorted_index = dataframe.index
    else:
        sorted_index = []
        for feature in feature_order:
            if feature in dataframe["Feature"].tolist():
                sorted_index.append(dataframe.index[dataframe["Feature"] == feature][0])

    sorted_df = dataframe.reindex(sorted_index).reset_index(drop = True)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    #Define feature colors
    feature_colors = {feature: sns.color_palette("husl", n_colors=dataframe.shape[0])[i]
                  for i, feature in enumerate(dataframe.index)}
    
    sorted_df.insert(1,'Features', dataframe.index/dataframe.index.max())
    sorted_df.fillna(-0.1, inplace = True)


    #Plot line for each feature
    for i in sorted_df.index:
        x = np.arange(sorted_df.shape[1]-1)
        y = sorted_df.iloc[i,1:].values
        color = feature_colors[i]
        x, y = add_widths(x, y)
        xs = np.linspace(0, x[-1], num=1024)
        plt.plot(xs, interpolate.PchipInterpolator(x, y)(xs), color=color, linewidth=lw, alpha = 0.75)

    #Setting y-ticks to values from the 'Feature' column
    feature_names = sorted_df['Feature'].unique()
    plt.yticks(sorted_df.Features, feature_names,rotation=45, ha='right', fontsize = 12)

    #Cell Names
    cell_names = sorted_df.columns[1:]

    #removing extra characters
    cell_names = [cell.replace('_Norm_Feature_Value', '') for cell in cell_names]

    plt.xticks(np.arange(len(cell_names)), cell_names, fontsize = 12)

    plt.title('Design Parameters for Cell Type-Specific LNP Formulations', fontsize = 20, weight = 'bold')
    
    sns.despine(left=True, bottom=True)  # Updated to keep the left axis
    ax.set_ylim(-0.20,1.05)
    ylim = ax.get_ylim()
    ax.set_xlim(-0.1,len(cell_names)-0.5)
    plt.grid(axis = 'x')


    # Adding right axis labels
    magnitude_labels = ['N/A', 'Low', 'Med', 'High']
    region_ticks = np.linspace(0,1,len(magnitude_labels)).tolist()
    region_ticks[-1] = ylim[1]
    region_ticks[0] = -0.05
    region_ticks.insert(0, ylim[0])


    magnitude_ticks = []
    for i in range(1,len(region_ticks)):
        magnitude_ticks.append((region_ticks[i]+region_ticks[i-1])/2)
    ax2 = ax.twinx() 
    ax2.set_ylabel('Relative Feature Value', color = 'black', fontsize = 15) 
    ax2.tick_params(axis ='y', labelcolor = 'black') 
    ax2.set_yticks(magnitude_ticks, magnitude_labels)
    
    ax2.set_ylim(ylim)

    magnitude_colors = sns.light_palette("seagreen", len(region_ticks)-1)
    magnitude_colors.insert(0,'lightcoral')


    # Calculate regions
    regions = [
        (region_ticks[0], region_ticks[1]),
        (region_ticks[1], region_ticks[2]),
        (region_ticks[2], region_ticks[3]),
        (region_ticks[3], region_ticks[4])
        ]

    # Color the regions
    for (start, end), color in zip(regions, magnitude_colors):
        ax2.axhspan(start, end, facecolor=color, alpha=0.25)

    plt.savefig(save_path, dpi = 600, transparent = True, bbox_inches = "tight")


# Function to add widths
def add_widths(x, y, width=0.1):
    new_x = []
    new_y = []
    for i,j in zip(x,y):
        new_x += [i-width, i, i+width]
        new_y += [j, j, j]
    return new_x, new_y


def main(cell_model_list, model_folder, shap_value_path, plot_save_path,  N_bins, feature_order):
    #Check/create correct save path
    if os.path.exists(plot_save_path) == False:
        os.makedirs(plot_save_path, 0o666)
    combined_best_values = pd.DataFrame()
    for cell_model in cell_model_list:
        cell = cell_model[0]
        model = cell_model[1]
        best_values = get_mean_shap(c = cell,
                                    model = model,
                                    model_save_path= model_folder,
                                    shap_value_path=shap_value_path,
                                    N_bins = N_bins)

        if combined_best_values.empty:
            combined_best_values = best_values
        else:
            combined_best_values = pd.merge(combined_best_values, best_values, on='Feature', how='outer')

    #Save to SHAP Values folder
    combined_best_values.to_csv(shap_value_path + f'combined_final_max_avg_shap.csv', index=False)

    #Save with figure
    combined_best_values.to_csv(plot_save_path + f'combined_final_max_avg_shap.csv', index=False)

    bumpplot(combined_best_values, 
             lw=5, 
             save_path=plot_save_path+f'Bump_Plot.svg', 
             feature_order= feature_order)



if __name__ == "__main__":
    main()