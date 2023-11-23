import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from utilities import extract_training_data, get_spearman, truncate_colormap
import plotly.figure_factory as ff

from sklearn.decomposition import PCA
from PIL import Image

########### MAIN ####################################
def tfxn_heatmap(datafile_path, cell_type_list, RLU_floor, prefix, save_path):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12
    
    df = pd.read_csv(datafile_path)
    #print(df)
    training_data = df.loc[:,df.columns.isin(prefix + x for x  in cell_type_list)] #Take input parameters and associated values
    training_data.head()
    training_data["Helper_lipid"] = df.loc[:,"Helper_lipid"]
    
    #floor RLU
    for cell in cell_type_list:
        training_data.loc[training_data[prefix + cell] < RLU_floor, prefix + cell] = RLU_floor 

    #Initiate a correlation matrix of zeros
    all_corr = pd.DataFrame(np.zeros((len(cell_type_list),len(cell_type_list))), index = cell_type_list, columns = cell_type_list)
    lipid_corr = pd.DataFrame(np.zeros((len(cell_type_list),len(cell_type_list))), index = cell_type_list, columns = cell_type_list)



    #Get correlation between all cell types
    for cell1 in cell_type_list:
        for cell2 in cell_type_list:
            all_corr.loc[cell1, cell2] = get_spearman(training_data, cell1, cell2)
    fig = plt.subplot()
    plt.figure(figsize=(3,3))
    heatmap = sns.heatmap(round(np.abs(all_corr),2), vmin=.2, vmax=1, cmap='Blues', annot = True, annot_kws={"size": 6}, cbar = False)
    heatmap.invert_yaxis()
    plt.tick_params(axis='y', which='both', labelsize=8)
    plt.tick_params(axis='x', which='both', labelsize=8)
    cbar = heatmap.figure.colorbar(heatmap.collections[0])
    cbar.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 4)
    cbar.ax.tick_params(labelsize=6)

    plt.gca()
    plt.savefig(save_path + 'All_Data_Tfxn_Heatmap.svg', dpi=600, format = 'svg', transparent=True, bbox_inches='tight')
    plt.close()           


    #### PLOTTING THE HEATMAP FOR LIPID SPECIFIC FORMULATIONS ACROSS CELL TYPES        
    # #Iterate through subsets based on helper lipid used in formulations
    # print(training_data["Helper_lipid"].unique())
    # for lipid in training_data["Helper_lipid"].unique():
    #     lipid_data = training_data.loc[training_data["Helper_lipid"] == lipid]
    #     for cell1 in cell_type_list:
    #         for cell2 in cell_type_list:
    #             lipid_corr.loc[cell1, cell2] = get_spearman(lipid_data, cell1, cell2)
  
            
    #     sns.heatmap(lipid_corr,vmin=-0.2, vmax=1, annot = True)
    #     plt.gca()
    #     plt.title(lipid)
    #     plt.savefig(save_path + f'{lipid}_Heatmap.svg', dpi=600, format = 'svg', transparent=True, bbox_inches='tight')
    #     plt.close()      
    #     # Save the all correlation data to csv
    #     with open(save_path + f'{lipid}_tfxn_correlation.csv', 'w', encoding = 'utf-8-sig') as file:
    #         lipid_corr.to_csv(file)


    # Save the all correlation data to csv
    with open(save_path + 'All_tfxn_correlation.csv', 'w', encoding = 'utf-8-sig') as file:
          all_corr.to_csv(file)

def plot_tfxn_dist(cell_type_list, data_file_path, input_param_names, size, PDI, keep_PDI, prefix, RLU_floor, save_path):
#   Create Subplots
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(2,3, sharex = True, sharey=True, figsize = (4, 3))

    #limits
    # plt.ylim(0, 215)
    # plt.xlim(0, 12)

    #loop through subplots
    for i, ax in enumerate(ax.flatten()):


        #Get Training Data for cell
        cell = cell_type_list[i]
        Train_X, Y, data = extract_training_data(data_file_path=data_file_path, 
                                                 input_param_names= input_param_names, 
                                                 cell_type=cell, 
                                                 size_cutoff=size, 
                                                 PDI_cutoff= PDI,
                                                 keep_PDI = keep_PDI, 
                                                 prefix=prefix,
                                                 RLU_floor= RLU_floor)


        sns.set_palette("husl", 6)
        sns.set(font_scale = 1)
        if i ==1:
            show_legend = True
        else:
            show_legend = False
        
        ax = sns.histplot(data=data, x="RLU_" + cell,
                          multiple="stack", 
                          hue="Helper_lipid", 
                          binwidth = 0.5,
                          hue_order= data.Helper_lipid.unique(), 
                          ax = ax, 
                          legend=show_legend,
                          line_kws={'linewidth': 2},
                          edgecolor='white') 
        # ax.set_yticks(np.arange(0, 300,50), fontsize = 6)
        ax.set_xticks(np.arange(RLU_floor, 15, 3), fontsize = 6)

        if i in [3, 4, 5]:
            ax.set_xlabel('ln(RLU)')


        ax.text(0.5, 0.85, cell, transform=ax.transAxes,
        fontsize=8, ha='center')

        if show_legend:
            sns.move_legend(
                ax, "lower center",
                bbox_to_anchor=(.5, 1), 
                ncol=3, 
                title='Helper Lipid Choice', 
                frameon=False)
            plt.setp(ax.get_legend().get_texts(), fontsize='8')
            plt.setp(ax.get_legend().get_title(), fontsize='8') 

    #Save Transfection Distribution
    plt.savefig(save_path + f'tfxn_dist.png', dpi = 600, transparent = True, bbox_inches = "tight")
    plt.close()

def tfxn_clustering(X, Y, input_params, figure_save_path, cell):


    shap_pca50 = PCA(n_components=2).fit_transform(X)
    #shap_embedded = TSNE(n_components=2, perplexity=50, random_state = 0).fit_transform(shap_values.values)

    # Dimensionality Reduction: SHAP values on Transfection Efficiency
    f = plt.figure(figsize=(5,5))
    plt.scatter(shap_pca50[:,0],
            shap_pca50[:,1],
            c=Y.values,
            linewidth=0.1, alpha=1.0, edgecolor='black', cmap='YlOrBr')
    plt.title('Dimensionality Reduction: SHAP values on Transfection Efficiency in ' + cell)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    cb = plt.colorbar(label=None, aspect=40, orientation="horizontal")
    cb.set_alpha(1)
    cb.draw_all()
    cb.outline.set_linewidth(0.2)
    cb.ax.tick_params('x', length=2)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.get_title()
    cb.ax.set_title(label="Normalized Transfection Efficiency", fontsize=10, color="black", weight="bold")
    cb.set_ticks([0.1, 0.9])
    cb.set_ticklabels(['Low', 'High'])
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(), fontsize=10, color="black", weight="bold")

    plt.gca().axis("off") # no axis

    # # y-axis limits and interval
    # plt.gca().set(ylim=(-30, 30), yticks=np.arange(-40, 40, 10))
    # plt.gca().set(xlim=(-30, 30), xticks=np.arange(-40, 40, 10))

    plt.savefig(figure_save_path + f'{cell}_clustered_tfxn.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
    plt.close()

def main(cell_type_list, data_file_path, input_param_names, save_path, size, PDI,keep_PDI, RLU_floor, prefix):

    #Check/create correct save path
    if os.path.exists(save_path) == False:
        os.makedirs(save_path, 0o666)


    plot_tfxn_dist(cell_type_list=cell_type_list,
                   data_file_path=data_file_path,
                   input_param_names=input_param_names,
                   size = size,
                   PDI = PDI,
                   keep_PDI = keep_PDI,
                   prefix=prefix,
                   RLU_floor=RLU_floor,
                   save_path=save_path)
    

    tfxn_heatmap(data_file_path, cell_type_list, RLU_floor, prefix, save_path)

    # for cell in cell_type_list:
    #     X,Y, cell_data = extract_training_data(data_file_path=data_file_path,
    #                                            input_param_names=input_param_names,
    #                                            cell_type=cell,
    #                                            size_cutoff=size,
    #                                            PDI_cutoff=PDI,
    #                                            prefix=prefix,
    #                                            RLU_floor=RLU_floor)
    #     tfxn_clustering(X=X,
    #                     Y=Y,
    #                     input_params=input_param_names,
    #                     figure_save_path=save_path,
    #                     cell=cell)



if __name__ == "__main__":
    main()