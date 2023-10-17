import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from utilities import extract_training_data, get_spearman
import plotly.figure_factory as ff
from Nested_CV_reformat import NESTED_CV_reformat

from sklearn.decomposition import PCA
from PIL import Image

########### MAIN ####################################
def tfxn_heatmap(datafile_path, cell_type_list, RLU_floor, prefix, save_path):
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
    plt.figure(figsize=(10,9))
    heatmap = sns.heatmap(round(np.abs(all_corr),2), vmin=.2, vmax=1, cmap='mako_r', annot = True, annot_kws={"size": 18}, cbar = False)
    heatmap.invert_yaxis()
    plt.tick_params(axis='y', which='both', labelsize=18)
    plt.tick_params(axis='x', which='both', labelsize=18)
    cbar = heatmap.figure.colorbar(heatmap.collections[0])
    cbar.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 30)
    cbar.ax.tick_params(labelsize=20)

    plt.gca()
    plt.savefig(save_path + 'All_Data_Tfxn_Heatmap.svg', dpi=600, format = 'svg', transparent=True, bbox_inches='tight')
    plt.close()           
            
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
    fig, ax = plt.subplots(2,3, figsize = (15, 10))
    com_ax = fig.add_subplot(111)
    plt.rcParams["font.family"] = "Arial"


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
        if i ==2:
            show_legend = True
        else:
            show_legend = False
        ax = sns.histplot(data=data, x="RLU_" + cell,multiple="stack", hue="Helper_lipid", hue_order= data.Helper_lipid.unique(), ax = ax, legend=show_legend)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black') 
        ax.set(xlim=(RLU_floor, 12), xticks=np.arange(RLU_floor, 15, 3), ylim=(0, 225), yticks=np.arange(0, 250,25))

        ax.set_yticklabels(ax.get_yticklabels(), size = 12)
        ax.set_xticklabels(ax.get_xticklabels(), size = 12)
        ax.set_title(cell, fontsize = 20)
        ax.set(xlabel=None, ylabel=None)
        # plt.xlabel("ln(RLU)", fontsize=20)
        # plt.ylabel('Counts', fontsize=20)
        
        # plt.setp(ax.get_legend().get_texts(), fontsize='20') # for legend text
        # plt.setp(ax.get_legend().get_title(), fontsize='20') # for legend title
        if show_legend:
            # for legend title
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), fontsize = 20)
            plt.setp(ax.get_legend().get_title(), fontsize='20')


    # Turn off axis lines and ticks of the big subplot
    com_ax.spines['top'].set_color('none')
    com_ax.spines['bottom'].set_color('none')
    com_ax.spines['left'].set_color('none')
    com_ax.spines['right'].set_color('none')
    com_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    # Set common labels
    com_ax.set_xlabel("Transfection Efficiency (ln[RLU])", fontsize=24)
    com_ax.set_ylabel('Counts', fontsize=24)
    fig.suptitle("Distribution of Screening Results by Cell Type and Helper Lipid", fontsize = 30, weight = "bold")

    #Save Transfection Distribution
    plt.savefig(save_path + f'tfxn_dist.svg', dpi = 600, transparent = True, bbox_inches = "tight")
    plt.close()
        # #Histogram plot for all features in training data
        # for param in Train_X.columns:
        #     f = plt.figure()
        #     sns.histplot(Train_X.loc[:,param])
        #     plt.savefig(save_path + f'{cell}_{param}_Dist.png', 
        #                 dpi=600, format = 'png', 
        #                 transparent=True, 
        #                 bbox_inches='tight')
        #     plt.close()

        
        #distribution plot of transfection by helper lipid used

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