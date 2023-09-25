import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from utilities import extract_training_data, select_input_params

import plotly.figure_factory as ff

from Nested_CV_reformat import NESTED_CV_reformat

################ Model Training ##############################################
cell_names = ['B16'] #'ARPE19','N2a',
model_list = ['LGBM', 'XGB', 'RF']
size_zeta = True
PDI = 1
size = 100000
N_CV = 5

################ Global Variables ##############################################
data_file_path = "Raw_Data/Final_Master_Formulas.csv"
save_path = "BMES_Figures/" # Where to save new models, results, and training data

########### MAIN ####################################

def main():
    prefix = "RLU_"
    # #Rerun model training based on clusters
    for cell in cell_names:
        #Features to use
        input_param_names = select_input_params(size_zeta, size_zeta)

        #Get Training Data for cell
        Train_X, Y, data = extract_training_data(data_file_path=data_file_path, 
                                                 input_param_names= input_param_names, 
                                                 cell_type=cell, 
                                                 size_cutoff=size, 
                                                 PDI_cutoff= PDI, 
                                                 prefix=prefix,
                                                 RLU_floor= 0)

        # #Check/create correct save path
        # if os.path.exists(save_path + f'/{cell}') == False:
        #     os.makedirs(save_path + f'/{cell}', 0o666)
        # #Histogram plot for all features in training data
        # for param in Train_X.columns:
        #     f = plt.figure()
        #     sns.histplot(Train_X.loc[:,param])
        #     plt.savefig(save_path + f'/{cell}/{param}_Dist.png', 
        #                 dpi=600, format = 'png', 
        #                 transparent=True, 
        #                 bbox_inches='tight')
        #     plt.close()

        
        #distribution plot of transfection by helper lipid used
        fig, ax = plt.subplots(figsize = (6,8))
        ax = sns.histplot(data=data, x="RLU_" + cell,multiple="stack", hue="Helper_lipid")
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black') 
        ax.set(xlim=(0, 12), xticks=np.arange(0,14,2), ylim=(0, 200), yticks=np.arange(0, 225,25))

        ax.set_yticklabels(ax.get_yticklabels(), size = 15)
        ax.set_xticklabels(ax.get_xticklabels(), size = 15)
        plt.xlabel('RLU_B16', fontsize=20)
        plt.ylabel('Counts', fontsize=20)
        #plt.legend(data.Helper_lipid.unique(), fontsize = 20)
        plt.setp(ax.get_legend().get_texts(), fontsize='20') # for legend text
        plt.setp(ax.get_legend().get_title(), fontsize='20') # for legend title
        plt.savefig(save_path + f'{cell}_tfxn_dist.svg', dpi = 600, transparent = True, bbox_inches = "tight")
if __name__ == "__main__":
    main()