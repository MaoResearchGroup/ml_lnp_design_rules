import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from copy import deepcopy

from Nested_CV_reformat import NESTED_CV_reformat

def extract_training_data(data_path, input_params, cell, prefix, size_zeta, size_cutoff, PDI_cutoff):
    #Load Training data
    df = pd.read_csv(data_path)
    #Remove unnecessary columns
    cell_data = df[['Formula label', 'Helper_lipid'] + input_params + ["PDI"] + [prefix + cell]]
    cell_data = cell_data.dropna() #Remove any NaN rows
    if size_zeta == True:
        cell_data = cell_data[cell_data.Size != 0] #Remove any rows where size = 0
        cell_data = cell_data[cell_data.Size <= size_cutoff]
        cell_data = cell_data[cell_data.Zeta != 0] #Remove any rows where zeta = 0
        cell_data = cell_data[cell_data.PDI <= PDI_cutoff] #Remove any rows where PDI > CUTOFF
        #Remove PDI column from input features
        cell_data.drop(columns = 'PDI', inplace = True)

    cell_data.loc[cell_data[prefix + cell] < 3, prefix + cell] = 3 #replace all RLU values below 3 to 3

    print("Input Parameters used:", input_params)
    print("Number of Datapoints used:", len(cell_data.index))

    X = cell_data[input_params]                         
    Y = cell_data[prefix + cell].to_numpy()
    scaler = MinMaxScaler().fit(Y.reshape(-1,1))
    temp_Y = scaler.transform(Y.reshape(-1,1))
    Y = pd.DataFrame(temp_Y, columns = [prefix + cell])
    return X, Y


################ Model Training ##############################################
cell_names = ['ARPE19','N2a','PC3','B16','HEK293','HepG2'] #'ARPE19','N2a',
model_list = ['LGBM', 'XGB', 'RF']
size_zeta = True
PDI = 1
size = 100000
N_CV = 5

################ Global Variables ##############################################
data_file_path = "Raw_Data/10_Master_Formulas.csv"
save_path = "Figures/Features/Size_all_Zeta_PDI_1/" # Where to save new models, results, and training data

########### MAIN ####################################

def main():

    # #Rerun model training based on clusters
    for cell in cell_names:
        #Features to use
        formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                      'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio']
        
        if cell in ['ARPE19','N2a']:
            #Total_Carbon_Tails Removed (It does not change in the formulations)
            lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA',
                                'Hbond_D', 'Hbond_A', 'Double_bonds'] 
        else:
            lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA',
                                'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
            
        if size_zeta == True:
            input_param_names = lipid_param_names +formulation_param_names +  ['Size', 'Zeta']
        else:
            input_param_names = lipid_param_names+ formulation_param_names 


        #Get Training Data for cell
        Train_X, Y = extract_training_data(data_file_path, input_param_names, cell, "RLU_", size_zeta, size, PDI)

        #Check/create correct save path
        if os.path.exists(save_path + f'/{cell}') == False:
            os.makedirs(save_path + f'/{cell}', 0o666)
        for param in Train_X.columns:
            f = plt.figure()
            sns.histplot(Train_X.loc[:,param])
            plt.savefig(save_path + f'/{cell}/{param}_Dist.png', 
                        dpi=600, format = 'png', 
                        transparent=True, 
                        bbox_inches='tight')
            plt.close()



if __name__ == "__main__":
    main()