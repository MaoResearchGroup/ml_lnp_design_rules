import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.preprocessing import MinMaxScaler

def init_data(filepath, cell_type_names):
     """ Takes a full file path to a spreadsheet and an array of cell type names. 
     Returns a dataframe with 0s replaced with 1s."""


################ Retreive Data ##############################################
datafile_path = 'Raw_Data/9_Master_Formulas.csv'
model_folder = "Trained_Models/Final_Models/"
plt_save = "Figures/Training_data_histo/"
retrain_model_save = "Trained_Models/Retrained/" 
cell_type_list = ['HEK293', 'HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']
model_list = ['RF', 'MLR', 'lasso', 'PLS', 'kNN', 'LGBM', 'XGB', 'DT']


wt_percent = False
size_zeta = False

if wt_percent == True:
          formulation_param_names = ['wt_Helper', 'wt_Dlin','wt_Chol', 'wt_DMG', 'wt_pDNA']
else:
          formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                        'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio'] 
          
lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA', 'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
#lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP','Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds', 'Helper_MW']
if size_zeta == True:
    input_param_names = lipid_param_names +  formulation_param_names + ['Size', 'Zeta']
else:
    input_param_names = lipid_param_names +  formulation_param_names 
#############################################################################

def main():
    df = pd.read_csv(datafile_path)
    for cell_type in cell_type_list:
        #df["RLU_" + cell_type].replace(0, 1, inplace= True) #Replace 0 transfection with 1
        df["Exp_" + cell_type] = np.exp(df["RLU_" + cell_type])
        #df["Exp_" + cell_type].mask(df["Exp_" + cell_type]<10, inplace=True) #Remove very low/ thus noisy transfecting formulations values
        df["Exp_" + cell_type] = df["Exp_" + cell_type] + 10 #Log(x+c) transform
        df["LN_" + cell_type] = np.log(df["Exp_" + cell_type]) #log10
        # df["Log10_" + cell_type] = np.log10(df["Exp_" + cell_type]) #log10
        # print(df.count())
        print(df)

    for cell_type in cell_type_list:
        #print(df.count("LN_" + cell_type))
        X = df[input_param_names]                         
        Y = df['RLU_' + cell_type].to_numpy()
        scaler = MinMaxScaler().fit(Y.reshape(-1,1))
        temp_Y = scaler.transform(Y.reshape(-1,1))
        Y = pd.DataFrame(temp_Y, columns = ['RLU_' + cell_type])
        F = df['Formula label']


    # for cell_type in cell_type_list:
    #     f = plt.figure(figsize = (10, 10))
    #     sns.histplot(df, x = "LN_"+ cell_type, hue = "Helper_lipid", multiple="stack")
    #     plt.savefig(plt_save + f'{cell_type}_add1.png', bbox_inches = 'tight')
    #     #plt.show()
    


if __name__ == "__main__":
    main()