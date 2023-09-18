import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns
import os


def get_spearman(data, cell1, cell2):
    if cell1 == cell2:
        corr = 1
    else:
        temp_data = data.loc[:,["RLU_"+cell1, "RLU_"+cell2]]
        temp_data.dropna(inplace= True)
        if temp_data.empty:
            corr = "N/A"
        corr = spearmanr(temp_data).correlation
    return corr



################ Global Variables ##############################################
datafile_path = "Raw_Data/10_Master_Formulas.csv"
save_path = "Figures/Features/"
cell_type_list = ["HEK293", "B16", "HepG2", "PC3", 'N2a', 'ARPE19'] #Only cell types with all 1080 datapoints
helper_lipid_list = ["DOPE", "DSPC", "DOTAP", "18PG", "14PA", "DDAB"]
"""**MAIN**"""
def main():
#Extract Training Data
    df = pd.read_csv(datafile_path)
    #print(df)
    training_data = df.loc[:,df.columns.isin("RLU_" + x for x  in cell_type_list)] #Take input parameters and associated values
    training_data.head()
    training_data["Helper_lipid"] = df.loc[:,"Helper_lipid"]
    #Initiate a correlation matrix of zeros
    all_corr = pd.DataFrame(np.zeros((len(cell_type_list),len(cell_type_list))), index = cell_type_list, columns = cell_type_list)
    lipid_corr = pd.DataFrame(np.zeros((len(cell_type_list),len(cell_type_list))), index = cell_type_list, columns = cell_type_list)


    #check save path
    if os.path.exists(save_path) == False:
       os.makedirs(save_path, 0o666)

    #Get correlation between all cell types
    for cell1 in cell_type_list:
        for cell2 in cell_type_list:
            all_corr.loc[cell1, cell2] = get_spearman(training_data, cell1, cell2)

    sns.heatmap(all_corr, vmin=-0.2, vmax=1, annot = True)
    plt.gca()
    plt.savefig(save_path + 'All_Data_Tfxn_Heatmap.png', dpi=600, format = 'png', transparent=True, bbox_inches='tight')
    plt.close()           
            
    #Iterate through subsets based on helper lipid used in formulations
    for lipid in helper_lipid_list:
        lipid_data = training_data.loc[training_data["Helper_lipid"] == lipid]
        for cell1 in cell_type_list:
            for cell2 in cell_type_list:
                lipid_corr.loc[cell1, cell2] = get_spearman(lipid_data, cell1, cell2)
  
            
        sns.heatmap(lipid_corr,vmin=-0.2, vmax=1, annot = True)
        plt.gca()
        plt.title(lipid)
        plt.savefig(save_path + f'{lipid}_Heatmap.png', dpi=600, format = 'png', transparent=True, bbox_inches='tight')
        plt.close()      
        # Save the all correlation data to csv
        with open(save_path + f'{lipid}_tfxn_correlation.csv', 'w', encoding = 'utf-8-sig') as file:
            lipid_corr.to_csv(file)


    # Save the all correlation data to csv
    with open(save_path + 'All_tfxn_correlation.csv', 'w', encoding = 'utf-8-sig') as file:
          all_corr.to_csv(file)


    
    #Get correlation for only helper lipid subsets

 



if __name__ == "__main__":
    main()