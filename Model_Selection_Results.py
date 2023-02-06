# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def main():

    ################ Retreive Data ##############################################
    result_folder = "Trained_Models/230204_Models/" 
    cell_type = ['HEK293', 'HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']
    model_list = ['RF', 'MLR', 'lasso', 'PLS', 'kNN', 'LGBM', 'XGB', 'DT']
    

    all_results = pd.DataFrame(columns = ['Model', 'Cell_Type', 'Valid Score', 'Test Score','Spearmans Rank','Pearsons Correlation','Model Parms'])

    for model in model_list:
        for cell in cell_type:
            result_file_path = result_folder + f'{model}/{cell}/{model}_HP_Tuning_Results.pkl'
            with open(result_file_path, 'rb') as file:
                results = pickle.load(file)
                results.drop(columns = ['Iter','Formulation_Index', 'Experimental_Transfection','Predicted_Transfection'], inplace = True)
                results = results.iloc[[0]] #keep only Best model, return dataframe type
                results.insert(0, 'Model', model) #Add model
                results.insert(1, 'Cell_Type', cell) #Add cell type
                all_results = pd.concat([results, all_results.loc[:]], ignore_index = True).reset_index(drop = True)

    #Save results
    with open(result_folder + "Model_Selection_Results.csv", 'w', encoding = 'utf-8-sig') as f:
        all_results.to_csv(f)
    print('Saved Results')
        
    ########## Tabulate Results ##################
    MAE_results = pd.DataFrame(index = model_list, columns = cell_type)
    spearman_results = pd.DataFrame(index = model_list, columns = cell_type)
    pearson_results = pd.DataFrame(index = model_list, columns = cell_type)
    for model in model_list:
        for cell in cell_type:
            m1 = all_results["Model"] == model
            m2 = all_results["Cell_Type"] == cell
            MAE_results.at[model, cell] = all_results[m1&m2]['Test Score'].values[0]
            spearman_results.at[model, cell] = all_results[m1&m2]['Spearmans Rank'].values[0][0]
            pearson_results.at[model, cell] = all_results[m1&m2]['Pearsons Correlation'].values[0][0]

    with open(result_folder + "Model_Selection_MAE.csv", 'w', encoding = 'utf-8-sig') as f:
        MAE_results.to_csv(f)
    with open(result_folder + "Model_Selection_spearman.csv", 'w', encoding = 'utf-8-sig') as f:
        spearman_results.to_csv(f)
    with open(result_folder + "Model_Selection_pearson.csv", 'w', encoding = 'utf-8-sig') as f:
        pearson_results.to_csv(f)   




if __name__ == "__main__":
    main()