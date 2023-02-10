# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def main():

    ################ Retreive Data ##############################################
    result_folder = "Trained_Models/Final_Models/" 
    cell_type = ['HEK293', 'HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']
    model_list = ['RF', 'MLR', 'lasso', 'PLS', 'kNN', 'LGBM', 'XGB', 'DT']
    

    all_results = pd.DataFrame(columns = ['Model', 'Cell_Type', 'Valid Score', 'Test Score','Spearmans Rank','Pearsons Correlation','Model Parms', 'Experimental_Transfection','Predicted_Transfection'])

    for model in model_list:
        for cell in cell_type:
            result_file_path = result_folder + f'{model}/{cell}/{model}_HP_Tuning_Results.pkl'
            with open(result_file_path, 'rb') as file:
                results = pickle.load(file)
                results.drop(columns = ['Iter','Formulation_Index'], inplace = True)
                results = results.iloc[[0]] #keep only Best model, return dataframe type
                results.insert(0, 'Model', model) #Add model
                results.insert(1, 'Cell_Type', cell) #Add cell type
                all_results = pd.concat([results, all_results.loc[:]], ignore_index = True).reset_index(drop = True)

    #Save results
    with open(result_folder + "Model_Selection_Results.csv", 'w', encoding = 'utf-8-sig') as f:
        all_results.to_csv(f)
    print('Saved Results')
        
    ########## Extract Results ##################
    MAE_results = pd.DataFrame(index = model_list, columns = cell_type)
    spearman_results = pd.DataFrame(index = model_list, columns = cell_type)
    pearson_results = pd.DataFrame(index = model_list, columns = cell_type)
    pred_transfection = pd.DataFrame(index = model_list, columns = cell_type)
    exp_transfection = pd.DataFrame(index = model_list, columns = cell_type)
    for model in model_list:
        for cell in cell_type:
            m1 = all_results["Model"] == model
            m2 = all_results["Cell_Type"] == cell
            MAE_results.at[model, cell] = all_results[m1&m2]['Test Score'].values[0]
            spearman_results.at[model, cell] = all_results[m1&m2]['Spearmans Rank'].values[0][0]
            pearson_results.at[model, cell] = all_results[m1&m2]['Pearsons Correlation'].values[0][0]
            pred_transfection.at[model, cell] = all_results[m1&m2]['Predicted_Transfection'].values[0]
            exp_transfection.at[model, cell] = all_results[m1&m2]['Experimental_Transfection'].values[0].transpose()[0] #Format as list
    
    ########## Tabulate Results ##################
    with open(result_folder + "Model_Selection_MAE.csv", 'w', encoding = 'utf-8-sig') as f:
        MAE_results.to_csv(f)
    with open(result_folder + "Model_Selection_spearman.csv", 'w', encoding = 'utf-8-sig') as f:
        spearman_results.to_csv(f)
    with open(result_folder + "Model_Selection_pearson.csv", 'w', encoding = 'utf-8-sig') as f:
        pearson_results.to_csv(f)   
    

    print(pred_transfection)

    ######### Hold Out Validation Pred vs Exp. Plots ########
    for model_name in model_list:
        fig = plt.figure(figsize=(15, 10))
        for cell in cell_type:
            predicted = pred_transfection.at[model_name, cell]
            experimental = exp_transfection.at[model_name, cell]
            plt.subplot(2, 3, cell_type.index(cell)+1)
            plt.scatter(predicted, experimental)
            plt.title(cell)
            if (cell_type.index(cell)+1 == 1 or cell_type.index(cell)+1 == 4):
                plt.ylabel('Predicted_RLU')
            if cell_type.index(cell)+1 >= 4:
                plt.xlabel('Experimental_RLU')
     
        plt.savefig(result_folder+ f'/{model_name}_predictions.png', bbox_inches = 'tight')


if __name__ == "__main__":
    main()