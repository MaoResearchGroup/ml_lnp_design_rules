# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

def main():

    ################ Retreive Data ##############################################
    result_folder = "Trained_Models/Final_Models/"
    save_folder = "BMES_Abstract" 
    #cell_type = ['HEK293', 'HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']
    cell_type = ['B16']
    #model_list = ['RF', 'MLR', 'lasso', 'PLS', 'kNN', 'LGBM', 'XGB', 'DT']
    model_list = ['RF','LGBM', 'XGB']
    
    ##########  Collect all Results ###############
    all_results = pd.DataFrame(columns = ['Model', 'Cell_Type', 'Valid Score', 'Test Score','Spearmans Rank','Pearsons Correlation','Model Parms', 'Experimental_Transfection','Predicted_Transfection'])

    for model in model_list:
        for cell in cell_type:
            result_file_path = result_folder + f'{model}/{cell}/{model}_{cell}_HP_Tuning_Results.pkl'
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
    

    ######### Hold Out Validation Pred vs Exp. Plots ########
    for model_name in model_list:
        fig = plt.figure(figsize=(5, 5))
        for cell in cell_type:
            predicted = pred_transfection.at[model_name, cell]
            experimental = exp_transfection.at[model_name, cell]

            sns.regplot(x = experimental, y = predicted, color = "k")
            #plt.plot([0, 1], [0, 1], linestyle = 'dotted', color = 'r') #Ideal line
            plt.annotate('Pearsons r = {:.2f}'.format(pearson_results.at[model, cell]), xy=(0.1, 0.9), xycoords='axes fraction', fontsize=14)
            plt.annotate('Spearmans r = {:.2f}'.format(spearman_results.at[model, cell]), xy=(0.1, 0.8), xycoords='axes fraction', fontsize=14)
            plt.ylabel('ML Predicted Normalized RLU', fontsize=12)
            plt.xlabel('Experimental Normalized RLU', fontsize=12)
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.title(cell, fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=10)

            # plt.subplot(2, 3, cell_type.index(cell)+1)
            # plt.scatter(predicted, experimental)
            # plt.title(cell)


            # if (cell_type.index(cell)+1 == 1 or cell_type.index(cell)+1 == 4):
            #     plt.ylabel('Predicted_RLU')
            # if cell_type.index(cell)+1 >= 4:
            #     plt.xlabel('Experimental_RLU')
     
        plt.savefig(save_folder+ f'/{model_name}_predictions.png', bbox_inches = 'tight')


if __name__ == "__main__":
    main()