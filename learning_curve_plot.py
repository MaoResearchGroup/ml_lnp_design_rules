from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import KFold
import os
#from ngboost import NGBRegressor
def extract_training_data(data_path, input_params, cell, prefix, size_cutoff, PDI_cutoff):
    #Load Training data
    df = pd.read_csv(data_path)
    #Remove unnecessary columns
    cell_data = df[['Formula label', 'Helper_lipid'] + input_params + [prefix + cell]]
    cell_data = cell_data.dropna() #Remove any NaN rows

    if "Size" in input_params:
        cell_data = cell_data[cell_data.Size != 0] #Remove any rows where size = 0
        cell_data = cell_data[cell_data.Size <= size_cutoff]
        cell_data = cell_data[cell_data.PDI <= PDI_cutoff] #Remove any rows where PDI > cutoff
    if "Zeta" in  input_params:
        cell_data = cell_data[cell_data.Zeta != 0] #Remove any rows where zeta = 0


    # #Remove PDI column from input features
    # cell_data.drop(columns = 'PDI', inplace = True)

    cell_data.loc[cell_data[prefix + cell] < 3, prefix + cell] = 3 #replace all RLU values below 3 to 3



    print(cell_data)


    print("Input Parameters used:", input_params)
    print("Number of Datapoints used:", len(cell_data.index))

    X = cell_data[input_params]                         
    Y = cell_data[prefix + cell].to_numpy()
    scaler = MinMaxScaler().fit(Y.reshape(-1,1))
    temp_Y = scaler.transform(Y.reshape(-1,1))
    Y = pd.DataFrame(temp_Y, columns = [prefix + cell])
    return X, Y

### Global Variables 
################ Retreive/Store Data ##############################################
RUN_NAME = "Feature_reduction_Size_600_Zeta_PDI_0.45"
save_path = f"Figures/Training_size/{RUN_NAME}/"
model_folder = f"Feature_Reduction/{RUN_NAME}/"
data_file_path = "Raw_Data/10_Master_Formulas.csv"
################ INPUT PARAMETERS ############################################




"""**MAIN**"""
def main():
##################### Run ###############################
    cell_type_list = ['HEK293', 'HepG2', 'N2a', 'B16', 'PC3', 'ARPE19']
    model_list = ['RF', 'XGB', 'LGBM']
    size = True
    zeta = True
    size_cutoff = 600
    PDI_cutoff = 0.45 #Use 1 to include all data
    NUM_ITER = 10
    prefix = "RLU_" #WARNING: HARDCODED

    # cell_type_list = ['B16']
    # model_list = ['LGBM']
    # #Rerun model training based on clusters
    for c in cell_type_list:

        #Features to use
        formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                      'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio']
        
        if c in ['ARPE19','N2a']:
            #Total_Carbon_Tails Removed (It does not change in the formulations)
            lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP',
                                'Hbond_D', 'Hbond_A', 'Double_bonds'] 
        else:
            lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP',
                                'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
        
        input_param_names = lipid_param_names +  formulation_param_names
        
        if size == True:
            input_param_names = input_param_names + ['Size', 'PDI']

        if zeta == True:
            input_param_names = input_param_names + ['Zeta']



        #Get Training Data for cell type
        X, Y = extract_training_data(data_file_path, input_param_names, c, prefix, size_cutoff, PDI_cutoff)

        #Check/create correct save path
        if os.path.exists(save_path + f'/{c}') == False:
            os.makedirs(save_path + f'/{c}', 0o666)


#################### MODEL TRAINING AND EVALUATION ##############
        for model_name in model_list:

            num_splits = 5

            #open model
            with open(model_folder + f"{c}/{model_name}_{c}_Best_Model.pkl", 'rb') as file: # import trained model
                model = pickle.load(file)


            train_size= np.linspace(0.01, 1, 100)*len(X)*(num_splits-1)/num_splits
            train_size = np.floor(train_size).astype(int)
            train_scores_mean = pd.DataFrame(index=train_size)
            validation_scores_mean = pd.DataFrame(index=train_size)
            print(f"\n############ Testing: {model_name}_{c} ###############")
            
            #Train model and record performance
            for i in range(NUM_ITER):
                cross_val = KFold(n_splits= num_splits, random_state= i, shuffle=True)
                train_sizes, train_scores, validation_scores = learning_curve(estimator = model, X = X, y = np.ravel(Y), cv = cross_val, train_sizes= train_size,
                                                                            scoring = 'neg_mean_absolute_error', n_jobs= -1)
                # print('Training scores:\n\n', train_scores)
                # print('\n', '-' * 70) # separator to make the output easy to read
                # print('\nValidation scores:\n\n', validation_scores)

                train_scores_mean[i] = -train_scores.mean(axis = 1)
                validation_scores_mean[i] = -validation_scores.mean(axis = 1)
                # print('Mean training scores\n\n', train_scores_mean)
                # print('\n', '-' * 20) # separator
                # print('\nMean validation scores\n\n',validation_scores_mean)
                

            #Save Data
            with open(save_path + f'{model_name}_{c}_training_results.csv', 'w', encoding = 'utf-8-sig') as f:
                train_scores_mean.to_csv(f)

            with open(save_path + f'{model_name}_{c}_val_results.csv', 'w', encoding = 'utf-8-sig') as f:
                validation_scores_mean.to_csv(f)



    # ############## LOAD DATA AND PLOTTING #####################
    #Initialize Data
    all_data = pd.DataFrame()
    for model_name in model_list:
        for cell_type in cell_type_list:

            print(f"\n############ Plotting: {model_name}_{c} ###############")
            mean_train_scores = pd.read_csv(save_path + f'{model_name}_{cell_type}_training_results.csv' )
            mean_train_scores.rename(columns={mean_train_scores.columns[0]: "Train_size" }, inplace = True)
            mean_train_scores["Score_Type"] = "Train"
            mean_train_scores["Mean_MAE"] = mean_train_scores.iloc[:, range(1, NUM_ITER+1)].mean(axis=1)
            mean_train_scores["sd"] = mean_train_scores.iloc[:, range(1, NUM_ITER+1)].std(axis=1)
        
            mean_validation_scores = pd.read_csv(save_path + f'{model_name}_{cell_type}_val_results.csv' )
            mean_validation_scores.rename(columns={mean_validation_scores.columns[0]: "Train_size" }, inplace = True)
            mean_validation_scores["Score_Type"] = "Validation"
            mean_validation_scores["Mean_MAE"] = mean_validation_scores.iloc[:, range(1, NUM_ITER+1)].mean(axis=1)
            mean_validation_scores["sd"] = mean_validation_scores.iloc[:, range(1, NUM_ITER+1)].std(axis=1)
        
            train_valid_scores = pd.concat([mean_validation_scores, mean_train_scores], ignore_index= True)
            train_valid_scores["Cell_Type"] = cell_type
            train_valid_scores["Model_Type"] = model_name
            all_data = pd.concat([all_data, train_valid_scores], ignore_index=True)

    ###### Plotting
    for model_name in model_list:
        fig, axes = plt.subplots(2,3, sharex = True, sharey = True, figsize = (18, 12))
        plt.ylim(0, 0.25)
        plt.xlim(0, 900)
        fig.suptitle(f'{model_name}_Train_size', fontsize = 18) 
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'HEK293')],
                        x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[0,0]).set(title ='HEK293')
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'B16')],
                        x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[0,1]).set(title ='B16')
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'HepG2')],
                        x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[0,2]).set(title ='HepG2')
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'N2a')],
                        x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[1,0]).set(title ='N2a')        
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'PC3')],
                        x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[1,1]).set(title ='PC3') 
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'ARPE19')],
                        x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[1,2]).set(title ='ARPE19')
        
        plt.savefig(save_path + f'{model_name}_Train_size.png', bbox_inches = 'tight')


if __name__ == "__main__":
    main()