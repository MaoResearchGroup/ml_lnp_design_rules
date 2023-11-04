from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import KFold
import os
from utilities import extract_training_data
#from ngboost import NGBRegressor

### Global Variables 

"""**MAIN**"""
def main(cell_model_list, data_file_path, save_path, model_folder,input_param_names, size_cutoff, PDI_cutoff, keep_PDI, prefix, NUM_ITER = 10, plot_only = False):

    if plot_only == False:
        ################### Learning Curve TRAINING AND EVALUATION ##############
        for cell_model in cell_model_list:
            c = cell_model[0]
            model_name = cell_model[1]
            #Get Training Data for cell type
            X, Y, _ = extract_training_data(data_file_path, input_param_names, c, size_cutoff, PDI_cutoff, keep_PDI, prefix)
            #Check/create correct save path
            if os.path.exists(save_path + f'/{c}') == False:
                os.makedirs(save_path + f'/{c}', 0o666)

            num_splits = 5

            # #open model
            # with open(model_folder + f"{c}/{model_name}_{c}_Best_Model.pkl", 'rb') as file: # import trained model
            #     model = pickle.load(file)
            #open model
            with open(model_folder + f"{model_name}/{c}/{model_name}_{c}_Trained.pkl", 'rb') as file: # import trained model
                model = pickle.load(file)

            train_size= np.linspace(0.005, 1, 50)*len(X)*(num_splits-1)/num_splits #####
            train_size = np.floor(train_size).astype(int)
            train_scores_mean = pd.DataFrame(index=train_size)
            validation_scores_mean = pd.DataFrame(index=train_size)
            print(f"\n############ Learning Curve: {model_name}_{c}############ ")
            
            #Train model and record performance
            for i in range(NUM_ITER):
                cross_val = KFold(n_splits= num_splits, random_state= i+10, shuffle=True)
                train_sizes, train_scores, validation_scores = learning_curve(estimator = model, X = X, y = np.ravel(Y), cv = cross_val, train_sizes= train_size,
                                                                            scoring = 'neg_mean_absolute_error', shuffle= True, n_jobs= -1)

                train_scores_mean[i] = -train_scores.mean(axis = 1)
                validation_scores_mean[i] = -validation_scores.mean(axis = 1)
                

            #Save Data
            with open(save_path + f'{model_name}_{c}_training_results.csv', 'w', encoding = 'utf-8-sig') as f:
                train_scores_mean.to_csv(f)

            with open(save_path + f'{model_name}_{c}_val_results.csv', 'w', encoding = 'utf-8-sig') as f:
                validation_scores_mean.to_csv(f)



    # ############## LOAD DATA AND PLOTTING #####################
    #Initialize Data
    all_data = pd.DataFrame()
    for cell_model in cell_model_list:
        cell_type = cell_model[0]
        model_name = cell_model[1]

        print(f"\n############ Plotting Learning Curve: {model_name}_{cell_type} ###############")
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


        #Convert mean MAE into Percent error
        all_data['Error'] = all_data['Mean_MAE'] * 100  

        fig, ax = plt.subplots(figsize = (2.5,1))
        sns.set_theme(font='Arial', font_scale= 2)
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12
        line = sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == cell_type)],
                                x = "Train_size", y = 'Error', hue = "Score_Type", errorbar = "sd", linewidth = 3)
        

        line.set(xlim=(0, 900), xticks=np.linspace(0,900,6), ylim=(0, 30), yticks=np.arange(0,35,5))
        line.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
        # add tick marks on x-axis or y-axis
        line.tick_params(bottom=True, left=True)
        # x-axis and y-axis label color
        line.axes.yaxis.label.set_color('black')
        line.axes.xaxis.label.set_color('black')


        line.set_yticklabels(line.get_yticklabels())
        line.set_xticklabels(line.get_xticklabels())
        # plt.tick_params(axis='both', which='major', labelsize=10)

        line.spines['left'].set_color('black')
        line.spines['bottom'].set_color('black')        # x-axis and y-axis spines
        line.spines['right'].set_visible(False)
        line.spines['top'].set_visible(False)

        line.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
        line.set_title("Learning Curve",weight="bold", fontsize=15)

        plt.xlabel('Training size')
        plt.ylabel('Percent Error')
        plt.legend(fontsize = 'small', loc='upper right', framealpha = 0)
        plt.savefig(save_path + f'{model_name}_{cell_type}_learning_curve.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')

if __name__ == "__main__":
    main()