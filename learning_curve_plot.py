from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import KFold
import os
import copy
import time


### Global Variables 

"""**MAIN**"""
def main(c, model_name, trained_model, X, Y, save_path, NUM_ITER = 10, plot_only = False):
    start_time = time.time()
    print('\n###########################\n\n RUNNING LEARNING CURVE')

    
    if plot_only == False:
        ################### Learning Curve TRAINING AND EVALUATION ##############
        #Check/create correct save path
        if os.path.exists(save_path + f'/{c}') == False:
            os.makedirs(save_path + f'/{c}', 0o666)

        #Copy Original Model to not change it when retraining
        model = copy.deepcopy(trained_model)

        num_splits = 5

        #initialize training sizes
        train_size= np.linspace(0.005, 1, 50)*len(X)*(num_splits-1)/num_splits
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
            train_scores_mean.to_csv(f, index = False)

        with open(save_path + f'{model_name}_{c}_val_results.csv', 'w', encoding = 'utf-8-sig') as f:
            validation_scores_mean.to_csv(f, index = False)



    # ############## LOAD DATA AND PLOTTING #####################
    #Initialize Data
    all_data = pd.DataFrame()

    print(f"\n############ Plotting Learning Curve: {model_name}_{c} ###############")

    #Extract Results
    mean_train_scores = pd.read_csv(save_path + f'{model_name}_{c}_training_results.csv' )
    mean_train_scores.rename(columns={mean_train_scores.columns[0]: "Train_size"}, inplace = True)
    mean_train_scores["Score_Type"] = "Train"
    mean_train_scores["Mean_MAE"] = mean_train_scores.iloc[:, range(1, NUM_ITER+1)].mean(axis=1)
    mean_train_scores["sd"] = mean_train_scores.iloc[:, range(1, NUM_ITER+1)].std(axis=1)

    mean_validation_scores = pd.read_csv(save_path + f'{model_name}_{c}_val_results.csv' )
    mean_validation_scores.rename(columns={mean_validation_scores.columns[0]: "Train_size" }, inplace = True)
    mean_validation_scores["Score_Type"] = "Validation"
    mean_validation_scores["Mean_MAE"] = mean_validation_scores.iloc[:, range(1, NUM_ITER+1)].mean(axis=1)
    mean_validation_scores["sd"] = mean_validation_scores.iloc[:, range(1, NUM_ITER+1)].std(axis=1)

    train_valid_scores = pd.concat([mean_validation_scores, mean_train_scores], ignore_index= True)
    train_valid_scores["Cell_Type"] = c
    train_valid_scores["Model_Type"] = model_name
    all_data = pd.concat([all_data, train_valid_scores], ignore_index=True)


    #Convert mean MAE into Percent error
    all_data['Error'] = all_data['Mean_MAE'] * 100  

    #Plot Results
    fig, ax = plt.subplots(figsize = (2.5,1.5))
    sns.set_theme(font='Arial', font_scale= 2)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    

    line = sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == c)],
                            x = "Train_size", y = 'Error', hue = "Score_Type", errorbar = "sd", linewidth = 3, palette=sns.color_palette("Set2"))
    # line = sns.scatterplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == cell_type)],
    #                 x = "Train_size", y = 'Error', hue = "Score_Type", markers = "o")

    line.set(xlim=(0, 900), xticks=np.linspace(0,900,6), ylim=(0, 15), yticks=np.arange(0,20,5))
    line.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
    # add tick marks on x-axis or y-axis
    line.tick_params(bottom=True, left=True)
    # x-axis and y-axis label color
    line.axes.yaxis.label.set_color('black')
    line.axes.xaxis.label.set_color('black')


    line.set_yticklabels(line.get_yticklabels(), fontsize = 12)
    line.set_xticklabels(line.get_xticklabels(), fontsize = 12)
    # plt.tick_params(axis='both', which='major', labelsize=10)

    line.spines['left'].set_color('black')
    line.spines['bottom'].set_color('black')        # x-axis and y-axis spines
    line.spines['right'].set_visible(False)
    line.spines['top'].set_visible(False)

    line.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
    # line.set_title("Learning Curve",weight="bold", fontsize=15)

    plt.xlabel('Training size', fontsize = 12)
    plt.ylabel('Percent Error', fontsize = 12)
    plt.legend(fontsize = 'small', loc='upper right', framealpha = 0)
    plt.savefig(save_path + f'{model_name}_{c}_learning_curve.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')



    print("\n\n--- %s minutes for Learning Curve---" % ((time.time() - start_time)/60))
if __name__ == "__main__":
    main()