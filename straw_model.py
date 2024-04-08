import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pickle
import os
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, make_scorer
from copy import deepcopy
import time 

def shuffle_column(df, col):
    if col == 'No Shuffle':
        return df.copy()
    
    shuffled_df = df.copy()
    shuffled_df[col] = shuffled_df[col].sample(frac=1).reset_index(drop = True)
    return shuffled_df


def evaluate_model(X,y, model, N_CV = 5):

    #Kfold spltting
    kf = KFold(n_splits=N_CV, random_state= 200, shuffle=True)

    # Lists to store fold results
    test_indices_list = []
    predictions_list = []
    MAE_list = []
    spearman_list = []
    pearson_list = []

    for j, (train_index, test_index) in enumerate(kf.split(X)):
        #Split X
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        #Split Y
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
                
        model.fit(X_train, np.ravel(y_train)) # fit the selected model with the training set
        y_pred = model.predict(X_test) # predict test set based on selected input features

        # append predictions and hold-out set
        predictions_list.append(y_pred)
        test_indices_list.append(y_test.values.flatten()) 
        # Calculate and store the scores
        MAE_list.append(mean_absolute_error(y_test.values.flatten(), y_pred))
        spearman_list.append(spearmanr(y_test.values.flatten(), y_pred)[0])
        pearson_list.append(pearsonr(y_test.values.flatten(), y_pred)[0])
    
    #Calculate overall Performance Statistics

    acc = np.mean(MAE_list)
    acc_sd = np.std(MAE_list)

    spearman = np.mean(spearman_list)
    spearman_sd = np.std(spearman_list)

    pearson = np.mean(pearson_list)
    pearson_sd = np.std(pearson_list)

    return MAE_list, acc, acc_sd, spearman, spearman_sd, pearson, pearson_sd, test_indices_list, predictions_list


def main(pipeline:dict, params_to_test:list):
    
    print('\n###########################\n\n TESTING STRAW MODELS')
    start_time = time.time()

    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    model = deepcopy(pipeline['Model_Selection']['Best_Model']['Model'])
    input_params = pipeline['Data_preprocessing']['Input_Params'].copy()
    X = pipeline['Feature_Reduction']['Refined_X'].copy()
    y = pipeline['Data_preprocessing']['y'].copy()
    N_CV = pipeline['Model_Selection']['N_CV']


    #Check/create correct save path
    RUN_NAME = pipeline['Saving']['RUN_NAME']
    save_path = f'{RUN_NAME}{cell}/Straw_Models/'
    if os.path.exists(save_path) == False:
        os.makedirs(save_path, 0o666)


    #Check whether param is used in the model
    params_used = list(set(params_to_test) & set(X.columns.values))
    test_list = ['No Shuffle'] + params_used
    straw_result_df = pd.DataFrame(index = test_list, columns = ['Shuffled df', 'MAE_list', 'avg_MAE', 'avg_MAE_sd', 'spearman', 'spearman_sd', 'pearson', 'pearson_sd', 'pred', 'test'])
    straw_result_df.index.name = "Feature"

    for param in test_list:
        #shuffle column of interest
        shuffled_X = shuffle_column(X, param)

        #evaluate model with shuffled training data
        MAE_list, acc, acc_sd, spearman, spearman_sd, pearson, pearson_sd, test, pred = evaluate_model(shuffled_X, y, model, N_CV)

        #Results
        print(f"Shuffled {param} ACCURACY:{acc} +- {acc_sd}")
        print(f"Shuffled {param} Spearman:{spearman} +- {spearman_sd}")
        print(f"Shuffled {param} Pearson :{pearson} +- {pearson_sd}")


        #Storing results
        straw_result_df.at[param, 'Shuffled df'] = shuffled_X
        straw_result_df.at[param, 'MAE_list'] = MAE_list
        straw_result_df.at[param, 'avg_MAE'] = acc
        straw_result_df.at[param, 'spearman'] = spearman
        straw_result_df.at[param, 'pearson'] = pearson
        straw_result_df.at[param, 'avg_MAE_sd'] = acc_sd
        straw_result_df.at[param, 'spearman_sd'] = spearman_sd
        straw_result_df.at[param, 'pearson_sd'] = pearson_sd
        straw_result_df.at[param, 'pred'] = pred
        straw_result_df.at[param, 'test'] = test

    #Save results to excel for user
    straw_result_df.to_excel(f'{save_path}Straw_model_results.xlsx')

    #Update Pipeline with results
    outer_key = 'Straw_Model'
    inner_key_list = ['X', 'y', 'Model', 'N_CV', 'Results']
    for new_inner_key in inner_key_list:
        if outer_key in pipeline:
            pipeline[outer_key][new_inner_key] = None
        else:
            pipeline[outer_key] = {new_inner_key: None}

    pipeline['Straw_Model']['X'] = X
    pipeline['Straw_Model']['y'] = y
    pipeline['Straw_Model']['Model'] = model
    pipeline['Straw_Model']['N_CV'] = N_CV
    pipeline['Straw_Model']['Results'] = straw_result_df
    pipeline['STEPS_COMPLETED']['Straw_Model'] = True

    print("\n\n--- %s minutes for feature reduction---" % ((time.time() - start_time)/60))  

    return pipeline

if __name__ == "__main__":
    main()