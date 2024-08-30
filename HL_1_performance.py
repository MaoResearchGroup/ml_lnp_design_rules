import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, ttest_ind
from sklearn.model_selection import LeaveOneGroupOut
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pickle
import os
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from copy import deepcopy
import time

def eval_HL_1(dist_linkage, X_features, Y, model, N_CV, repeats, method:str = 'MAE'):
    N_features = len(X_features.columns)
    N_feature_tracker = 100
    pvalue_list = []
    MAE_list = [] # empty list to store MAE values
    MAE_std_list = []
    spear_list = []
    spear_std_list = []
    pear_list = []
    pear_std_list = []
    pred_list = []
    exp_list = []

    feature_name_list = [] # empty list to store features names
    feature_number_list = [] # empty list to store number of features
    linkage_distance_list = [] # empty list to store the Ward'slinkage distance

    best_MAE = 1

    #Determine ordered list of features to remove
    ordered_feature_removal_list = ['None']
    for n in range(0, 1000, 1):
        features_to_remove = []
        distance = n/500
        cluster_ids = hierarchy.fcluster(dist_linkage, distance, criterion="distance") 
        cluster_id_to_feature_ids = defaultdict(list) 
        
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)

        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

        linkage_distance_list.append(distance) # append linkage distance to empty list

        retained_features = []  # create empty list to save feature names
            
        for feature in selected_features: # for loop to append the utilized input feature names to the empty list
            retained_features.append(X_features.columns[feature])

        for initial_feature in X_features.columns:
            if initial_feature not in retained_features:
                features_to_remove.append(initial_feature)

        #Add new removed feature to ordered feature list
        for feature in features_to_remove:
            if feature not in ordered_feature_removal_list:
                ordered_feature_removal_list.append(feature)

    #test how the feature removal affects performance
    #if better or equal remove feature and update feature list
    #if worse do not remove feature and continue testing the new feature

    initial_feature_list = X_features.columns.tolist() #Start with all features
    removed_feature_list = []
    #Remove the first feature from the ordered feature removal list from all features
    iteration = 0
    best_MAE = 1
    for feature in ordered_feature_removal_list:
        tested_features = initial_feature_list.copy()

        #Remove features from feature list 
        if feature in initial_feature_list:
            tested_features.remove(feature)

        print(f"#############\nTESTING WHEN {feature} IS REMOVED")
        print("No. FEATURES for TESTING: ", len(tested_features))
        selected_features = [X_features.columns.get_loc(col) for col in tested_features]
        acc_results, spearman_results, pearson_results, predictions, experimental, new_model = evaluate_model(X_features, 
                                                                                                              Y, 
                                                                                                              selected_features, 
                                                                                                              model, 
                                                                                                              N_CV,
                                                                                                              repeats=repeats)
        
        
        print(f'CURRENT BEST AVERAGE MAE IS {np.round(best_MAE, 5)}')
        print(f'TEST AVERAGE MAE IS {np.round(np.mean(acc_results),5)}')
        feature_removal = False
        if method == 't-test':
            #Conduct one-way T test
            if feature == 'None':
                control_values = acc_results

            test_values = acc_results
            t_stat, p_value_two_sided = stats.ttest_ind(control_values, test_values)

        
            # Adjust p-value for one-sided test (if the mean of sample1 is hypothesized to be less than sample2)
            if t_stat < 0:
                p_value = p_value_two_sided / 2
            else:
                p_value = 1 - (p_value_two_sided / 2)
            print(f'TEST p-value IS {np.round(p_value,5)}')
        
            # Criteria is a one-way students' T test, alpha of 0.05
            if p_value > 0.05: #NOT SIGNIFICNATLY DIFFERENT
                feature_removal = True
                control_values = test_values
        elif method == 'MAE':
            feature_removal = round(np.mean(acc_results),5) <= round(best_MAE,5)
            p_value = None
        else:
            raise("INVALID TEST METHOD")
        
        if feature_removal:
            print(f'\n{feature} WAS REMOVED AND BEST RESULTS UPDATED\n') 



            #Record Features used/removed
            feature_number_list.append(len(tested_features)) # append the number of input features to empty list
            feature_name_list.append(tested_features) # append the list of feature names to an empty list of lists
            initial_feature_list = tested_features #update the initial feature list with reduced feature list
            removed_feature_list.append(feature) #Add removed feature to removed list
            
            #Calculate results
            pvalue_list.append(p_value)
            best_MAE = np.mean(acc_results)
            MAE_std = np.std(acc_results)
            spearman = np.mean(spearman_results)
            spear_std = np.std(spearman_results)
            pearson  = np.mean(pearson_results)
            pear_std = np.std(pearson_results)

            MAE_list.append(best_MAE) # append average MAE value to empty list
            MAE_std_list.append(MAE_std)
            spear_list.append(spearman)
            spear_std_list.append(spear_std)
            pear_list.append(pearson)
            pear_std_list.append(pear_std)

            #Append model predictions and hold out set values
            pred_list.append(predictions)
            exp_list.append(experimental)
            

            


            #update best results
            best_model = deepcopy(new_model)
            best_training_data = X_features.iloc[:,selected_features].copy()
        
            best_results = [len(initial_feature_list), initial_feature_list, 
                            removed_feature_list,
                            p_value,
                            best_MAE, MAE_std,
                            spearman, spear_std,
                            pearson, pear_std,
                            predictions, experimental,
                            n/N_features]
        else:
            print(f'\n{feature} WAS NOT REMOVED \n') 
        iteration +=1
    print('###############\nFINAL RETAINED FEATURES', best_results[1])
    print('\n# RETAINED FEATURES', len(best_results[1]))
    print('\nFINAL REMOVED FEATURES', best_results[2])

    # create a list of tuples with results model refinement
    list_of_tuples = list(zip(feature_number_list, feature_name_list, removed_feature_list,
                              pvalue_list, 
                              MAE_list, MAE_std_list,
                              spear_list, spear_std_list,
                              pear_list, pear_std_list,
                              pred_list,
                              exp_list,
                              linkage_distance_list)) 
    

    # create a dataframe with results model refinement
    results_df = pd.DataFrame(list_of_tuples, columns = ['# of Features', 
                                                         'Feature names', 
                                                         'Removed Feature Names',
                                                         'p_value', 
                                                         'MAE', 'MAE_std',
                                                         'Spearman', 'Spearman_std',
                                                         'Pearson', 'Pearson_std',
                                                         'Predictions',
                                                         'Hold_out_set',
                                                         'linkage distance']) 
    
    # create a dataframe with best model information
    best_df = pd.DataFrame([best_results], columns= ['# of Features', 
                                                               'Feature names', 
                                                               'Removed Feature Names', 
                                                               'p_value',
                                                                'MAE', 'MAE_std',
                                                                'Spearman','Spearman_std',
                                                                'Pearson', 'Pearson_std',
                                                                'Predictions',
                                                                'Hold_out_set',
                                                                'linkage distance']) 


    return results_df, best_df, best_model, best_training_data


#Use to retrain current models with different feature data
def evaluate_HL_1_model(X,y, groups, selected_features, model):

    test_model = deepcopy(model)# assign selected model to clf_sel
    selected_X = X.loc[:,selected_features].copy()
    Y = y.copy()

    
    #data collection
    group_order = np.unique(groups)
    group_order = np.concatenate([group_order, np.array(['aggregate'])])

    pred_list = []
    test_list = []
    AE_list = []
    acc_list = []
    acc_std_list = []
    spear_list = []
    pears_list = []


    #Group Splitting byLeaveonegroupout evaluation
    logo = LeaveOneGroupOut()

    for j, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
        #Split X
        X_train = selected_X.iloc[train_index]
        X_test = selected_X.iloc[test_index]

        #Split Y
        y_train = Y.iloc[train_index]
        y_test = Y.iloc[test_index]
                
        test_model.fit(X_train, np.ravel(y_train)) # fit the selected model with the training set
        y_pred = test_model.predict(X_test) # predict test set based on selected input features

        #Calculate Performance Statistics
        y_test = np.ravel(y_test)
        AE_values = np.abs(np.array(y_pred) - np.array(y_test))
        acc = AE_values.mean()
        acc_std = AE_values.std()
        spearman = stats.spearmanr(y_pred, y_test)[0]
        pearson = stats.pearsonr(y_pred, y_test)[0]

        #append to lists
        pred_list.append(y_pred)
        test_list.append(y_test) 
        AE_list.append(AE_values)
        acc_list.append(acc)
        acc_std_list.append(acc_std)
        spear_list.append(spearman)
        pears_list.append(pearson)


    #create aggregate metrics
    agg_pred = flatten(pred_list)
    agg_test = flatten(test_list)

    pred_list.append(agg_pred)
    test_list.append(agg_test)

    AE_values = np.abs(np.array(agg_pred) - np.array(agg_test))
    AE_list.append(AE_values)
    acc_list.append(AE_values.mean())
    acc_std_list.append(AE_values.std())
    spear_list.append(stats.spearmanr(agg_pred, agg_test)[0])
    pears_list.append(stats.pearsonr(agg_pred, agg_test)[0])



    df = pd.DataFrame({
        'pred': pred_list,
        'test': test_list,
        'AE': AE_list,
        'MAE': acc_list,
        'MAE_std': acc_std_list,
        'spearman': spear_list,
        'pearson':pears_list
    }, index=group_order )

    return df

def flatten(list):
    return [x for xs in list for x in xs]

def main(pipeline):

    """
    HL_1_performance script
    
    - Used to evaluate leave-one-lipid-out model performance
    - This provides sample performance metrics of model performance on novel lipid structures

    """
    
    print('\n###########################\n\n RUNNING HL HOLD-ONE-OUT EVALUATION')
    start_time = time.time()

    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    model = deepcopy(pipeline['Model_Selection']['Best_Model']['Model'])
    input_params = pipeline['Data_preprocessing']['Input_Params'].copy()
    X = pipeline['Data_preprocessing']['X'].copy()
    y = pipeline['Data_preprocessing']['y'].copy()

    result_save_path = pipeline['Saving']['Models'] + f'/HL-1/'

    #Check save path
    if os.path.exists(result_save_path) == False:
            os.makedirs(result_save_path, 0o666)

    HL_groups = pipeline['Data_preprocessing']['all_proc_data']['Helper_lipid']


    #Evalute best model using split by Helper lipid groups
    df_results = evaluate_HL_1_model(X,y, HL_groups, input_params, model)

    #Save Results into CSV file
    with open(result_save_path + f'/{model_name}_HL_1_Results.csv', 'w', encoding = 'utf-8-sig', newline='') as f: #Save file to csv
        df_results.to_csv(f, index = True)

    # loop through and save the AE as a single csv
    columns = []
    for index, row in df_results.iterrows():
        columns.append(pd.Series(df_results.at[index, 'AE']))

    HL_1_AE = pd.concat(columns, axis=1)
    HL_1_AE.columns = [df_results.index]

    with open(result_save_path + f'/HL_1_AE.csv', 'w', encoding = 'utf-8-sig', newline='') as f: #Save file to csv
            HL_1_AE.to_csv(f, index = True)

    #Update Pipeline
    pipeline['Model_Selection']['Best_Model']['HL_1'] = df_results

    print("\n\n--- %s minutes for HL_1 ---" % ((time.time() - start_time)/60))  

    return pipeline, df_results

if __name__ == "__main__":
    main()