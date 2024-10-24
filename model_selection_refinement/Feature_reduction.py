import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, ttest_ind
from sklearn.model_selection import train_test_split, KFold
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


"""
Feature_reduction

- Test model performance after the removal of features, if removal of feature does not negatively impact performance, then it is removed from the training data
- Outputs models trained on a refined dataset
- Results save within the run dictionary and run directory

"""
def feature_correlation(X, cell, save):
    
    fig = plt.figure(figsize=(12, 8))
    corr = spearmanr(X).correlation  # generate a correlation matrix is symmetric
    corr = (corr + corr.T) / 2  # ensure the correlation matrix is symmetric
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr)  # convert the correlation matrix to a distance matrix
    dist_linkage = hierarchy.ward(squareform(distance_matrix))  # generate Ward's linkage values for hierarchical clustering


    # Print dist_linkage without scientific notation
    np.set_printoptions(precision=2, suppress=True)

    #save as csv
    np.savetxt(save + f"dist_linkage.csv", dist_linkage, delimiter=",")

    dendro = hierarchy.dendrogram(dist_linkage, labels=X.columns.tolist(), leaf_rotation=90)

    dendro_idx = np.arange(0, len(dendro["ivl"]))

    #Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.set_title("Hierarchical Clustering (Ward-linkage)", fontsize=14, color="black", weight="bold")
    ax1.set_xlabel('FEATURE NAMES', fontsize=14, color="black")
    ax1.set_ylabel('HEIGHT', fontsize=14, color="black")
    ax1.tick_params(axis='y', which='both', labelsize=12)
    ax1.tick_params(axis='x', which='both', labelsize=12)


    im = ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]], 
                    alpha = 1.0)
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    ax2.figure.colorbar(im, format='% .2f')
    ax2.tick_params(axis='y', which='both', labelsize=12)
    ax2.tick_params(axis='x', which='both', labelsize=12)
    ax2.set_title("Spearman's Rank Correlation", fontsize=14, color="black", weight="bold")
    fig.tight_layout()

    corr_X = X
    correlations = corr_X.corr()

    dissimilarity = 1 - abs(correlations)
    Z = linkage(squareform(dissimilarity), 'complete')

    dendrogram(Z, labels=corr_X.columns, orientation='top', 
            leaf_rotation=90)

    # Clusterize the data
    threshold = 0.8
    labels = fcluster(Z, threshold, criterion='distance')

    # Show the cluster
    labels
    # Keep the indices to sort labels
    labels_order = np.argsort(labels)

    # Build a new dataframe with the sorted columns
    for idx, i in enumerate(corr_X.columns[labels_order]):
        if idx == 0:
            clustered = pd.DataFrame(corr_X[i])
        else:
            df_to_append = pd.DataFrame(corr_X[i])
            clustered = pd.concat([clustered, df_to_append], axis=1)

    # plt.figure(figsize=(10,10))
    correlations = clustered.corr()


    my_list = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 
            'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 
            'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 
            'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 
            'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 
            'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 
            'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 
            'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 
            'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 
            'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 
            'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 
            'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 
            'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r']

    # Colors - cmap="mako", cmap="viridis", cmap="Blues", cmap='RdBu', rocket, flare, "seagreen", Reds, Magma
    #fig, ax = plt.subplots()
    kws = dict(cbar_kws=dict(ticks=[0, 0.50, 1], orientation='vertical'), figsize=(12, 12))
    g = sns.clustermap(round(np.abs(correlations),2), method="complete", cmap= "flare", annot=True, 
                annot_kws={"size": 12}, vmin=0, vmax=1, cbar_pos = None, **kws)


    plt.tick_params(axis='y', which='both', labelsize=20)
    plt.tick_params(axis='x', which='both', labelsize=20)

    plt.tight_layout()

    plt.savefig(save + f'Feature_Correlation_Plot.png', dpi=600, transparent = True, bbox_inches='tight')


    return dist_linkage

def eval_feature_reduction(dist_linkage, X_features, Y, model, N_CV, repeats, method:str = 'MAE'):
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
def evaluate_model(X,y, selected_features, model, N_CV = 5, repeats = 5):
    repeat_acc_list  = []
    repeat_spear_list= []
    repeat_pears_list= []
    repeat_pred_list = []
    repeat_test_list = []

    test_model = deepcopy(model)# assign selected model to clf_sel
    selected_X = X.iloc[:,selected_features].copy()
    Y = y.copy()
    #Kfold spltting
    for i in range(repeats):
        pred_list = []
        test_list = []
        Kfold = KFold(n_splits=N_CV, random_state= i + 40, shuffle=True)
        for j, (train_index, test_index) in enumerate(Kfold.split(X)):
            #Split X
            X_train = selected_X.iloc[train_index]
            X_test = selected_X.iloc[test_index]

            #Split Y
            y_train = Y.iloc[train_index]
            y_test = Y.iloc[test_index]
                    
            test_model.fit(X_train, np.ravel(y_train)) # fit the selected model with the training set
            y_pred = test_model.predict(X_test) # predict test set based on selected input features

            # append predictions and hold-out set
            pred_list.append(y_pred)
            test_list.append(y_test.values.flatten()) 

        pred = np.concatenate(pred_list)
        test = np.concatenate(test_list)

        #Calculate Performance Statistics
        acc = mean_absolute_error(pred, test)
        spearman = stats.spearmanr(pred, test)[0]
        pearson = stats.pearsonr(pred, test)[0]

        #append to lists
        repeat_acc_list.append(acc)
        repeat_spear_list.append(spearman)
        repeat_pears_list.append(pearson)
        repeat_pred_list.append(pred)
        repeat_test_list.append(test)

    #Refit new model on all selected training data
    new_model = test_model.fit(selected_X, np.ravel(Y))

    return repeat_acc_list, repeat_spear_list, repeat_pears_list, repeat_pred_list, repeat_test_list,  new_model


def main(pipeline, repeats = 5):
    
    print('\n###########################\n\n RUNNING FEATURE REDUCTION')
    start_time = time.time()

    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    model = deepcopy(pipeline['Model_Selection']['Best_Model']['Model'])
    X = pipeline['Data_preprocessing']['X'].copy()
    y = pipeline['Data_preprocessing']['y'].copy()
    refined_model_save_path =pipeline['Saving']['Refined_Models']
    N_CV = pipeline['Model_Selection']['N_CV']

    #Check/create correct save path
    if os.path.exists(refined_model_save_path) == False:
        os.makedirs(refined_model_save_path, 0o666)

    #Calculate and Plot Feature Correlation
    dist_link = feature_correlation(X, cell, refined_model_save_path)
    
    #Test and save models using new Feature clusters 
    results, best_results, best_model, best_data = eval_feature_reduction(dist_link, X, y, model, N_CV, repeats)

    #Save Results into CSV file
    with open(refined_model_save_path + f'/{model_name}_Feature_Red_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
        results.to_csv(f, index = False)

    #Save Best Results into CSV file
    with open(refined_model_save_path + f'/{model_name}_Best_Model_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
        best_results.to_csv(f, index=False)

    #Save Best Results into pkl file
    with open(refined_model_save_path + f'/{model_name}_Best_Model_Results.pkl', 'wb') as file:
        pickle.dump(best_results, file)

    #Save Best training data into CSV file
    with open(refined_model_save_path + f'/{model_name}_Best_Training_Data.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
        best_data.to_csv(f, index = False)       

    #Save Best training data into pkl file
    with open(refined_model_save_path + f'/{model_name}_Best_Training_Data.pkl', 'wb') as file:
        pickle.dump(best_data, file)

    #Save Y into CSV file
    with open(refined_model_save_path + f'/{model_name}_output.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
        y.to_csv(f, index = False)       

    #Save Best training data into pkl file
    with open(refined_model_save_path + f'/{model_name}_output.pkl', 'wb') as file:
        pickle.dump(y, file)

    # Save the Model to pickle file
    with open(refined_model_save_path + f'/{model_name}_Best_Model.pkl', 'wb') as file: 
        pickle.dump(best_model, file)

    reduced_features = best_results.at[0,'Feature names']
    removed_features = best_results.at[0,'Removed Feature Names']


    #Update Pipeline
    pipeline['Feature_Reduction']['N_CV'] = N_CV
    pipeline['Feature_Reduction']['Repeats'] = repeats
    pipeline['Feature_Reduction']['Refined_Params'] = reduced_features
    pipeline['Feature_Reduction']['Removed_Params'] = removed_features
    pipeline['Feature_Reduction']['Refined_X'] = best_data
    pipeline['Feature_Reduction']['Refined_Model'] = best_model
    pipeline['Feature_Reduction']['Final_Results'] = best_results
    pipeline['Feature_Reduction']['Reduction_Results'] = results
    pipeline['STEPS_COMPLETED']['Feature_Reduction'] = True

    print("\n\n--- %s minutes for feature reduction---" % ((time.time() - start_time)/60))  

    return pipeline, reduced_features, removed_features, best_results, best_model, best_data

if __name__ == "__main__":
    main()