import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
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

def feature_correlation(X, cell, save):
    
    fig = plt.figure(figsize=(12, 8))
    corr = spearmanr(X).correlation  # generate a correlation matrix is symmetric
    corr = (corr + corr.T) / 2  # ensure the correlation matrix is symmetric
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr)  # convert the correlation matrix to a distance matrix
    dist_linkage = hierarchy.ward(squareform(distance_matrix))  # generate Ward's linkage values for hierarchical clustering


    # Print dist_linkage without scientific notation
    np.set_printoptions(precision=2, suppress=True)
    #print(dist_linkage)

    #save as csv
    np.savetxt(save + f"{cell}/dist_linkage.csv", dist_linkage, delimiter=",")

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
    # sns.heatmap(round(np.abs(correlations),2), annot=True, 
    #             annot_kws={"size": 7}, vmin=0, vmax=1);


    # plt.figure(figsize=(12,5))
    dissimilarity = 1 - abs(correlations)
    Z = linkage(squareform(dissimilarity), 'complete')

    dendrogram(Z, labels=corr_X.columns, orientation='top', 
            leaf_rotation=90);

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

    plt.savefig(save + f'{cell}/Feature_Correlation_Plot.png', dpi=600, transparent = True, bbox_inches='tight')


    return dist_linkage

def eval_feature_reduction(dist_linkage, X_features, Y, model, N_CV):
    N_features = len(X_features.columns)
    N_feature_tracker = 100
    MAE_list = [] # empty list to store MAE values
    MAE_std_list = [] # empty list to store MAE values
    spear_list = []
    spear_std_list = []
    pear_list = []
    pear_std_list = []

    feature_name_list = [] # empty list to store features names
    feature_number_list = [] # empty list to store number of features
    linkage_distance_list = [] # empty list to store the Ward'slinkage distance

    best_MAE = 1
    previous_MAE = 1

    #Determine ordered list of features to remove
    ordered_feature_removal_list = []
    for n in range(0, 400, 1):
        # select input features to be included in this model iteration based on Ward's linkage of n/10
        features_to_remove = []
        distance = n/200
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

        if len(features_to_remove) <= len(ordered_feature_removal_list):
            continue
        
        #Add new removed feature to ordered feature list
        for feature in features_to_remove:
            if feature not in ordered_feature_removal_list:
                ordered_feature_removal_list.append(feature)

    #test how the feature removal affects performance
    #if better or equal remove feature and update feature list
    #if worse do not remove feature and continue testing the new feature

    initial_feature_list = X_features.columns.tolist() #Start with all features
    removed_feature_list = []


    #Evaluate model with all features to find the best MAE
    acc_results,spearman_results,pearson_results, _ = evaluate_model(X_features, Y, range(len(initial_feature_list)), model, N_CV)
    best_MAE = np.mean(acc_results)

    #append initial data to storage lists
    feature_number_list.append(len(initial_feature_list)) # append the number of input features to empty list
    feature_name_list.append(initial_feature_list) # append the list of feature names to an empty list of lists
    removed_feature_list.append('')

    MAE_list.append(np.mean(acc_results)) # append average MAE value to empty list
    MAE_std_list.append(np.std(acc_results)) # append average MAE value to empty list

    spear_list.append(np.mean(spearman_results)) # append average MAE value to empty list
    spear_std_list.append(np.std(spearman_results)) # append average MAE value to empty list

    pear_list.append(np.mean(pearson_results)) # append average MAE value to empty list
    pear_std_list.append(np.std(pearson_results)) # append average MAE value to empty list
    print(f'\n INITIAL MAE IS {best_MAE}')

    best_results = [len(initial_feature_list), initial_feature_list, 
                            removed_feature_list, 
                            best_MAE, np.std(spearman_results),
                            np.mean(spearman_results), np.std(spearman_results),
                            np.mean(pearson_results), np.std(pearson_results),
                            n/N_features]

    #Remove the first feature from the ordered feature removal list from all features
    iteration = 0
    for feature in ordered_feature_removal_list:
        tested_features = initial_feature_list.copy()

        #Remove features from feature list
        if feature in initial_feature_list:
            tested_features.remove(feature)


        print(f"\nTESTING WHEN {feature} IS REMOVED")
        print("\nNo. FEATURES for TESTING: ", len(tested_features))
        selected_features = [X_features.columns.get_loc(col) for col in tested_features]
        acc_results, spearman_results, pearson_results, new_model = evaluate_model(X_features, Y, selected_features, model, N_CV)

        
        iteration +=1
        # If the performance is equal or worse than previous than update parameters else keep the feature
        if round(np.mean(acc_results),3) <= round(best_MAE,3):
            print(f'\n\n{feature} WAS REMOVED AND BEST RESULTS UPDATED\n\n') 

            feature_number_list.append(len(tested_features)) # append the number of input features to empty list
            feature_name_list.append(tested_features) # append the list of feature names to an empty list of lists
            initial_feature_list = tested_features #update the initial feature list to remove feature
            removed_feature_list.append(feature)

            #update best results
            best_MAE = np.mean(acc_results)
            best_model = deepcopy(new_model)
            best_training_data = X_features.iloc[:,selected_features].copy(deep=True)
            best_model.fit(best_training_data, Y) #Fit best model using all data

            #best_training_data = pd.concat([best_training_data, Y], axis = 1, ignore_index= True)
            best_results = [len(tested_features), tested_features, 
                            removed_feature_list,
                            best_MAE, np.std(spearman_results),
                            np.mean(spearman_results), np.std(spearman_results),
                            np.mean(pearson_results), np.std(pearson_results),
                            n/N_features]
 
            # Only append new reduction data if sucessful
            MAE_list.append(np.mean(acc_results)) # append average MAE value to empty list
            MAE_std_list.append(np.std(acc_results)) # append average MAE value to empty list

            spear_list.append(np.mean(spearman_results)) # append average MAE value to empty list
            spear_std_list.append(np.std(spearman_results)) # append average MAE value to empty list

            pear_list.append(np.mean(pearson_results)) # append average MAE value to empty list
            pear_std_list.append(np.std(pearson_results)) # append average MAE value to empty list

        # print('\n##############################\n\nSTATUS REPORT:') 
        # print('Iteration '+str(iteration)+' of '+str(len(ordered_feature_removal_list))+' completed') 
        # print('No_Tested_Features:', len(tested_features))
        # print('Test_Score: %.3f' % (np.mean(acc_results)))
        # print('Spearman_Score: %.3f' % (np.mean(spearman_results)))
        # print('Pearson_Score: %.3f' % (np.mean(pearson_results)))
        # print('No_Best_features:', best_results[0])      
        # print('Best_Features:', best_results[1])
        # print('Currently removed features', best_results[2])
                
        # print("\n#############################\n ")

    print('\nFINAL RETAINED FEATURES', best_results[1])
    print('\nFINAL REMOVED FEATURES', best_results[2])
    # create a list of tuples with results model refinement
    list_of_tuples = list(zip(feature_number_list, feature_name_list, removed_feature_list, 
                              MAE_list, MAE_std_list, 
                              spear_list, spear_std_list, 
                              pear_list, pear_std_list, 
                              linkage_distance_list)) 
    

    # create a dataframe with results model refinement
    results_df = pd.DataFrame(list_of_tuples, columns = ['# of Features', 'Feature names', 'Removed Feature Names', 
                                                         'MAE', 'MAE_std', 
                                                         'Spearman', 'Spearman_std', 
                                                         'Pearson', 'Pearson_std',
                                                         'linkage distance']) 
    
    # create a dataframe with best model information
    best_df = pd.DataFrame(np.transpose(best_results), index= ['# of Features', 'Feature names', 'Removed Feature Names', 
                                                               'MAE', 'MAE_std', 
                                                         'Spearman', 'Spearman_std', 
                                                         'Pearson', 'Pearson_std',
                                                         'linkage distance']) 


    return results_df, best_df, best_model, best_training_data


#Use to retrain current models with different feature data
def evaluate_model(X,Y, selected_features, model, N_CV):
    acc_list = []
    spearman_list = []
    pearson_list = []

    model # assign selected model to clf_sel

    #Kfold CV
    for i in range(5): #For loop that splits and evaluates the data ten times
        cv_outer = KFold(n_splits=N_CV, random_state= i+100, shuffle=True)
        for j, (train_index, test_index) in enumerate(cv_outer.split(X)):
            #Split X
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            #Split Y
            y_train = Y.iloc[train_index]
            y_test = Y.iloc[test_index]

                        
            X_train_sel = X_train.iloc[:,selected_features] # select input features from training dataset based on Ward's Linkage value
            X_test_sel = X_test.iloc[:,selected_features] # select input features from test dataset based on Ward's Linkage value
                    
            model.fit(X_train_sel, np.ravel(y_train)) # fit the selected model with the training set
            y_pred = model.predict(X_test_sel) # predict test set based on selected input features

            # Get Model Statistics
            acc = round(mean_absolute_error(y_pred, y_test), 3)
            spearman = stats.spearmanr(y_pred, y_test)[0]
            pearson = stats.pearsonr(y_pred, y_test)[0]

            # Append model performance results to associated list
            acc_list.append(acc)
            spearman_list.append(spearman) 
            pearson_list.append(pearson) 

    return acc_list, spearman_list, pearson_list, model


def main(pipeline):
    
    print('\n###########################\n\n RUNNING FEATURE REDUCTION')
    start_time = time.time()

    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    model = deepcopy(pipeline['Model_Selection']['Best_Model']['Model'])
    input_params = pipeline['Data_preprocessing']['Input_Params'].copy()
    X = pipeline['Data_preprocessing']['X'].copy()
    y = pipeline['Data_preprocessing']['y'].copy()
    refined_model_save_path =pipeline['Saving']['Refined_Models']
    N_CV = pipeline['Model_Selection']['N_CV']

    #Check/create correct save path
    if os.path.exists(refined_model_save_path + f'/{cell}') == False:
        os.makedirs(refined_model_save_path + f'/{cell}', 0o666)


    #Calculate and Plot Feature Correlation
    dist_link = feature_correlation(X, cell, refined_model_save_path)
    
    #Test and save models using new Feature clusters 
    results, best_results, best_model, best_data = eval_feature_reduction(dist_link, X, y, model, N_CV)

    #Save Results into CSV file
    with open(refined_model_save_path + f'/{model_name}_Feature_Red_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
        results.to_csv(f, index = False)

    #Save Best Results into CSV file
    with open(refined_model_save_path + f'/{model_name}_Best_Model_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
        best_results.to_csv(f, index = False)

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

    reduced_features = best_results.loc['Feature names', 0]
    removed_features = best_results.loc['Removed Feature Names', 0]


    #Update Pipeline
    pipeline['Feature_Reduction']['Refined_Params'] = reduced_features
    pipeline['Feature_Reduction']['Removed_Params'] = removed_features
    pipeline['Feature_Reduction']['Refined_X'] = best_data
    pipeline['Feature_Reduction']['Refined_Model'] = best_model
    pipeline['Feature_Reduction']['Final_Results'] = best_results
    pipeline['Feature_Reduction']['Reduction_Results'] = results

    print("\n\n--- %s minutes for feature reduction---" % ((time.time() - start_time)/60))  

    return pipeline, reduced_features, removed_features, best_results, best_model, best_data

if __name__ == "__main__":
    main()