import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
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
    sns.heatmap(round(np.abs(correlations),2), annot=True, 
                annot_kws={"size": 7}, vmin=0, vmax=1);


    plt.figure(figsize=(12,5))
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

    plt.figure(figsize=(10,10))
    correlations = clustered.corr()
    plot = sns.heatmap(round(correlations,2), cmap='mako', annot=True, 
                annot_kws={"size": 7}, vmin=-1, vmax=1);



    plt.figure(figsize=(15,10))

    for idx, t in enumerate(np.arange(0.2,1.1,0.1)):
        
        # Subplot idx + 1
        plt.subplot(3, 3, idx+1)
        
        # Calculate the cluster
        labels = fcluster(Z, t, criterion='distance')

        # Keep the indices to sort labels
        labels_order = np.argsort(labels)

        # Build a new dataframe with the sorted columns
        for idx, i in enumerate(corr_X.columns[labels_order]):
            if idx == 0:
                clustered = pd.DataFrame(corr_X[i])
            else:
                df_to_append = pd.DataFrame(corr_X[i])
                clustered = pd.concat([clustered, df_to_append], axis=1)
                
        # Plot the correlation heatmap
        correlations = clustered.corr()
        sns.heatmap(round(correlations,2), cmap='RdBu', vmin=-1, vmax=1, 
                    xticklabels=False, yticklabels=False)
        plt.title("Threshold = {}".format(round(t,2)))


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
    for color in my_list:
        kws = dict(cbar_kws=dict(ticks=[0, 0.25, 0.50, 0.75, 1], orientation='horizontal'), figsize=(6, 6))

    g = sns.clustermap(round(np.abs(correlations),2), method="complete", row_cluster=False, cmap=my_list[14], annot=True, 
                annot_kws={"size": 10}, vmin=0, vmax=1, figsize=(10,10));

    x0, _y0, _w, _h = g.cbar_pos

    g.ax_cbar.set_position([x0, 1.0, g.ax_row_dendrogram.get_position().width, 0.15])
    g.ax_cbar.set_title("Spearman's Rank Correlation", fontsize = 12)
    g.ax_cbar.tick_params(axis='x', length=10)
    for spine in g.ax_cbar.spines:
        g.ax_cbar.spines[spine].set_color('crimson')
        g.ax_cbar.spines[spine].set_linewidth(3)


    plt.tick_params(axis='y', which='both', labelsize=15)
    plt.tick_params(axis='x', which='both', labelsize=15)

    plt.tight_layout()
    # #Save Figure
    plt.savefig(save + f'{cell}/Feature_Correlation_Plot.png', dpi=600, transparent = True, bbox_inches='tight')
    # #plt.show()


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



    for n in range(0, 200, 1):
        # select input features to be included in this model iteration based on Ward's linkage of n/10
        distance = n/100
        cluster_ids = hierarchy.fcluster(dist_linkage, distance, criterion="distance") 
        cluster_id_to_feature_ids = defaultdict(list) 
        
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
                
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        linkage_distance_list.append(distance) # append linkage distance to empty list

        tested_features = []  # create empty list to save feature names
            
        for feature in selected_features: # for loop to append the utilized input feature names to the empty list
            tested_features.append(X_features.columns[feature])
        
        #If the number of selected features is not less than previous, than skip iteration
        if len(tested_features) >= N_feature_tracker:
            continue

        feature_number_list.append(len(tested_features)) # append the number of input features to empty list
        feature_name_list.append(tested_features) # append the list of feature names to an empty list of lists

        print("\nSELECTED FEATURES: ", tested_features)
        acc_results, spearman_results, pearson_results, new_model = evaluate_model(X_features, Y, selected_features, model, N_CV)
        N_feature_tracker = len(tested_features)
        # find best model
        if round(np.mean(acc_results),3) <= round(best_MAE,3):
            print('\n BEST RESULTS UPDATED\n') 
            best_MAE = np.mean(acc_results)
            best_model = deepcopy(new_model)
            best_training_data = X_features.iloc[:,selected_features].copy(deep=True)
            best_model.fit(best_training_data, Y) #Fit best model using all data

            #best_training_data = pd.concat([best_training_data, Y], axis = 1, ignore_index= True)
            best_results = [len(tested_features), tested_features, 
                            best_MAE, np.std(spearman_results),
                            np.mean(spearman_results), np.std(spearman_results),
                            np.mean(pearson_results), np.std(pearson_results),
                            n/N_features]
 
        MAE_list.append(np.mean(acc_results)) # append average MAE value to empty list
        MAE_std_list.append(np.std(acc_results)) # append average MAE value to empty list

        spear_list.append(np.mean(spearman_results)) # append average MAE value to empty list
        spear_std_list.append(np.std(spearman_results)) # append average MAE value to empty list

        pear_list.append(np.mean(pearson_results)) # append average MAE value to empty list
        pear_std_list.append(np.std(pearson_results)) # append average MAE value to empty list

                    
        print('\n################################################################\n\nSTATUS REPORT:') 
        print('Iteration '+str(n+1)+' of '+str(100)+' completed') 
        print('No_Tested_Features:', len(tested_features))
        print('Ward Linkage Distance:', distance)
        print('Test_Score: %.3f' % (np.mean(acc_results)))
        print('Spearman_Score: %.3f' % (np.mean(spearman_results)))
        print('Pearson_Score: %.3f' % (np.mean(pearson_results)))
        print('No_Best_features:', best_results[0])      
        print('Best_Features:', best_results[1])
                
        print("\n################################################################\n ")
    
    # create a list of tuples with results model refinement
    list_of_tuples = list(zip(feature_number_list, feature_name_list, MAE_list, MAE_std_list, 
                              spear_list, spear_std_list, 
                              pear_list, pear_std_list, 
                              linkage_distance_list)) 
    

    # create a dataframe with results model refinement
    results_df = pd.DataFrame(list_of_tuples, columns = ['# of Features', 'Feature names', 'MAE', 'MAE_std', 
                                                         'Spearman', 'Spearman_std', 
                                                         'Pearson', 'Pearson_std',
                                                         'linkage distance']) 
    
    # create a dataframe with best model information
    best_df = pd.DataFrame(np.transpose(best_results), index= ['# of Features', 'Feature names', 'MAE', 'MAE_std', 
                                                         'Spearman', 'Spearman_std', 
                                                         'Pearson', 'Pearson_std',
                                                         'linkage distance']) 
    return results_df, best_df, best_model, best_training_data


#Use to retrain current models with different feature data
def evaluate_model(X,Y, selected_features, model, N_CV):
    acc_list = []
    spearman_list = []
    pearson_list = []

    #Kfold CV
    for i in range(10): #For loop that splits and evaluates the data ten times

        #print(f"\n LOOP: {i+1}/10")

        cv_outer = KFold(n_splits=N_CV, random_state= i+1, shuffle=True)
        for j, (train_index, test_index) in enumerate(cv_outer.split(X)):
            #Split X
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            #Split Y
            y_train = Y.iloc[train_index]
            y_test = Y.iloc[test_index]

                        
            X_train_sel = X_train.iloc[:,selected_features] # select input features from training dataset based on Ward's Linkage value
            X_test_sel = X_test.iloc[:,selected_features] # select input features from test dataset based on Ward's Linkage value
                    

            clf_sel = model # assign selected model to clf_sel
            clf_sel.fit(X_train_sel, np.ravel(y_train)) # fit the selected model with the training set
            y_pred = clf_sel.predict(X_test_sel) # predict test set based on selected input features

            # Get Model Statistics
            acc = round(mean_absolute_error(y_pred, y_test), 3)
            spearman = stats.spearmanr(y_pred, y_test)[0]
            pearson = stats.pearsonr(y_pred, y_test)[0]

            # Append model performance results to associated list
            acc_list.append(acc)
            spearman_list.append(spearman) 
            pearson_list.append(pearson) 

    return acc_list, spearman_list, pearson_list, clf_sel

#define a function called plot_feature_reduction 
def plot_feature_reduction(stats_df, cell_type, model_name, save):

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor='white')
    ax2 = ax1.twinx()

    # Plot the points with error bars for Average MAE
    
    ax1.errorbar(stats_df['# of Features'], stats_df['MAE'], yerr=stats_df['MAE_std'], fmt='o',markersize = 10, color='black',
                ecolor='darkgray', elinewidth=3, capsize=4, capthick=3, label='Average MAE')

    # Draw a line connecting the points for Average MAE
    ax1.plot(stats_df['# of Features'], stats_df['MAE'], color='blue', linewidth=6)

    # Plot error bars for Spearman correlation coefficient
    ax2.errorbar(stats_df['# of Features'], stats_df['Spearman'], yerr=stats_df['Spearman_std'], fmt='v', markersize = 10, color='black',
                ecolor='darkgray', elinewidth=3, capsize=5, capthick=3, label='Average Spearman')

    # Draw a line connecting the points for Spearman correlation coefficient
    ax2.plot(stats_df['# of Features'], stats_df['Spearman'], color='red', linewidth=6)

    # Plot error bars for Pearson correlation coefficient
    ax2.errorbar(stats_df['# of Features'], stats_df['Pearson'], yerr=stats_df['Pearson_std'], fmt='^', markersize = 10, color='black',
                ecolor='darkgray', elinewidth=3, capsize=5, capthick=3, label='Average Pearson')

    # Draw a line connecting the points for Pearson correlation coefficient
    ax2.plot(stats_df['# of Features'], stats_df['Pearson'], color='green', linewidth=6)


    # Set labels for the x-axis and y-axes
    ax1.set_xlabel('Number of Features', fontsize = 20)
    ax1.set_ylabel('Average MAE', fontsize = 20)
    ax2.set_ylabel('Correlation Coefficients', fontsize = 20)
    ax1.set_title("Feature Reduction",weight="bold", fontsize=30)
    # Reverse the x-axis
    ax1.invert_xaxis()

    # Adjust font size and style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15

    # Set integer labels on the x-axis
    ax1.set_xticks(np.arange(int(stats_df['# of Features'].min()), int(stats_df['# of Features'].max()) + 1, 2))
    ax1.set_yticks(np.linspace(0.05, 0.12, 8))
    ax1.tick_params(axis = 'y', labelsize=15)
    ax1.tick_params(axis = 'x', labelsize=15)
    # Combine the legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc='upper left')

    # Update the legend titles
    ax1.legend(lines, ['Average MAE', 'Average Spearman', 'Average Pearson'], loc='center left')

    # Save the plot as a high-resolution image (e.g., PNG or PDF)
    plt.savefig(save + f'{cell_type}/{cell_type}_{model_name}_Feature_Reduction_Plot.svg', dpi=600, transparent = True, bbox_inches='tight')

    plt.close()


def main(cell_names, model_list, model_save_path, refined_model_save_path, input_param_names, prefix, N_CV = 5 ):

    #Remove PDI from input parameters as it was used as a preprocessing variable.
    input_params = input_param_names.copy()
    
    while "PDI" in input_params:
        input_params.remove("PDI")
    for cell in cell_names:

        #Total carbon tails does not change for any datapoints
        if cell in ['ARPE19','N2a']:
            while "Total_Carbon_Tails" in input_params:
                input_params.remove("Total_Carbon_Tails")


        for model_name in model_list:
            #Get Training Data for cell
            with open(model_save_path + f"{model_name}/{cell}/{cell}_Training_Data.pkl", 'rb') as file: # import trained model
                training_data = pickle.load(file)
            Train_X = training_data[input_params]
            Y = training_data[prefix + cell].to_numpy()
            scaler = MinMaxScaler().fit(Y.reshape(-1,1))
            temp_Y = scaler.transform(Y.reshape(-1,1))
            Y = pd.DataFrame(temp_Y, columns = [prefix + cell])

            #Check/create correct save path
            if os.path.exists(refined_model_save_path + f'/{cell}') == False:
                os.makedirs(refined_model_save_path + f'/{cell}', 0o666)



            #Calculate and Plot Feature Correlation
            dist_link = feature_correlation(Train_X, cell, refined_model_save_path)

            #Open correct model
            model_path = model_save_path + f'{model_name}/{cell}/{model_name}_{cell}_Trained.pkl'
            with open(model_path, 'rb') as file: # import trained model
                trained_model = pickle.load(file)
            
            #Test and save models using new Feature clusters 
            results, best_results, best_model, best_data = eval_feature_reduction(dist_link, Train_X, Y, trained_model, N_CV)

            #Plot feature reduction results
            plot_feature_reduction(results, cell, model_name, refined_model_save_path)

            #Save Results into CSV file
            with open(refined_model_save_path + f'/{cell}/{model_name}_{cell}_Feature_Red_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
                results.to_csv(f)

            #Save Best Results into CSV file
            with open(refined_model_save_path + f'/{cell}/{model_name}_{cell}_Best_Model_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
                best_results.to_csv(f)

            #Save Best Results into pkl file
            with open(refined_model_save_path + f'/{cell}/{model_name}_{cell}_Best_Model_Results.pkl', 'wb') as file:
                pickle.dump(best_results, file)

            #Save Best training data into CSV file
            with open(refined_model_save_path + f'/{cell}/{model_name}_{cell}_Best_Training_Data.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
                best_data.to_csv(f)       

            #Save Best training data into pkl file
            with open(refined_model_save_path + f'/{cell}/{model_name}_{cell}_Best_Training_Data.pkl', 'wb') as file:
                pickle.dump(best_data, file)

            #Save Y into CSV file
            with open(refined_model_save_path + f'/{cell}/{model_name}_{cell}_output.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
                Y.to_csv(f)       

            #Save Best training data into pkl file
            with open(refined_model_save_path + f'/{cell}/{model_name}_{cell}_output.pkl', 'wb') as file:
                pickle.dump(Y, file)

            # Save the Model to pickle file
            with open(refined_model_save_path + f'/{cell}/{model_name}_{cell}_Best_Model.pkl', 'wb') as file: 
                pickle.dump(best_model, file)

if __name__ == "__main__":
    main()