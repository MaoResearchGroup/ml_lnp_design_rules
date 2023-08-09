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
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from sklearn.metrics import mean_absolute_error

from Nested_CV_reformat import NESTED_CV_reformat

def extract_training_data(data_path, input_params, cell, prefix, size_zeta, PDI_cutoff):
    #Load Training data
    df = pd.read_csv(data_path)
    #Remove unnecessary rows
    cell_data = df[['Formula label', 'Helper_lipid'] + input_params + [prefix + cell]]
    cell_data = cell_data.dropna() #Remove any NaN rows
    if size_zeta == True:
        cell_data = cell_data[cell_data.Size != 0] #Remove any rows where size = 0
        cell_data = cell_data[cell_data.Zeta != 0] #Remove any rows where zeta = 0
        cell_data = cell_data[cell_data.PDI <= PDI_cutoff] #Remove any rows where PDI > 0.45

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
    np.savetxt("dist_linkage.csv", dist_linkage, delimiter=",")

    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X.columns.tolist(), leaf_rotation=90
    )

    #Save Figure
    plt.savefig(save + f'{cell}/Feature_Correlation_Plot.png', dpi=300, bbox_inches='tight')
    #plt.show()


    return dist_linkage

def eval_feature_reduction(dist_linkage, X_features, Y, model, N_CV):
    N_features = len(X_features.columns)
    MAE_list = [] # empty list to store MAE values
    MAE_std_list = [] # empty list to store MAE values
    spear_list = []
    spear_std_list = []
    pear_list = []
    pear_std_list = []

    feature_name_list = [] # empty list to store features names
    feature_number_list = [] # empty list to store number of features
    linkage_distance_list = [] # empty list to store the Ward'slinkage distance

    for n in range(0, N_features, 1):
        # select input features to be included in this model iteration based on Ward's linkage of n/10
        cluster_ids = hierarchy.fcluster(dist_linkage, (n/N_features), criterion="distance") 
        cluster_id_to_feature_ids = defaultdict(list) 
        
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
                
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        linkage_distance_list.append(n/N_features) # append linkage distance to empty list

        tested_features = []  # create empty list to save feature names
            
        for feature in selected_features: # for loop to append the utilized input feature names to the empty list
            tested_features.append(X_features.columns[feature])
                
        feature_number_list.append(len(tested_features)) # append the number of input features to empty list
        feature_name_list.append(tested_features) # append the list of feature names to an empty list of lists
                
        

        acc_results, spearman_results, pearson_results = evaluate_model(X_features, Y, selected_features, model, N_CV )
        MAE_list.append(np.mean(acc_results)) # append average MAE value to empty list
        MAE_std_list.append(np.std(acc_results)) # append average MAE value to empty list

        spear_list.append(np.mean(spearman_results)) # append average MAE value to empty list
        spear_std_list.append(np.std(spearman_results)) # append average MAE value to empty list

        pear_list.append(np.mean(pearson_results)) # append average MAE value to empty list
        pear_std_list.append(np.std(pearson_results)) # append average MAE value to empty list

                    
        print('\n################################################################\n\nSTATUS REPORT:') 
        print('Iteration '+str(n+1)+' of '+str(N_features)+' completed') 
        print('Test_Score: %.3f' % (np.mean(acc_results)))
        print('Spearman_Score: %.3f' % (np.mean(spearman_results)))
        print('Pearson_Score: %.3f' % (np.mean(pearson_results)))
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
    
    return results_df


#Use to retrain current models with different feature data
def evaluate_model(X,Y, selected_features, model, N_CV):
    acc_list = []
    spearman_list = []
    pearson_list = []

    #Kfold CV
    print("\nSELECTED FEATURES: ", selected_features)
    for i in range(10): #For loop that splits and evaluates the data ten times

        print(f"\n LOOP: {i+1}/10")

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
            clf_sel.fit(X_train_sel, y_train) # fit the selected model with the training set
            y_pred = clf_sel.predict(X_test_sel) # predict test set based on selected input features

            # Get Model Statistics
            acc = round(mean_absolute_error(y_pred, y_test), 3)
            spearman = stats.spearmanr(y_pred, y_test)[0]
            pearson = stats.pearsonr(y_pred, y_test)[0]

            # Append model performance results to associated list
            acc_list.append(acc)
            spearman_list.append(spearman) 
            pearson_list.append(pearson) 

    return acc_list, spearman_list, pearson_list

#Use if wanted to retrain and reoptimize models with different features
def retrain_feature_reduction_CV(model_name, data_file_path, save_path, cell, wt_percent, size_zeta, CV,input_param_names, feature):

  """
  Function that:
  - runs the NESTED_CV for a desired model in the class, cell type, and for a given number of folds
  - default is 10-folds i.e., CV = None. CV = # Trials... # outerloop repeats
  - prints status and progress of NESTED_CV
  - formats the results as a datafarme, and saves them locally
  - assigns the best HPs to the model, trains, and saves its locally
  - then returns the results dataframe and the saved model
  """
  if __name__ == '__main__':
    model_instance = NESTED_CV_reformat(data_file_path, model_name)
    model_instance.input_target(cell,size_zeta, input_param_names)
    model_instance.cross_validation(CV)
    model_instance.results()
    model_instance.best_model() 

    # Check if save path exists (if not, then create path)
    if os.path.exists(save_path + f'{model_name}/{cell}/') == False:
       os.makedirs(save_path + f'{model_name}/{cell}/', 0o666)

    # Save Tuning Results CSV
    with open(save_path + f'{cell}/' + str(len(input_param_names)) + "_TotalFeatures_" + feature + 'Preserved' + '_HP_Tuning_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
      model_instance.CV_dataset.to_csv(f)
    
    # Save Tuning Results PKL
    model_instance.CV_dataset.to_pickle(save_path + f'{model_name}/{cell}/' +str(len(input_param_names)) + "_TotalFeatures_" + feature + 'Preserved' + '_HP_Tuning_Results.pkl', compression='infer', protocol=5, storage_options=None) 
    
    # Save the Model to pickle file
    with open(save_path + f'{model_name}/{cell}/' + str(len(input_param_names)) + "_TotalFeatures_" + feature + 'Preserved' + '_Trained.pkl', 'wb') as file: 
          pickle.dump(model_instance.best_model, file)

    # Save the Training Data used to .pkl
    with open(save_path + f'{model_name}/{cell}/' +str(len(input_param_names)) + "_TotalFeatures_" + feature + 'Preserved' + '_Training_Data.pkl', 'wb') as file:
          pickle.dump(model_instance.cell_data, file)

    # Save the Training Data used to csv
    with open(save_path + f'{model_name}/{cell}/' +str(len(input_param_names)) + "_TotalFeatures_" + feature + 'Preserved' + '_Training_Data.csv', 'w', encoding = 'utf-8-sig') as file:
          model_instance.cell_data.to_csv(file)
    
    print('Sucessfully save NESTED_CV Results, Final Model, and Training dataset')
    return model_instance


#define a function called plot_feature_reduction 
def plot_feature_reduction(stats_df, cell_type, model_name, save):

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor='white')
    ax2 = ax1.twinx()

    # Plot the points with error bars for Average MAE
    ax1.errorbar(stats_df['# of Features'], stats_df['MAE'], yerr=stats_df['MAE_std'], fmt='o', color='black',
                ecolor='darkgray', elinewidth=2, capsize=4, capthick=2, label='Average MAE')

    # Draw a line connecting the points for Average MAE
    ax1.plot(stats_df['# of Features'], stats_df['MAE'], color='blue', linewidth=1)

    # Plot error bars for Spearman correlation coefficient
    ax2.errorbar(stats_df['# of Features'], stats_df['Spearman'], yerr=stats_df['Spearman_std'], fmt='o', color='red',
                ecolor='darkgray', elinewidth=2, capsize=4, capthick=2, label='Average Spearman')

    # Draw a line connecting the points for Spearman correlation coefficient
    ax2.plot(stats_df['# of Features'], stats_df['Spearman'], color='red', linewidth=1)

    # Plot error bars for Pearson correlation coefficient
    ax2.errorbar(stats_df['# of Features'], stats_df['Pearson'], yerr=stats_df['Pearson_std'], fmt='o', color='green',
                ecolor='darkgray', elinewidth=2, capsize=4, capthick=2, label='Average Pearson')

    # Draw a line connecting the points for Pearson correlation coefficient
    ax2.plot(stats_df['# of Features'], stats_df['Pearson'], color='green', linewidth=1)

    # Set labels for the x-axis and y-axes
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('Average MAE')
    ax2.set_ylabel('Correlation Coefficients')

    # Reverse the x-axis
    ax1.invert_xaxis()

    # Adjust font size and style
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    # Set integer labels on the x-axis
    ax1.set_xticks(range(int(stats_df['# of Features'].min()), int(stats_df['# of Features'].max()) + 1))

    # Combine the legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc='upper left')

    # Update the legend titles
    ax1.legend(lines, ['Average MAE', 'Average Spearman', 'Average Pearson'], loc='center left')

    # Save the plot as a high-resolution image (e.g., PNG or PDF)
    plt.savefig(save + f'{cell_type}/{cell_type}_{model_name}Feature_Reduction_Plot.png', dpi=300, bbox_inches='tight')

    # Show the plot
    #plt.show()





################ Model Training ##############################################
cell_names = ['ARPE19','N2a','PC3','B16','HEK293','HepG2'] #'ARPE19','N2a',
model_list = ['LGBM', 'XGB', 'RF']
size_zeta = False

PDI = 1
N_CV = 5

################ Global Variables ##############################################
data_file_path = "Raw_Data/10_Master_Formulas.csv"
model_folder = "Trained_Models/Models_Size_Zeta_PDI/"
save_path = "Trained_Models/Feature_reduction_NoSizeZeta/" # Where to save new models, results, and training data






########### MAIN ####################################

def main():

    # #Rerun model training based on clusters
    for cell in cell_names:

        #Features to use
        formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                      'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio']
        
        if cell in ['ARPE19','N2a']:
            #Total_Carbon_Tails Removed (It does not change in the formulations)
            lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA',
                                'Hbond_D', 'Hbond_A', 'Double_bonds'] 
        else:
            lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA',
                                'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
            
        if size_zeta == True:
            input_param_names = lipid_param_names +formulation_param_names +  ['Size', 'Zeta', 'PDI']
        else:
            input_param_names = lipid_param_names+ formulation_param_names 


        #Get Training Data for cell
        Train_X, Y = extract_training_data(data_file_path, input_param_names, cell, "RLU_", size_zeta, PDI)

        #Check/create correct save path
        if os.path.exists(save_path + f'/{cell}') == False:
            os.makedirs(save_path + f'/{cell}', 0o666)


        #Calculate and Plot Feature Correlation
        dist_link = feature_correlation(Train_X, cell, save_path)

        for model_name in model_list:

            #Open correct model
            model_path = model_folder + f'{model_name}/{cell}/{model_name}_{cell}_Trained.pkl'
            with open(model_path, 'rb') as file: # import trained model
                trained_model = pickle.load(file)
            
            #Test and save models using new Feature clusters 
            results = eval_feature_reduction(dist_link, Train_X, Y, trained_model, N_CV)

            #Plot feature reduction results
            plot_feature_reduction(results, cell, model_name, save_path)

            #Save Results into CSV file
            with open(save_path + f'/{cell}/{model_name}_{cell}_Feature_Red_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
                results.to_csv(f)

    #         stats_df = pd.DataFrame(columns=['Feature_No','Cluster Index','Cluster','Feature Preserved','Average MAE', 'Std MAE', 'Average Pearson', 'Std Pearson', 'Average Spearman', 'Std Spearman'])
    #         model_instance = feature_reduction_CV(model_name, data_file_path, save_path, cell_type_name, wt_percent, size_zeta, N_CV, input_param_names, 'All')
    #         average_mae, std_mae, average_pearson, std_pearson, average_spearman, std_spearman = model_instance.stats()
    #         Feature_No = len(input_param_names)
    #         stats_df = stats_df.append({
    #             'Feature_No': Feature_No,
    #             'Cluster Index': 0,
    #             'Cluster': 'All',
    #             'Feature Preserved': 'All',
    #             'Average MAE': average_mae,
    #             'Std MAE': std_mae,
    #             'Average Pearson': average_pearson,
    #             'Std Pearson': std_pearson,
    #             'Average Spearman': average_spearman,
    #             'Std Spearman': std_spearman
    #         }, ignore_index=True)


    #         reduced_params = input_param_names
    #         idx = 1; 
    #         for cluster in clusters: 
    #             best_feature = ''
    #             best_mae = 10000000000
    #             temp_params = input_param_names
    #             for feature in cluster:
    #                 if feature in reduced_params:
    #                     print('Feature Preserved in Cluster' + str(idx) + ": ", feature)
    #                     features_to_exclude = [col for col in cluster if col not in feature]
    #                     selected_columns = [col for col in reduced_params if col not in features_to_exclude]
    #                     model_instance = feature_reduction_CV(model_name, data_file_path, save_path, cell_type_name, wt_percent, size_zeta, N_CV, selected_columns, feature)
    #                     average_mae, std_mae, average_pearson, std_pearson, average_spearman, std_spearman = model_instance.stats()
    #                     if average_mae < best_mae:
    #                         best_feature = feature
    #                         best_mae = average_mae
    #                         temp_params = selected_columns
    #             Feature_No = len(selected_columns)
    #             stats_df = stats_df.append({
    #             'Feature_No': Feature_No,
    #             'Cluster Index': idx,
    #             'Cluster': cluster,
    #             'Feature Preserved': best_feature,
    #             'Average MAE': average_mae,
    #             'Std MAE': std_mae,
    #             'Average Pearson': average_pearson,
    #             'Std Pearson': std_pearson,
    #             'Average Spearman': average_spearman,
    #             'Std Spearman': std_spearman
    #             }, ignore_index=True)
    #             reduced_params = temp_params
    #             idx += 1

    #         # save stats_df to csv
    #         with open(save_path + f'/{cell_type_name}/Feature_Reduction_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
    #             stats_df.to_csv(f)
    #         plot_feature_reduction(stats_df, cell_type_name)
if __name__ == "__main__":
    main()