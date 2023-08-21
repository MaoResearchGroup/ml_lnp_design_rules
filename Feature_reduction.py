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
from copy import deepcopy

from Nested_CV_reformat import NESTED_CV_reformat

def extract_training_data(data_path, input_params, cell, prefix, size_zeta, size_cutoff, PDI_cutoff):
    #Load Training data
    df = pd.read_csv(data_path)
    #Remove unnecessary columns
    cell_data = df[['Formula label', 'Helper_lipid'] + input_params + ["PDI"] + [prefix + cell]]
    cell_data = cell_data.dropna() #Remove any NaN rows
    if size_zeta == True:
        cell_data = cell_data[cell_data.Size != 0] #Remove any rows where size = 0
        cell_data = cell_data[cell_data.Size <= size_cutoff]
        cell_data = cell_data[cell_data.Zeta != 0] #Remove any rows where zeta = 0
        cell_data = cell_data[cell_data.PDI <= PDI_cutoff] #Remove any rows where PDI > CUTOFF
        #Remove PDI column from input features
        cell_data.drop(columns = 'PDI', inplace = True)

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
    np.savetxt(save + f"{cell}/dist_linkage.csv", dist_linkage, delimiter=",")

    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X.columns.tolist(), leaf_rotation=90
    )

    #Save Figure
    plt.savefig(save + f'{cell}/Feature_Correlation_Plot.png', dpi=300, bbox_inches='tight')
    #plt.show()


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
            
            best_model.fit(X_features.iloc[:,selected_features], Y) #Fit best model using all data
            best_training_data = X_features.copy(deep=True)

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

    plt.close()
    # Show the plot
    #plt.show()





################ Model Training ##############################################
cell_names = ['ARPE19','N2a','PC3','B16','HEK293','HepG2'] #'ARPE19','N2a',
model_list = ['LGBM', 'XGB', 'RF']
size_zeta = True
PDI = 0.45
size_cutoff = 600
N_CV = 5

################ Global Variables ##############################################
data_file_path = "Raw_Data/10_Master_Formulas.csv"
model_folder = "Trained_Models/Models_Size_600_Zeta_PDI_0.45/"
save_path = "Feature_Reduction/Feature_reduction_Size_600_Zeta_PDI_0.45/" # Where to save new models, results, and training data

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
            input_param_names = lipid_param_names +formulation_param_names +  ['Size', 'Zeta']
        else:
            input_param_names = lipid_param_names+ formulation_param_names 


        #Get Training Data for cell
        Train_X, Y = extract_training_data(data_file_path, input_param_names, cell, "RLU_", size_zeta, size_cutoff, PDI)

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
            results, best_results, best_model, best_data = eval_feature_reduction(dist_link, Train_X, Y, trained_model, N_CV)

            #Plot feature reduction results
            plot_feature_reduction(results, cell, model_name, save_path)

            #Save Results into CSV file
            with open(save_path + f'/{cell}/{model_name}_{cell}_Feature_Red_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
                results.to_csv(f)

            #Save Best Results into CSV file
            with open(save_path + f'/{cell}/{model_name}_{cell}_Best_Model_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
                best_results.to_csv(f)

            #Save Best training data into CSV file
            with open(save_path + f'/{cell}/{model_name}_{cell}_Best_Training_Data.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
                best_data.to_csv(f)       

            #Save Best training data into pkl file
            with open(save_path + f'/{cell}/{model_name}_{cell}_Best_Training_Data.pkl', 'wb') as file:
                pickle.dump(best_data, file)

            #Save Best Results into pkl file
            with open(save_path + f'/{cell}/{model_name}_{cell}_Best_Model_Results.pkl', 'wb') as file:
                pickle.dump(best_results, file)

            # Save the Model to pickle file
            with open(save_path + f'/{cell}/{model_name}_{cell}_Best_Model.pkl', 'wb') as file: 
                pickle.dump(best_model, file)

if __name__ == "__main__":
    main()