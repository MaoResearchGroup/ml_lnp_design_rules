import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statsmodels.stats.multicomp as mc
from scipy.stats import spearmanr
import pickle

def extract_training_data(data_file_path, input_param_names, cell_type, size_cutoff, PDI_cutoff, keep_PDI, prefix, RLU_floor = 3):
    print("############ EXTRACTING DATA ###############")
    #Extract datafile
    df = pd.read_csv(data_file_path)

    #copy input parameters
    input_params = input_param_names.copy()
    #Formatting Training Data
    cell_data = df[['Formula label', 'Helper_lipid'] + input_params + [prefix + cell_type]]
    cell_data = cell_data.dropna() #Remove any NaN rows

    if "Size" in input_params:
        cell_data = cell_data[cell_data.Size != 0] #Remove any rows where size = 0
        cell_data = cell_data[cell_data.Size <= size_cutoff]
        cell_data = cell_data[cell_data.PDI <= PDI_cutoff] #Remove any rows where PDI > cutoff
        
        if keep_PDI == False:
            #Remove PDI column from input features
            cell_data.drop(columns = 'PDI', inplace = True)
            input_params.remove('PDI')

    if "Zeta" in  input_params:
        cell_data = cell_data[cell_data.Zeta != 0] #Remove any rows where zeta = 0
    

    #floor all RLU values below the noise
    cell_data.loc[cell_data[prefix + cell_type] < RLU_floor, prefix + cell_type] = RLU_floor 

    print("Input Parameters used:", input_params)
    print("Number of Datapoints used:", len(cell_data.index))

    X = cell_data[input_params]                         
    Y = cell_data[prefix + cell_type].to_numpy()
    scaler = MinMaxScaler().fit(Y.reshape(-1,1))
    temp_Y = scaler.transform(Y.reshape(-1,1))
    Y = pd.DataFrame(temp_Y, columns = ["NORM_" + prefix + cell_type])
    return X,Y, cell_data


def select_input_params(size, zeta, ratio = True):
    #Input parameters
    if ratio:
        formulation_param_names = ['NP_ratio',
                                   'Chol_DMG-PEG_ratio',
                                   'Dlin-MC3+Helper lipid percentage',
                                   'Dlin-MC3_Helper lipid_ratio' ] 
    else:
        formulation_param_names = ['wt_Helper', 'wt_Dlin',
                        'wt_Chol', 'wt_DMG', 'wt_pDNA'] 
    
    lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
    
    input_param_names = lipid_param_names + formulation_param_names

    #Add physiochemical parameters to inputparameters
    if size == True:
        input_param_names = input_param_names + ['Size', 'PDI']

    if zeta == True:
        input_param_names = input_param_names + ['Zeta']
    return input_param_names



def run_tukey(data, save_path, cell):

   # Extract the absolute error values for each model
    absolute_errors = [data[model].tolist() for model in data.columns]
    
    # Flatten the list of absolute errors
    all_errors = np.concatenate(absolute_errors)
    
    # Create a list of group labels
    group_labels = [model for model in data.columns for _ in range(len(data))]
    
    # Perform Tukey's HSD test
    tukey_results = mc.MultiComparison(all_errors, group_labels).tukeyhsd()
    
    # Convert the results to a DataFrame
    results_df = pd.DataFrame(tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])
    
    # Save the results to a CSV file
    results_df.to_csv(f'{save_path}{cell}_tukey_results.csv', index=False, )
    
    # # Print the results
    # print(tukey_results)

def get_spearman(data, cell1, cell2):
    if cell1 == cell2:
        corr = 1
    else:
        temp_data = data.loc[:,["RLU_"+cell1, "RLU_"+cell2]]
        temp_data.dropna(inplace= True)
        if temp_data.empty:
            corr = "N/A"
        corr = spearmanr(temp_data).correlation
    return corr

def get_mean_shap(c, model, model_save_path, shap_value_path, N_bins = 10):
    #Get feature names used to train model
    with open(model_save_path + f"{c}/{model}_{c}_Best_Model_Results.pkl", 'rb') as file: # import best model results
                best_results = pickle.load(file)
    input_param_names = best_results.loc['Feature names'][0]  

    mean_shap = pd.DataFrame(columns = input_param_names)
    #Get SHAP Values
    with open(shap_value_path + f"{model}_{c}_SHAP_values.pkl", "rb") as file:   # Unpickling
        shap_values = pickle.load(file)

    #Dataframe of input data to generate SHAP values
    df_input_data = pd.DataFrame(shap_values.data, columns = input_param_names)

    #Dataframe of shap values
    df_values = pd.DataFrame(shap_values.values, columns = input_param_names)

    #dataframe to store average SHAP values
    mean_storage = pd.DataFrame(columns = ["Feature", "Feature_Value_bin", "Feature_Value", "Avg_SHAP"])
    
    #List to store best feature values
    best_feature_values = []

    for f in input_param_names:
        combined = pd.DataFrame()
        combined['Input'] = df_input_data[f]
        combined["SHAP"] = df_values[f]
        #check if a physiochemical feature (continous)
        if f in ["Size", "Zeta"]:
            #bins
            feature_bins = create_bins(int(np.floor(combined['Input'].min())),int(np.ceil(combined['Input'].max())), N_bins)
            #print(feature_bins)
            bin_means = []
            for bin in feature_bins:
                bin_means.append(np.mean(bin))

            binned_inputs = []
            for value in combined['Input']:
                bin_index = find_bin(value, feature_bins)
                binned_inputs.append(bin_index)
            combined['bins'] = binned_inputs
            unique_bins = combined['bins'].unique()
            #Iterate through unique feature values to get average SHAP for that feature value
            for bins in unique_bins:
                #Get the mean shap value for the unique feature value
                bin_mean = combined.loc[combined['bins'] == bins, 'SHAP'].mean()
                #Store the mean shap value with the mean bin value
                mean_storage.loc[len(mean_storage)] = [f, feature_bins[bins], np.mean(feature_bins[bins]), bin_mean]
                            #composition or helper lipid features which are more categorical
        else:
            # #Create x,y plot of the feature value and 
            # line = plot_line(combined,'Input',"SHAP")
            unique_feature_values = df_input_data[f].unique()
            #Iterate through unique feature values to get average SHAP for that feature value
            for feature_value in unique_feature_values:
                #Get the mean shap value for the unique feature value
                feature_mean = df_values.loc[df_input_data[f] == feature_value, f].mean()

                #Store the mean shap value
                mean_storage.loc[len(mean_storage)] = [f, feature_value, feature_value, feature_mean]
            
                        #Find the feature value with the max average shap value and save normalized fraction
        best_feature_value = mean_storage['Feature_Value'][mean_storage.loc[mean_storage['Feature'] == f, "Avg_SHAP"].astype(float).idxmax()]
        min = mean_storage.loc[mean_storage['Feature'] == f, 'Feature_Value'].min()
        max = mean_storage.loc[mean_storage['Feature'] == f, 'Feature_Value'].max()
        normalized_value = (best_feature_value - min)/(max-min)
        best_feature_values.append((f, normalized_value))

    df_best_feature_values = pd.DataFrame(best_feature_values, columns = ["Feature", f"{c}_Norm_Feature_Value"])
    #print(df_best_feature_values)

        

    #Save average shap of the features as csv
    with open(shap_value_path + f"/{model}_{c}_mean_shap.csv", 'w', encoding = 'utf-8-sig') as f:
        mean_storage.to_csv(f)
    
    #Save average shap of the features as csv
    with open(shap_value_path + f"/{model}_{c}_best_feature_values.csv", 'w', encoding = 'utf-8-sig') as f:
        df_best_feature_values.to_csv(f)

    return df_best_feature_values


def create_bins(lower_bound, upper_bound, quantity):

    bins = []
    boundaries = np.linspace(lower_bound, upper_bound, quantity)
    for i in range(len(boundaries)-1):
        bins.append((boundaries[i],boundaries[i+1] ))
    return bins

def find_bin(value, bins):
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1

def extraction_all(name, cell, model_path, N_CV):
    '''
    function that extracts and compiles a results dataframe as well as an 
    absolute error array for all modesl in NESTED_CV_results pickle files
    '''
    df = pd.read_pickle(model_path + f"{name}/{cell}/{name}_{cell}_Best_Model_Results.pkl", compression='infer', storage_options=None)
    list_of_dataframes = []
    
    for n in range (N_CV): #Range corresponds to number of outerloop iterations
        dataframe = pd.DataFrame(df['Formulation_Index'][n], columns=['Formulation_Index'])
        dataframe['Experimental_Transfection'] = df['Experimental_Transfection'][n]
        dataframe['Predicted_Transfection'] = df['Predicted_Transfection'][n]
        dataframe['Formulation_Index'] = df['Formulation_Index'][n]
        dataframe['Absolute_Error'] = abs(dataframe['Experimental_Transfection'] - dataframe['Predicted_Transfection'])
        pd_series = dataframe['Absolute_Error']
        list_of_dataframes.append(dataframe)
    
    dataframe_all = pd.concat(list_of_dataframes, axis=0, ignore_index=True)
    
    return dataframe_all

def get_best_model_cell(figure_save_path, model_folder, cell_type_list):
    df = pd.DataFrame(columns = ['Cell_Type', 'Model', 'Test Score'])
    for cell in cell_type_list:
        data = pd.read_csv(figure_save_path + f"{cell}_Boxplot_dataset.csv")
        MAE = data.mean(axis = 0)
        df.loc[len(df)] = [cell, MAE.index[0], MAE[0]]
    df.sort_values(by = 'Test Score', inplace = True)
    with open(model_folder + "Best_Model_Cell.csv", 'w', encoding = 'utf-8-sig') as f:
        df.to_csv(f, index=False)
    print('Saved Best_Model_Cell')
    return df
        #best_model_for_cell.append([])

        # cell_df.sort_values(by = ['Test Score', 'Score_difference'], inplace = True)
        # new_df = cell_df.reindex(columns = ['Cell_Type','Model', 'Test Score'])
        # best_model_for_cell = best_model_for_cell.append(new_df.iloc[0])


        # cell_model.drop(cell_model.columns[0], axis=1, inplace = True)
        # cell_model.drop(cell_model.columns[2], axis=1, inplace = True)
        # best_cell_model = list(cell_model.itertuples(index=False, name=None))
        # print(best_cell_model)
