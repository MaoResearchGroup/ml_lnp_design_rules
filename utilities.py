import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statsmodels.stats.multicomp as mc
from scipy.stats import spearmanr, pearsonr
import scipy.stats as stats
import pickle
from matplotlib import colors
import os
from copy import copy, deepcopy

def init_pipeline(pipeline_path, RUN_NAME, cell, param_type, chemical_type, data_file_path, size_cutoff, PDI_cutoff,prefix, RLU_floor,N_CV, model_list ):
    
    print('\n\n########## INITIALIZING NEW PIPELINE ##############\n\n')
    #Saving/Loading
    model_save_path           = f"{RUN_NAME}{cell}/Trained_Models/" # Where to save model, results, and training data 
    refined_model_save_path   = f"{RUN_NAME}{cell}/Feature_Reduction/" #where to save refined model and results
    shap_value_save_path      = f'{RUN_NAME}{cell}/SHAP_Values/'
    straw_save_path           = f'{RUN_NAME}{cell}/Straw_Models/'
    figure_save_path          = f"{RUN_NAME}{cell}/Figures/" #where to save figures
    
    #Input_Params
    input_param_names = select_input_params(cell = cell, method = param_type, chemical_features = chemical_type)
    print(f"INITIAL INPUT PARAMS: {input_param_names}")
    
    #initialize Pipeline Config and Data Storage Dictionary
    pipeline_dict = {'Cell' : cell,
                    'STEPS_COMPLETED':{
                        'Preprocessing': False,
                        'Model_Selection': False,
                        'Feature_Reduction': False,
                        'Straw_Model': False,
                        'SHAP': False,
                        'Learning_Curve': False
                        },
                    'Saving':{
                        'RUN_NAME': RUN_NAME,
                        'Models': model_save_path,
                        'Refined_Models': refined_model_save_path,
                        'Straw_Models': straw_save_path,
                        'SHAP': shap_value_save_path,
                        'Figures': figure_save_path
                        },
                    'Data_preprocessing': {
                        'Data_Path': data_file_path,
                        'Formula_param_type': param_type,
                        'Input_Params': input_param_names,
                        'Size_cutoff': size_cutoff,
                        'PDI_cutoff': PDI_cutoff,
                        'prefix' : prefix,
                        'RLU_floor':RLU_floor, 
                        'Scaler': None,
                        'X' : None, 
                        'y': None, 
                        'all_proc_data' : None,
                        'raw_data' : None
                        },
                    'Model_Selection': {
                        'Method': 'Nested CV',
                        'N_CV' : N_CV,
                        'Model_list': model_list,
                        'NESTED_CV': {},
                        'Best_Model':{
                            'Model_Name' : None, 
                            'Model': None, 
                            'Hyper_Params': None,
                            'Predictions' : None,
                            'MAE': None,
                            'Spearman':None,
                            'Pearson': None
                            }
                        },
                    'Feature_Reduction':{
                        'Refined_Params': None,
                        'Removed_Params': None,
                        'Refined_X': None,
                        'Refined_Model': None,
                        'Final_Results': None,
                        'Reduction_Results': None
                        },
                    'Straw_Model':{
                        'X': None,
                        'y': None,
                        'Model': None,
                        'N_CV': None,
                        'Results': None
                        },
                    'SHAP': dict()
                    }
  
    ####check save paths ########
    for save_path in pipeline_dict['Saving'].values():
        if not os.path.exists(save_path):
            # Create the directory if it doesn't exist
            os.makedirs(save_path)
            print(f"Directory '{save_path}' created.")
    else:
        print(f"Directory '{save_path}' already exists.")

    with open(pipeline_path , 'wb') as file:
        pickle.dump(pipeline_dict, file)
        print(f"\n\n--- SAVED New {cell} Pipeline CONFIG  ---")
    return pipeline_dict    
def save_pipeline(pipeline, path, step):
    c = pipeline['Cell']
    with open(path , 'wb') as file:
            pickle.dump(pipeline, file)
    print(f"\n--- SAVED PIPELINE: {step} CONFIG AND RESULTS for {c}  ---")

def select_input_params(cell, method = None, chemical_features = None):
    #Input parameters

    #composition parameters
    if method == 'ratio':
        formulation_param_names = ['NP_ratio',
                                   'Chol_DMG-PEG_ratio',
                                   '(IL+HL)',
                                   'Dlin-MC3_Helper lipid_ratio' ] 
    elif method == 'percent':
        formulation_param_names = ['NP_ratio',
                                    'PEG_(Chol+PEG)',
                                    '(IL+HL)',
                                    'HL_(IL+HL)'] 
    elif method == 'weight':
        formulation_param_names = ['wt_Helper', 'wt_Dlin',
                        'wt_Chol', 'wt_DMG', 'wt_pDNA'] 
    else:
        raise "INVALID FORMULATION PARAMETER TYPE"


    #chemical features parameters
    if chemical_features == 'OHE':
        lipid_param_names = ['18PG', '14PA', 'DOPE', 'DSPC', 'DOTAP', 'DDAB']
                #Total carbon tails does not change for any datapoints for these cell
    elif chemical_features == 'HL':
        lipid_param_names = ['P_charged_centers', 
                         'N_charged_centers', 
                         'cLogP', 
                         'Hbond_D', 
                         'Hbond_A', 
                         'Total_Carbon_Tails', 
                         'Double_bonds']
                
    elif chemical_features == 'blended':
        lipid_param_names = ['P_charged_centers', 
                         'N_charged_centers', 
                         'cLogP', 
                         'Hbond_D', 
                         'Hbond_A', 
                         'Total_Carbon_Tails', 
                         'Double_bonds',
                         '18PG', '14PA', 'DOPE', 'DSPC', 'DOTAP', 'DDAB']

    else:
        raise "INVALID CHEMICAL PARAMETER TYPE"
    

    NP_level_params = ['Size', 'PDI', 'Zeta']
    
    input_param_names =  lipid_param_names + NP_level_params + formulation_param_names

    if cell in ['ARPE19','N2a']:
        while "14PA" in input_param_names:
            input_param_names.remove("14PA")
        while "DDAB" in input_param_names:
            input_param_names.remove("DDAB")
        while "Total_Carbon_Tails" in input_param_names:
            input_param_names.remove("Total_Carbon_Tails")

    return input_param_names


def extract_training_data(pipeline):
    #Assign variables based on dictionary
    cell_type = pipeline['Cell']
    data_path = pipeline['Data_preprocessing']['Data_Path']
    input_params = pipeline['Data_preprocessing']['Input_Params']
    size_cutoff = pipeline['Data_preprocessing']['Size_cutoff']
    PDI_cutoff = pipeline['Data_preprocessing']['PDI_cutoff']
    prefix = pipeline['Data_preprocessing']['prefix']
    RLU_floor = pipeline['Data_preprocessing']['RLU_floor']
    
    #Extract datafile
    df = pd.read_csv(data_path)

    #Formatting Training Data
    raw_data = df[['Formula label', 'Helper_lipid'] + input_params + [prefix + cell_type]].copy()
    raw_data = raw_data.dropna() #Remove any NaN rows

    processed_data = raw_data.copy()
    if "Size" in input_params:
        processed_data = processed_data[processed_data.Size != 0] #Remove any rows where size = 0
        processed_data = processed_data[processed_data.Size <= size_cutoff]
        processed_data = processed_data[processed_data.PDI <= PDI_cutoff] #Remove any rows where PDI > cutoff


    if "Zeta" in  input_params:
        processed_data = processed_data[processed_data.Zeta != 0] #Remove any rows where zeta = 0
    

    #floor all RLU values below the noise
    processed_data.loc[processed_data[prefix + cell_type] < RLU_floor, prefix + cell_type] = RLU_floor 

    print("Input Parameters used:", input_params)
    print("Number of Datapoints used:", len(processed_data.index))

    X = processed_data[input_params]                         
    Y = processed_data[prefix + cell_type].to_numpy()
    scaler = MinMaxScaler().fit(Y.reshape(-1,1))
    temp_Y = scaler.transform(Y.reshape(-1,1))
    Y = pd.DataFrame(temp_Y, columns = ["NORM_" + prefix + cell_type])

    #Update Pipeline dictionary
    pipeline['Data_preprocessing']['Scaler'] = scaler
    pipeline['Data_preprocessing']['X'] = X
    pipeline['Data_preprocessing']['y'] = Y
    pipeline['Data_preprocessing']['all_proc_data'] = processed_data
    pipeline['Data_preprocessing']['raw_data'] = raw_data
    pipeline['STEPS_COMPLETED']['Preprocessing'] = True

    return pipeline, X,Y, processed_data

def run_anova(data):
    # Extract the groups from the DataFrame
    groups = []
    for column in data.columns:
        groups.append(data[column].dropna())  # Ensure to drop NaNs that might affect the ANOVA

    # Perform the ANOVA using the collected groups
    f_stat, p_value = stats.f_oneway(*groups)
    
    # # Print the results
    # print("F-statistic:", f_stat)
    # print("P-value:", p_value)

    return f_stat, p_value

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
    results_df.to_csv(f'{save_path}{cell}_tukey_results.csv', index=False)

    return
    

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

def get_mean_shap(c, input_params, shap_values, N_bins = 10):

    mean_shap = pd.DataFrame(columns = input_params)

    #Dataframe of input data to generate SHAP values
    df_input_data = pd.DataFrame(shap_values.data, columns = input_params)

    #Dataframe of shap values
    df_values = pd.DataFrame(shap_values.values, columns = input_params)

    #dataframe to store average SHAP values
    mean_storage = pd.DataFrame(columns = ["Feature", "Feature_Value_bin", "Feature_Value", "Avg_SHAP"])
    
    #List to store best feature values
    best_feature_values = []

    for f in input_params:
        combined = pd.DataFrame()
        combined['Input'] = df_input_data[f]
        combined["SHAP"] = df_values[f]
        #check if a physiochemical feature (continous)
        if f in ['Size', 'Zeta' 'PDI']:
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
        best_feature_values.append((f, best_feature_value, normalized_value))

    df_best_feature_values = pd.DataFrame(best_feature_values, columns = ["Feature", f"{c}_Feature_Value", f"{c}_Norm_Feature_Value"])

    return df_best_feature_values, mean_storage


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

def extraction_all(pipeline, model_name, loop = 'test'):
    if loop == 'test': #Test set
        df = copy(pipeline['Model_Selection']['NESTED_CV'][model_name]['Final_model']['Test_pred'])
        df['Absolute_Error'] = abs(df['Experimental_Transfection'] - df['Predicted_Transfection'])
        df_compiled = df
    elif loop == 'CV': #validation set
        df = pipeline['Model_Selection']['NESTED_CV'][model_name]['HP_tuning_df']
        list_of_dataframes = []

        #Compiled the CV validation across all outerloops
        for n in range (pipeline['Model_Selection']['N_CV']): #Range corresponds to number of outerloop CV iterations
            dataframe = pd.DataFrame(df['Formulation_Index'][n], columns=['Formulation_Index'])
            dataframe['Experimental_Transfection'] = df['Experimental_Transfection'][n]
            dataframe['Predicted_Transfection'] = df['Predicted_Transfection'][n]
            dataframe['Formulation_Index'] = df['Formulation_Index'][n]
            dataframe['Absolute_Error'] = abs(dataframe['Experimental_Transfection'] - dataframe['Predicted_Transfection'])
            list_of_dataframes.append(dataframe)
        
        df_compiled = pd.concat(list_of_dataframes, axis=0, ignore_index=True)
    else:
        raise("INVALID LOOP TYPE")

    return df_compiled
def get_Model_Selection_Error(pipeline, loop):
    
    #Collect error for all best models
    model_list = pipeline['Model_Selection']['Model_list']
    ALL_AE = pd.DataFrame(columns = model_list)

    #Extract and Calculate Absolute Error for each model
    for model_name in model_list:
        ALL_AE[model_name]= extraction_all(pipeline=pipeline, model_name=model_name, loop = loop)['Absolute_Error']

    #Sort models from left to right by increasing MAE
    sorted_index = ALL_AE.mean().sort_values().index
    sorted_AE =ALL_AE[sorted_index]

    return sorted_AE

def get_Model_Selection_performance(pipeline, loop, save):
    #Collect error for all best models
    model_list = pipeline['Model_Selection']['Model_list']
    model_dict = {}
    for model in model_list:
        model_dict[model] = extraction_all(pipeline=pipeline, model_name=model, loop = loop)

    # Dictionary to hold results
    results = {}

    # Compute metrics for each model and store in the results dictionary
    results = {model_name: compute_performance_metrics(data, 'Predicted_Transfection', "Experimental_Transfection") for model_name, data in model_dict.items()}

    # Organize results into a DataFrame and sort by MAE
    results_df = pd.DataFrame(results, index=["MAE Mean", "MAE Std", "Spearman Correlation", "Pearson Correlation"])
    results_df = results_df.T.sort_values("MAE Mean").T

    # Save the DataFrame to a user-defined path
    results_df.to_excel(save)

    return results_df

# Function to calculate the performance metrics
def compute_performance_metrics(df, predicted_column, actual_column):
    # Calculate Mean Absolute Error and its standard deviation
    mae = (df[predicted_column] - df[actual_column]).abs()
    mae_mean = mae.mean()
    mae_std = mae.std()

    # Calculate Spearman's correlation
    spearman_corr, _ = spearmanr(df[predicted_column], df[actual_column])
    
    # Calculate Pearson's correlation
    pearson_corr, _ = pearsonr(df[predicted_column], df[actual_column])
    
    return mae_mean, mae_std, spearman_corr, pearson_corr


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


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap