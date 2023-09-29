import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statsmodels.stats.multicomp as mc

def extract_training_data(data_file_path, input_param_names, cell_type, size_cutoff, PDI_cutoff, prefix, RLU_floor = 3):
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
        
        #Remove PDI column from input features
        cell_data.drop(columns = 'PDI', inplace = True)
        input_params.remove('PDI')

    if "Zeta" in  input_params:
        cell_data = cell_data[cell_data.Zeta != 0] #Remove any rows where zeta = 0
    

    #replace all RLU values below 3 to 3 to reduce noise
    cell_data.loc[cell_data[prefix + cell_type] < RLU_floor, prefix + cell_type] = 3 

    print("Input Parameters used:", input_params)
    print("Number of Datapoints used:", len(cell_data.index))

    X = cell_data[input_params]                         
    Y = cell_data[prefix + cell_type].to_numpy()
    scaler = MinMaxScaler().fit(Y.reshape(-1,1))
    temp_Y = scaler.transform(Y.reshape(-1,1))
    Y = pd.DataFrame(temp_Y, columns = ["NORM_" + prefix + cell_type])
    return X,Y, cell_data


def select_input_params(size, zeta):
    #Input parameters
    formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                    'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio'] 
    
    lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
    
    input_param_names = lipid_param_names +  formulation_param_names

    #Add physiochemical parameters to inputparameters
    if size == True:
        input_param_names = input_param_names + ['Size', 'PDI']

    if zeta == True:
        input_param_names = input_param_names + ['Zeta']
    return input_param_names



def run_tukey(data, save_path):

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
    results_df.to_csv(f'{save_path}tukey_results.csv', index=False, )
    
    # # Print the results
    # print(tukey_results)