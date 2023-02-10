# Set up
import pandas as pd
import numpy as np
import sympy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
from itertools import product
from operator import truediv

"""**ALL FUNCTIONS HERE:**

**Data Processing**
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
def init_data(filepath,cell_type_names):
     """ Takes a full file path to a spreadsheet and an array of cell type names. 
     Returns a dataframe with 0s replaced with 1s."""
     df = pd.read_csv(filepath)
     for cell_type in df.columns[-len(cell_type_names):]:
       zero_rows = np.where(df[cell_type] == 0)
       for i in zero_rows:
         df[cell_type][i] = 1
     return df

def apply_minmaxscaling(X, y):
  scaler = MinMaxScaler()
  scaler.fit(y.reshape(-1,1))
  temp = scaler.transform(y.reshape(-1,1))
  y = temp.flatten()
  return X, y, scaler

def write_df_to_sheets(data_df, save_path):
    
  with open(save_path, 'w', encoding = 'utf-8-sig') as f:
    data_df.to_csv(f)

    return

def create_X_prediction_array(initial_df, lipid_param_names, formulation_param_names,  wt_percentage = False):
  helper_lipid_names = initial_df['Helper_lipid'].drop_duplicates().to_numpy()
  #Extract lipid parameterizations from inital dataset
  lipid_parameterization = initial_df.loc[:, initial_df.columns.isin(['Helper_lipid'] + lipid_param_names)].drop_duplicates(keep='first').reset_index(drop= True).set_index('Helper_lipid')
  # These values are rather arbitrary right now.... is there a better way?
  formulation_params_dic = {'Helper_lipid' : helper_lipid_names,
                            'NP_ratio' : [5,6,7,9,10,11],
                            'Dlin-MC3_Helper lipid_ratio' : [5, 30, 75, 125, 150, 175],
                            'Dlin-MC3+Helper lipid percentage': [30, 50, 70], 
                            'Chol_DMG-PEG_ratio': [25, 50, 75, 200, 300, 400]}
  
  formulation_param_array = pd.DataFrame([row for row in product(*formulation_params_dic.values())], 
                       columns=formulation_params_dic.keys())
   
  # Remove formulations that are present in Training dataset
  # train_test_array = initial_df[['Helper_lipid', 'NP_ratio', 'Dlin-MC3:Helper lipid', 'Dlin-MC3+Helper lipid percentage', 'Chol:DMG-PEG']]                   
  # formulation_param_array = pd.concat([formulation_param_array, train_test_array], axis = 0)
  # formulation_param_array.drop_duplicates(subset = ['Helper_lipid', 'NP_ratio', 'Dlin-MC3:Helper lipid', 'Dlin-MC3+Helper lipid percentage', 'Chol:DMG-PEG'], keep=False, inplace = True)
  # formulation_param_array.reset_index(drop= True, inplace = True)
  if wt_percentage == True:
    #Convert Parameterizations to wt%
    wt_percent_param_array = np.array(['Helper_lipid', 'wt_Helper', 'wt_Dlin', 'wt_Chol', 'wt_DMG', 'wt_pDNA'])
    for row in formulation_param_array.to_numpy():
        _, wt = convert_to_wt_percentage(row[0], float(row[2]), float(row[4]), float(row[1]), float(row[3]))
        new_row = np.concatenate((row[:1], wt, row[5:]))
        wt_percent_param_array = np.vstack([wt_percent_param_array, new_row])
    formulation_param_array = pd.DataFrame(wt_percent_param_array[1:], columns = ['Helper_lipid', 'wt_Helper', 'wt_Dlin', 'wt_Chol', 'wt_DMG', 'wt_pDNA'])
    
  #Add lipid parameterization for each helper lipid to formulation parameter array
  for parameter in lipid_param_names:
    for helper_lipid in helper_lipid_names:
      formulation_param_array.loc[formulation_param_array.Helper_lipid == helper_lipid, parameter] = lipid_parameterization.loc[helper_lipid, parameter]
  print('Number of new formulations: ', len(formulation_param_array.index))
  return formulation_param_array[['Helper_lipid'] + lipid_param_names + formulation_param_names], helper_lipid_names #reorder columns

def get_best_predictions(input_array, best_model, scaler, input_param_names, helper_lipid_names, cell_type, name, save_path):

  #Run ML on input_array to get untested formulation predictions
  y_predictions = best_model.predict(input_array[input_param_names])
  
  #Store Y_predictions by training fold used
  prediction_results = input_array.copy()
  prediction_results['Cell_Type'] = [cell_type]*len(y_predictions)
  prediction_results['Scaled_Predicted_Values'] = y_predictions
  prediction_results['RLU_Predicted_Values'] = scaler.inverse_transform(y_predictions.reshape(-1, 1))

  #Generate In Vitro Validation Formulation List
  generate_iv_validation_list(prediction_results, helper_lipid_names, cell_type, name, save_path)
  
  return prediction_results

def generate_iv_validation_list(predictions, helper_lipid_names, cell_type, name, save_path):
   #Extract High/Mid/Low Transfection Predictions
  if cell_type == 'HEK293':
    high_bar = 0.7
    low_bar = 0.3
  elif cell_type == 'B16':
    high_bar = 0.7
    low_bar = 0.15
  else:
    high_bar = 0.66
    low_bar = 0.33
  

  high_pred_formulations = predictions.loc[predictions['Scaled_Predicted_Values'] >= high_bar]
  mod_pred_formulations = predictions.loc[(predictions['Scaled_Predicted_Values'] < high_bar) & (predictions['Scaled_Predicted_Values'] > low_bar)]
  low_pred_formulations = predictions.loc[predictions['Scaled_Predicted_Values'] <= low_bar]

  #Extract Random Formulations to Test
  num_formulations = 46
  random_high_formulations = pd.DataFrame()
  random_low_formulations = pd.DataFrame()
  all_random_formulations = pd.concat([high_pred_formulations.sample(n = num_formulations), low_pred_formulations.sample(n = num_formulations)])


  #Redo randomized search until conditions are met
  min_lipid = 12
  min_Dlin_helper_ratio = 3
  min_Dlin_helper_percent = 2
  min_Chol_PEG = 3
  min_NP_ratio = 3
  iter = 0
  max_iter = 50000
  while not(all(len(all_random_formulations[all_random_formulations['Helper_lipid'] == lipid]) >= min_lipid for lipid in helper_lipid_names) and 
            all(all_random_formulations.loc[all_random_formulations['Helper_lipid'] == lipid, 'Dlin-MC3_Helper lipid_ratio'].unique().size >= min_Dlin_helper_ratio for lipid in helper_lipid_names) and
            all(all_random_formulations.loc[all_random_formulations['Helper_lipid'] == lipid, 'Chol_DMG-PEG_ratio'].unique().size >= min_Chol_PEG for lipid in helper_lipid_names) and
            all(all_random_formulations.loc[all_random_formulations['Helper_lipid'] == lipid, 'Dlin-MC3+Helper lipid percentage'].unique().size >= min_Dlin_helper_percent for lipid in helper_lipid_names) and
            all(all_random_formulations.loc[all_random_formulations['Helper_lipid'] == lipid, 'NP_ratio'].unique().size >= min_NP_ratio for lipid in helper_lipid_names)): 

        
    all_random_formulations = pd.concat([high_pred_formulations.sample(n = num_formulations), low_pred_formulations.sample(n = num_formulations)]) #Take new samples from high and low lists
    iter += 1
    if iter > max_iter:
      print(f'Could not find any samples that met condition after {max_iter} combinations')
      break
  all_random_formulations.sort_values(by = ['Helper_lipid'], inplace = True)
  print('Final Iter #: ', iter)  
  print('Total Unique Helper Lipids:', all_random_formulations['Helper_lipid'].unique().size, '/', predictions['Helper_lipid'].unique().size)
  print('Total Unique Dlin:helper :', all_random_formulations['Dlin-MC3_Helper lipid_ratio'].unique().size, '/', predictions['Dlin-MC3_Helper lipid_ratio'].unique().size)
  print('Total Unique Dlin + helper :', all_random_formulations['Dlin-MC3+Helper lipid percentage'].unique().size, '/', predictions['Dlin-MC3+Helper lipid percentage'].unique().size)
  print('Total Unique Chol:DMG :', all_random_formulations['Chol_DMG-PEG_ratio'].unique().size, '/', predictions['Chol_DMG-PEG_ratio'].unique().size)
  print('Total Unique NP :', all_random_formulations['NP_ratio'].unique().size, '/', predictions['NP_ratio'].unique().size)

  # print('High', high_pred_formulations)
  print('High Helper Lipids', high_pred_formulations['Helper_lipid'].unique())
  print('Number of high Formulations:', len(high_pred_formulations.index))
  # print('Moderate', high_pred_formulations)
  # print('Number of Moderate Formulations:', len(mod_pred_formulations.index))
  # print('Low', low_pred_formulations) 
  print('Helper Lipids', low_pred_formulations['Helper_lipid'].unique())
  print('Number of low Formulations:', len(low_pred_formulations.index))
  
  #save predictions
  write_df_to_sheets(predictions, save_path + f'{name}_All_Predictions_{cell_type}.csv')
  write_df_to_sheets(high_pred_formulations, save_path + f'{name}_High_Predictions_{cell_type}.csv')
  write_df_to_sheets(low_pred_formulations, save_path + f'{name}_Low_Predictions_{cell_type}.csv')

  #save randomly sampled formulation list
  write_df_to_sheets(all_random_formulations, save_path + f'{name}_In_Vitro_Formulation_List_{cell_type}.csv')
  
  # write_df_to_sheets(mod_pred_formulations, mod_pred_save_path)
  # write_df_to_sheets(random_high_formulations, random_high_formulations_save_path)
  # write_df_to_sheets(random_low_formulations, random_low_formulations_save_path)


"""**MAIN**"""

def main():

  ################ Retreive/Store Data ##############################################
  datafile_path = "Raw_Data/7_Master_Formulas.csv"
  model_path = 'Trained_Models/Final_Models/'
  save_path = "Predictions/Final_Models/"
  ################ INPUT PARAMETERS ############################################

  wt_percent = False
  if wt_percent == True:
    formulation_param_names = ['wt_Helper', 'wt_Dlin','wt_Chol', 'wt_DMG', 'wt_pDNA']
  else:
    formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                        'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio']
  helper_lipid_names = ['18PG', 'DOPE','DOTAP','DSPC', '14PA', 'DDAB']

  
  lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA',
                       'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
  input_param_names = lipid_param_names + formulation_param_names


##################### Run Predictions ###############################
  #Training Data
  cell_type = 'HepG2'
  model_list = ['RF', 'XGB', 'LGBM']
  df = init_data(datafile_path, cell_type)
  training_data = df.loc[:,df.columns.isin(['Formula label', 'Helper_lipid'] + input_param_names + ['RLU_'+ cell_type])]
  training_data.dropna(inplace = True)
  training_data.reset_index(drop = True, inplace=True)

  #Initialize Scaler
  X = training_data.loc[:, training_data.columns.isin(input_param_names)]                         
  y = training_data['RLU_' + cell_type].to_numpy()
  scaled_X, scaled_y, scaler = apply_minmaxscaling(X, y)
  
  #Create Prediction Array (Change to extracting from Excel file)
  X_predictions, helper_lipids = create_X_prediction_array(df, lipid_param_names, formulation_param_names, wt_percent)
  
  for model_name in model_list:
  #Optimized and Trained ML Model
    with open(model_path +f'{model_name}/{cell_type}/{model_name}_{cell_type}_Trained.pkl', 'rb') as file: # import trained model
      model = pickle.load(file)
    print(model)

    predictions = get_best_predictions(X_predictions, model, scaler, input_param_names, helper_lipids, cell_type, model_name, save_path)



if __name__ == "__main__":
    main()