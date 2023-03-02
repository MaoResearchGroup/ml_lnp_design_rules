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
def init_data(filepath, cell_type_names):
     """ Takes a full file path to a spreadsheet and an array of cell type names. 
     Returns a dataframe with 0s replaced with 1s."""
     df = pd.read_csv(filepath)
     for cell_type in cell_type_names:
        df["RLU_" + cell_type].replace(0, 1, inplace= True) #Replace 0 transfection with 1
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

def create_X_prediction_array(training):
  helper_lipid_names = training['Helper_lipid'].drop_duplicates().to_numpy()
  #Extract lipid parameterizations from inital dataset
  lipid_parameterization = training.loc[:, training.columns.isin(['Helper_lipid'] + lipid_param_names)].drop_duplicates(keep='first').reset_index(drop= True).set_index('Helper_lipid')
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
  if wt_percent == True:
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
  formulation_param_array = formulation_param_array[['Helper_lipid'] + lipid_param_names + formulation_param_names]#reorder columns
  formulation_param_array.reset_index(inplace = True, names = "Formula label") #Bring down index as formulation labels
  return formulation_param_array, helper_lipid_names #reorder columns

def get_predictions(input_array, best_model, scaler, cell):

  #Run ML on input_array to get untested formulation predictions
  y_predictions = best_model.predict(input_array[input_param_names])
  
  #Store Y_predictions by training fold used
  prediction_results = input_array.copy()
  prediction_results[f'{cell}_Prediction'] = y_predictions
  prediction_results[f'{cell}_Prediction_RLU'] = scaler.inverse_transform(y_predictions.reshape(-1, 1))

  return prediction_results

def generate_iv_validation_list(predictions, helper_lipid_names, cell, name, save_path):
   #Extract High/Mid/Low Transfection Predictions
  low_bar = predictions[f'{cell}_Prediction'].quantile(0.2)
  high_bar = predictions[f'{cell}_Prediction'].quantile(0.8)
  print("High Trasfection Min:", high_bar)
  print("Low Trasfection Max:", low_bar)
  
  
  high_formulations = predictions.loc[predictions[f'{cell}_Prediction'] >= high_bar]
  #mod_formulations = predictions.loc[(predictions[f'{cell}_Prediction'] < high_bar) & (predictions[f'{cell}_Prediction'] > low_bar)]
  low_formulations = predictions.loc[predictions[f'{cell}_Prediction'] <= low_bar]
  print('High Helper Lipids', high_formulations['Helper_lipid'].unique())
  print('Number of high Formulations:', len(high_formulations.index))

  print('Helper Lipids', low_formulations['Helper_lipid'].unique())
  print('Number of low Formulations:', len(low_formulations.index))
  
  #Extract Random Formulations to Test
  num_formulations = 46
  formulation_list = pd.concat([high_formulations.sample(n = num_formulations), low_formulations.sample(n = num_formulations)])


  #Redo randomized search until conditions are met
  min_lipid = 12
  min_Dlin_helper_ratio = 5
  min_Dlin_helper_percent = 2
  min_Chol_PEG = 5
  min_NP_ratio = 5
  iter = 0
  max_iter = 100000
  while not(all(len(formulation_list[formulation_list['Helper_lipid'] == lipid]) >= min_lipid for lipid in helper_lipid_names) and 
            all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'Dlin-MC3_Helper lipid_ratio'].unique().size >= min_Dlin_helper_ratio for lipid in helper_lipid_names) and
            all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'Chol_DMG-PEG_ratio'].unique().size >= min_Chol_PEG for lipid in helper_lipid_names) and
            all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'Dlin-MC3+Helper lipid percentage'].unique().size >= min_Dlin_helper_percent for lipid in helper_lipid_names) and
            all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'NP_ratio'].unique().size >= min_NP_ratio for lipid in helper_lipid_names)): 

        
    formulation_list = pd.concat([high_formulations.sample(n = num_formulations), low_formulations.sample(n = num_formulations)]) #Take new samples from high and low lists
    iter += 1
    if iter > max_iter:
      print(f'Could not find any samples that met condition after {max_iter} combinations')
      formulation_list = pd.DataFrame() #Return empty dataframe
      return formulation_list, high_formulations, low_formulations
      
  formulation_list.sort_values(by = ['Helper_lipid'], inplace = True)
  print('Final Iter #: ', iter)  
  print('Total Unique Helper Lipids:', formulation_list['Helper_lipid'].unique().size, '/', predictions['Helper_lipid'].unique().size)
  print('Total Unique Dlin:helper :', formulation_list['Dlin-MC3_Helper lipid_ratio'].unique().size, '/', predictions['Dlin-MC3_Helper lipid_ratio'].unique().size)
  print('Total Unique Dlin + helper :', formulation_list['Dlin-MC3+Helper lipid percentage'].unique().size, '/', predictions['Dlin-MC3+Helper lipid percentage'].unique().size)
  print('Total Unique Chol:DMG :', formulation_list['Chol_DMG-PEG_ratio'].unique().size, '/', predictions['Chol_DMG-PEG_ratio'].unique().size)
  print('Total Unique NP :', formulation_list['NP_ratio'].unique().size, '/', predictions['NP_ratio'].unique().size)


  return formulation_list, high_formulations, low_formulations

def init_training_data(df, cell):
  training_data = df.loc[:,df.columns.isin(['Formula label', 'Helper_lipid'] + input_param_names + ['RLU_'+ cell])]
  formatted_training_data = training_data.dropna()
  formatted_training_data.reset_index(drop = True, inplace=True)
  training_data = formatted_training_data

  X = training_data.loc[:, training_data.columns.isin(input_param_names)]                         
  y = training_data['RLU_' + cell].to_numpy()
  scaled_X, scaled_y, scaler = apply_minmaxscaling(X, y)

  return training_data, scaler

################ Global Variables ##############################################
datafile_path = "Raw_Data/7_Master_Formulas.csv"
model_path = 'Trained_Models/Final_Models/'
save_path = "Predictions/test/"
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

"""**MAIN**"""
def main():
##################### Run Predictions ###############################
  #Training Data
  cell_type_list = ["HEK293", "HepG2", "B16"]
  model_list = ['RF']
  df = init_data(datafile_path, cell_type_list)

  #Created CSVs with all the prediction data per model/per cell type
  for cell_type in cell_type_list:
    training_data, scaler = init_training_data(df, cell_type)
    #Create Prediction Array (Change to extracting from Excel file)
    X_predictions, helper_lipids = create_X_prediction_array(training_data)
    for model_name in model_list:
      print(f'############ Creating formulations for : {model_name} and {cell_type} ###############')
      #Optimized and Trained ML Model
      with open(model_path +f'{model_name}/{cell_type}/{model_name}_{cell_type}_Trained.pkl', 'rb') as file: # import trained model
        model = pickle.load(file)
      #Prediction Transfection for Formulations in Formulation Array
      predictions = get_predictions(X_predictions, model, scaler, cell_type)
      write_df_to_sheets(predictions, save_path + f"{model_name}_{cell_type}_All_Predictions.csv")
      print(f"saved all {model} predictions for {cell_type}")

      
  for cell_type in cell_type_list:
    for model_name in model_list:    
      print(f'############ Formulation List for : {model_name} and {cell_type} ###############\n')
      #Generate In Vitro Validation Formulation List based on the predictions of a single cell
      prediction_list = pd.read_csv(save_path + f"{model_name}_{cell_type}_All_Predictions.csv")
      prediction_list.drop(columns=prediction_list.columns[0], axis=1,  inplace=True) #Remove extra indexing column

      #Append other cell prediction results to the prediction list
      add_cell_list = cell_type_list.copy()
      add_cell_list.remove(cell_type)
      add_cell_list
      if len(add_cell_list) > 0:
        for add_cell in add_cell_list:
          add_predictions = pd.read_csv(save_path + f"{model_name}_{add_cell}_All_Predictions.csv")
          prediction_list[f'{add_cell}_Prediction'] = add_predictions[f'{add_cell}_Prediction']
          prediction_list[f'{add_cell}_Prediction_RLU'] = add_predictions[f'{add_cell}_Prediction_RLU']
          print(f"{add_cell} Predictions added to prediction list")

      
      #Saving Formulation Lists
      #formulation_list, high_formulations, low_formulations = generate_iv_validation_list(prediction_list, helper_lipids, cell_type, model_name, save_path)
      # write_df_to_sheets(formulation_list, save_path + f"{model_name}_{cell_type}_Formulation_List.csv")
      # print("saved in vitro formulation list")    
      # write_df_to_sheets(high_formulations, save_path + f"{model_name}_{cell_type}_High_Predictions.csv")
      # print("saved high predictions")
      # write_df_to_sheets(low_formulations, save_path + f"{model_name}_{cell_type}_Low_Predictions.csv")
      # print("saved low predictions")
      # print("#######################\n")



if __name__ == "__main__":
    main()