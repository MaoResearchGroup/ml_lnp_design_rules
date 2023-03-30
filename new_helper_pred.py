# Set up
import pandas as pd
import numpy as np
import sympy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
from itertools import product
from operator import truediv
import seaborn as sns
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

def create_X_prediction_array(training, helper_lipid_path):
  #Extract lipid parameterizations
  lipid_parameterization = pd.read_csv(helper_lipid_path)
  lipid_parameterization.set_index('Helper_lipid', inplace = True)
  print(lipid_parameterization)
  #lipid_parameterization = training.loc[:, training.columns.isin(['Helper_lipid'] + lipid_param_names)].drop_duplicates(keep='first').reset_index(drop=True).set_index('Helper_lipid')

  formulation_params_dic = {'Helper_lipid' : helper_lipid_names,
                            'NP_ratio' : [4,8,12],
                            'Dlin-MC3_Helper lipid_ratio' : [1, 10, 50, 100, 200],
                            'Dlin-MC3+Helper lipid percentage': [20,40,60,80], 
                            'Chol_DMG-PEG_ratio': [10, 100, 500]}
  
  formulation_param_array = pd.DataFrame([row for row in product(*formulation_params_dic.values())], 
                       columns=formulation_params_dic.keys())

  formulation_param_array.drop_duplicates(subset = ['Helper_lipid', 'NP_ratio', 'Dlin-MC3_Helper lipid_ratio', 'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio'], keep=False, inplace = True)
  formulation_param_array.reset_index(drop= True, inplace = True)

  #Add lipid parameterization for each helper lipid to formulation parameter array
  for parameter in lipid_param_names:
    for helper_lipid in helper_lipid_names:
      formulation_param_array.loc[formulation_param_array.Helper_lipid == helper_lipid, parameter] = lipid_parameterization.loc[helper_lipid, parameter]
  print('Number of new formulations: ', len(formulation_param_array.index))
  formulation_param_array = formulation_param_array[['Helper_lipid'] + lipid_param_names + formulation_param_names]#reorder columns
  formulation_param_array.reset_index(inplace = True, names = "Formula label") #Bring down index as formulation labels

  print(formulation_param_array)
  return formulation_param_array, helper_lipid_names #reorder columns

def get_predictions(input_array, best_model, scaler, cell):

  #Run ML on input_array to get untested formulation predictions
  print(input_array[input_param_names])
  y_predictions = best_model.predict(input_array[input_param_names])
  
  #Store Y_predictions by training fold used
  prediction_results = input_array.copy()
  prediction_results[f'{cell}_Prediction'] = y_predictions
  prediction_results[f'{cell}_Prediction_RLU'] = scaler.inverse_transform(y_predictions.reshape(-1, 1))

  return prediction_results


def init_training_data(df, cell):
  training_data = df[['Formula label', 'Helper_lipid'] + input_param_names + ['RLU_'+ cell]]
  training_data = training_data.dropna()
  training_data.reset_index(drop = True, inplace=True)

  X = training_data.loc[:, training_data.columns.isin(input_param_names)]                         
  y = training_data['RLU_' + cell].to_numpy()
  scaled_X, scaled_y, scaler = apply_minmaxscaling(X, y)

  training_data['Norm_Transfection_' + cell] = scaled_y

  return training_data, scaler

def create_pred_plot(save_folder, cell_types, model_name, saving):
  print(f'############ Generate plot for : {model_name} ###############\n')
  #Repeat Plotting for every cell type
  fig, axes = plt.subplots(len(cell_types), 1,figsize=(15, 18))
  fig.suptitle("In Silico Predicted Transfection", fontsize = 18, y =0.92)
  for i, axs in enumerate(axes):
    #Retreive prediction data
    prediction_list = pd.read_csv(save_folder + f"{model_name}_{cell_types[i]}_All_Predictions.csv")
    prediction_list.drop(columns=prediction_list.columns[0], axis=1,  inplace=True) #Remove extra indexing column

    #Plot Predictions
    # sns.stripplot(x = "Helper_lipid", y= f'{cell_types[i]}_Prediction', data = prediction_list, size = 2.5, marker="o", 
    #                         alpha=0.5, linewidth=0.3, ax = axs)
    # sns.violinplot(x = "Helper_lipid", y= f'{cell_types[i]}_Prediction', data = prediction_list, color = "skyblue", alpha = 0.2, ax = axs)
    sns.swarmplot(x = "Helper_lipid", y= f'{cell_types[i]}_Prediction', data = prediction_list, size = 3, 
                             alpha=0.8, linewidth=0.3, ax = axs)
    axs.axhline(y = 0.7, color = 'r', linestyle = 'dashed', alpha = 1, linewidth = 0.8)  
    
    axs.set_ylabel("Fraction of Maximum Observed Transfection")
    axs.set_title(cell_types[i], fontsize = 18)
    axs.set_ylim(0, 0.9)
    #axs.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
  
  plt.savefig(saving, bbox_inches = 'tight')

def create_train_feature_plot(df, cell_type, saving):
  print(f'############ Generate {cell_type} Training Feature Plot ###############\n')
  training_data, scaler = init_training_data(df, cell_type)
  #Repeat Plotting for every formulation feature
  fig, axes = plt.subplots(4, 1,figsize=(12, 24))
  fig.suptitle(cell_type, fontsize = 24, y =0.90)

  labels = ["A", "B", "C", "D"]
  for i, axs in enumerate(axes):
    #sns.set(font_scale=1.2)
    # sns.swarmplot(x = formulation_param_names[i], y= f'RLU_{cell_type}', data = training_data, hue = "Helper_lipid", size = 4, marker="o", 
    #                         alpha=0.8, linewidth=0.3, ax = axs)
    sns.stripplot(x = formulation_param_names[i], y= f'RLU_{cell_type}', data = training_data, hue = "Helper_lipid", size = 5, marker="o", 
                        alpha=0.8, linewidth=0.3, ax = axs)
    sns.violinplot(x = formulation_param_names[i], y= f'RLU_{cell_type}', data = training_data, color = "skyblue", alpha = 0.2, ax = axs)
    #axs.axhline(y = 8, color = 'r', linestyle = 'dashed', alpha = 1, linewidth = 0.8)  
    axs.set_ylabel("Relative Luminescence Units", fontsize=15)
    axs.set_xlabel(formulation_param_names[i], fontsize=15)
    axs.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    axs.set_ylim(-2, 14)
    #Add Figure labels
    axs.text(.02, .96, labels[i], ha='left', va='top', fontsize = 18, transform=axs.transAxes)
    axs.tick_params(axis='both', labelsize=14)
  plt.savefig(saving, bbox_inches = 'tight')



################ Global Variables ##############################################
datafile_path = "Raw_Data/7_Master_Formulas.csv"
helper_lipid_path = "Raw_Data/Helper_Lipid_Param.csv"
model_path = 'Trained_Models/Final_Models/'
save_path = "IP_Predictions/230329/"
wt_percent = False
if wt_percent == True:
  formulation_param_names = ['wt_Helper', 'wt_Dlin','wt_Chol', 'wt_DMG', 'wt_pDNA']
else:
  formulation_param_names = ['NP_ratio','Dlin-MC3_Helper lipid_ratio',
                      'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio']
helper_lipid_names = ['aLin', 'eico', 'docosahex', 'gLin', 'arach', 'docosapent']


lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA',
                      'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
input_param_names = lipid_param_names + formulation_param_names

"""**MAIN**"""
def main():
##################### Run Predictions ###############################
  #Training Data
  cell_type_list = ["HepG2", "PC3", "B16"]
  model_list = ['RF', "XGB"]
  df = init_data(datafile_path, cell_type_list)

  #Created CSVs with all the prediction data per model/per cell type
  for cell_type in cell_type_list:
    training_data, scaler = init_training_data(df, cell_type)
    #Create Prediction Array (Change to extracting from Excel file)
    X_predictions, helper_lipids = create_X_prediction_array(training_data,helper_lipid_path)
    for model_name in model_list:
      print(f'############ Creating formulations for : {model_name} and {cell_type} ###############')
      #Optimized and Trained ML Model
      with open(model_path +f'{model_name}/{cell_type}/{model_name}_{cell_type}_Trained.pkl', 'rb') as file: # import trained model
        model = pickle.load(file)
      #Prediction Transfection for Formulations in Formulation Array
      predictions = get_predictions(X_predictions, model, scaler, cell_type)
      write_df_to_sheets(predictions, save_path + f"{model_name}_{cell_type}_All_Predictions.csv")
      print(f"saved all {model} predictions for {cell_type}")
  
  #Plotting
  for model_name in model_list: 
    create_pred_plot(save_path, cell_type_list,
                      model_name, save_path + f'{model_name}_new_helper_Predictions_Strip.png') #Plotting Prediction Data
       
if __name__ == "__main__":
    main()