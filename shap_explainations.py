# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap

def init_data(filepath,cell_type_names):
    """ Takes a full file path to a spreadsheet and an array of cell type names. 
    Returns a dataframe with 0s replaced with 1s."""
    df = pd.read_csv(filepath)
    for cell_type in df.columns[-len(cell_type_names):]:
      zero_rows = np.where(df[cell_type] == 0)
      for i in zero_rows:
        df[cell_type][i] = 1
    return df

def get_shap(model, X_train, input_param_names, cell_type, model_name, save_path):
  explainer = shap.Explainer(model, X_train)
  shap_values = explainer(X_train)
  #shap_values = shap.TreeExplainer(model).shap_values(X_train)
  print('Bar Summary Plot')
  #shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names = input_param_names, title = cell_type) #Variable Importance Plot
  #f = plt.figure()
  print('Dot Summary Plot')
  #shap.summary_plot(shap_values, X_train, plot_type = "dot", feature_names = input_param_names) #SHAP Variable Importance Plot
 
  
  col2num = {col: i for i, col in enumerate(X_train.columns)}
  feature_order = list(map(col2num.get, input_param_names))
  shap.plots.beeswarm(shap_values, max_display=12, show=False, color_bar=False, order=feature_order)
  print("Beeswarm Completed")
  plt.colorbar()
  #plt.show()
  plt.savefig(save_path + f'{model_name}_{cell_type}_Summary.png', bbox_inches = 'tight')
  print('SHAP Dependence plot')
  #for params in input_param_names:
    #shap.dependence_plot(params, shap_values, features = X_train, feature_names = input_param_names) # Dependence plot

def extract_training_data(data_path, cell_type_list, input_param_names):
    df = init_data(data_path, cell_type_list)
    training_data_array = []
    for cell in cell_type_list:
       training_data = df.loc[:,df.columns.isin(['Formula label', 'Helper_lipid'] + input_param_names + ['RLU_'+ cell])]
       training_data.dropna(inplace = True)
       training_data.reset_index(drop = True, inplace=True)
       training_data_array.append(training_data.loc[:, training_data.columns.isin(input_param_names)]) #Store only X training data
    return training_data_array
"""**MAIN**"""
def main():

  ################ Retreive Data ##############################################
  model_folder = "Trained_Models/Final_Models/" #Leo
  datafile_path = 'Raw_Data/7_Master_Formulas.csv' 
  plot_save_path = "Figures/SHAP/Final_Models_wt/"
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
  cell_type = ['HEK293', 'HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']
  #cell_type = ['HEK293', 'HepG2', 'N2a']
  model_list = ['LGBM', 'RF']
  #model_list = ['RF', 'MLR', 'lasso', 'PLS', 'SVR', 'kNN', 'LGBM', 'XGB', 'DT']
  
  train_data_list = extract_training_data(datafile_path, cell_type, input_param_names)
  #print(train_data_list)
  #Extracting SHAP Values
  for model_name in model_list:
    print(f'###############{model_name}#############')
    shap_values_list = []
    for c in cell_type:
      print(f'\n################################################################\n\n{c}:')
      model_path = model_folder + f'{model_name}/{c}/{model_name}_{c}_Trained.pkl'
      with open(model_path, 'rb') as file: # import trained model
        trained_model = pickle.load(file)
      if model_name == 'XGB':
        explainer = shap.Explainer(trained_model.predict, train_data_list[cell_type.index(c)]) #for RF, XGB, LGBM
      else:
        explainer = shap.Explainer(trained_model, train_data_list[cell_type.index(c)]) #for RF, XGB, LGBM
      shap_values = explainer(train_data_list[cell_type.index(c)])
      shap_values_list.append(shap_values)
    with open(f"SHAP_Values/Final_Models/{model_name}_SHAP_value_list.pkl",  'wb') as file:
      pickle.dump(shap_values_list, file)

if __name__ == "__main__":
    main()