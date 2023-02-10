# -*- coding: utf-8 -*-
# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
from Nested_CV import NESTED_CV



def write_df_to_sheets(data_df, save_path):
    
  with open(save_path, 'w', encoding = 'utf-8-sig') as f:
    data_df.to_csv(f)

    return


def run_NESTED_CV(model_name, data_file_path, save_path, cell, wt_percent, CV):

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
    model_instance = NESTED_CV(data_file_path, model_name)
    model_instance.input_target(cell_type = cell, wt_percent = wt_percent)
    model_instance.cross_validation(CV)
    model_instance.results()
    model_instance.best_model() 
    ###
    with open(save_path + f'{model_name}/{cell}/{model_name}_HP_Tuning_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
      model_instance.CV_dataset.to_csv(f)
    
    
    #### Save model parameters and model as pickle file in Google Colab
    model_instance.CV_dataset.to_pickle(save_path + f'{model_name}/{cell}/{model_name}_HP_Tuning_Results.pkl', compression='infer', protocol=5, storage_options=None) # save dataframe as pickle file
    with open(save_path + f'{model_name}/{cell}/{model_name}_{cell}_Trained.pkl', 'wb') as file: # Save the Model to pickle file
          pickle.dump(model_instance.best_model, file)
    print('Sucessfully save NESTED_CV Results and Final Model')
    return model_instance.CV_dataset, model_instance.best_model

"""**Visualization**"""

def plot_predictions(prediction_data, cell_type_names, random_states, n_folds):
  prediction_data = prediction_data.set_index(['Cell_Type','Random_State', 'Fold']) #reformatting
  for c in cell_type_names:
    for state in random_states:
      fig, axs = plt.subplots(1, n_folds, sharex=True, sharey=True, figsize=(10,4))
      for ax_ind, ax in enumerate(axs):
        predicted_values = prediction_data.loc[c, state, ax_ind]['Predicted_Values']
        actual_values = prediction_data.loc[c, state, ax_ind]['Actual_Values']
        ax.scatter(predicted_values, actual_values)
        ax.set_title(c + f'(Fold-{ax_ind})')
        if ax_ind == 0:
          ax.set_ylabel('Actual Value')
        elif ax_ind == 2:
          ax.set_xlabel('Predicted Value')
    
      ax.legend(frameon=False, handlelength=0)
      plt.tight_layout()
      plt.show()

def get_shap(model, X_train, input_param_names, cell_type):
  shap_values = shap.TreeExplainer(model).shap_values(X_train)
  print('Bar Summary Plot')
  shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names = input_param_names, title = cell_type) #Variable Importance Plot
  f = plt.figure()
  print('Dot Summary Plot')
  shap.summary_plot(shap_values, X_train, plot_type = "dot", feature_names = input_param_names) #SHAP Variable Importance Plot
  print('SHAP Dependence plot')
  #for params in input_param_names:
    #shap.dependence_plot(params, shap_values, features = X_train, feature_names = input_param_names) # Dependence plot

def main():
  ############### Type of ML Algorithm Used ####################################
  #model_type = 'RF' 
  #model_type = 'MLR'
  #model_type = 'lasso'
  #model_type = 'PLS'
  #model_type = 'SVR'
  #model_type = 'kNN'
  model_list = ['LGBM', 'XGB','RF', 'MLR', 'lasso', 'PLS', 'kNN', 'DT'] #Did not include SVR

  ################ INPUT PARAMETERS ############################################
  #file_path = "/content/drive/MyDrive/4_Master_Formulas_1080_lipidparam_csv.csv" #Ataes
  data_file_path = 'Raw_Data/7_Master_Formulas.csv' #leo
  save_path = "Trained_Models/Final_Models_wt/" # Add any descriptors such as date ran....


  wt_percent = True
  if wt_percent == True:
    formulation_param_names = ['wt_Helper', 'wt_Dlin','wt_Chol', 'wt_DMG', 'wt_pDNA']
  else:
    formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                        'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio']
  helper_lipid_names = ['18PG', 'DOPE','DOTAP','DSPC', '14PA', 'DDAB']
  cell_type_names = ['HepG2','HEK293','N2a', 'ARPE19', 'B16', 'PC3']

  
  lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA',
                       'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
  input_param_names = lipid_param_names + formulation_param_names

  ##################### Screen and Optimize Model #####################################
  for model_type in model_list:
    print("Algorithm used:", model_type)
    for c in cell_type_names:
      print('Cell Type:', c)
      run_NESTED_CV(model_type, data_file_path, save_path, c, wt_percent, CV = 5)

if __name__ == "__main__":
    main()

'''
Implement Gridsearch for hyperparameter tuning; this could be improved by focusing on a single hyperparameter and demonstrating the impact on accuracy based on chnaging this
'''

'''
See if areas of high transfection efficiency can be identified based on the attributes; another way of phrasing this problem can be of one based on unsupervised clustering; can try Bayesian optimization as well
'''