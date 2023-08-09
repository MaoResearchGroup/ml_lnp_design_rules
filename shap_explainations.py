# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import seaborn as sns
import os


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

################ Retreive Data ##############################################
model_folder = "Trained_Models/Models_Size_Zeta_PDI/" 
shap_save_path = 'SHAP_Values/Models_Size_Zeta_PDI/'
wt_percent = False
size_zeta = True

################ INPUT PARAMETERS ############################################
if wt_percent == True:
  formulation_param_names = ['wt_Helper', 'wt_Dlin','wt_Chol', 'wt_DMG', 'wt_pDNA']
else:
  formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                      'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio'] 

lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA',
                      'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
#lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP','Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds', 'Helper_MW']

if size_zeta == True:
  input_param_names = lipid_param_names +formulation_param_names +  ['Size', 'Zeta', 'PDI']
else:
  input_param_names = lipid_param_names+ formulation_param_names 


"""**MAIN**"""
def main():
##################### Run Predictions ###############################
  #Training Data
  cell_type = ['HepG2','HEK293','N2a', 'ARPE19', 'B16', 'PC3']
  model_list = ['LGBM', 'XGB','RF'] #Did not include SVR
  #model_list = ['RF', 'MLR', 'lasso', 'PLS', 'SVR', 'kNN', 'LGBM', 'XGB', 'DT']
  

  #Extracting SHAP Values
  print(input_param_names)
  for model_name in model_list:
    shap_values_list = []
    shap_inter_list = []
    for c in cell_type:
      with open(model_folder + f'{model_name}/{c}/{c}_Training_Data.pkl', "rb") as file:   # Unpickling
        train_data = pickle.load(file)
      train_data =  train_data[input_param_names]
      #print(train_data)
      print(f'\n################################################################\n\n{c} {model_name}:')
      model_path = model_folder + f'{model_name}/{c}/{model_name}_{c}_Trained.pkl'
      with open(model_path, 'rb') as file: # import trained model
        trained_model = pickle.load(file)
      if model_name == 'XGB':
        #explainer = shap.Explainer(trained_model.predict, train_data_list[cell_type.index(c)]) #XGB
        explainer = shap.Explainer(trained_model.predict, train_data) #XGB
        #explainer = shap.TreeExplainer(trained_model) #XGB
      else:
        #explainer = shap.Explainer(trained_model, train_data_list[cell_type.index(c)]) #for RF, LGBM
        explainer = shap.Explainer(trained_model) #for RF, LGBM
      #shap_values = explainer(train_data_list[cell_type.index(c)])

      #Get SHAP Values
      shap_values = explainer(train_data)
      shap_values_list.append(shap_values)

      # #Get SHAP Interaction Values
      # shap_inter_values = explainer.shap_interaction_values(train_data)
      # shap_inter_list.append(shap_inter_values)
      # #print(shap_inter_values)

      # #heat map
      # # Get absolute mean of matrices
      # mean_shap = np.abs(shap_inter_values).mean(0)
      # df_inter = pd.DataFrame(mean_shap,index=input_param_names,columns=input_param_names)

      # # times off diagonal by 2
      # df_inter.where(df_inter.values == np.diagonal(df_inter),df_inter.values*2,inplace=True)

      # # display 
      # plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
      # sns.set(font_scale=1.5)
      # sns.heatmap(df_inter,cmap='coolwarm',annot=True,fmt='.3g',cbar=False)
      # plt.yticks(rotation=0) 
      # #plt.show()

    #save SHAP Values
    if os.path.exists(shap_save_path) == False:
       os.makedirs(shap_save_path, 0o666)

    with open(shap_save_path + f"{model_name}_SHAP_value_list.pkl",  'wb') as file:
      pickle.dump(shap_values_list, file)

    #save SHAP Interaction Values
    with open(shap_save_path + f"{model_name}_SHAP_inter_value_list.pkl",  'wb') as file:
      pickle.dump(shap_inter_list, file)


if __name__ == "__main__":
    main()