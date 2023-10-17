# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

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



"""**MAIN**"""
def main():
  ################ Retreive Data ##############################################
  model_folder = "Feature_Reduction/Feature_reduction_NoSizeZeta/" 
  shap_save_path = 'SHAP_Values/Refined_Models_NoSizeZeta/Helper_Lipid/'
  data_path = "Raw_Data/10_Master_Formulas.csv"
  
  size_zeta = False
  PDI_cutoff = 1
##################### Run Predictions ###############################
  #Training Data
  cell_type = ['ARPE19','N2a','PC3','B16','HEK293','HepG2']
  model_list = ['LGBM', 'XGB','RF'] #Did not include SVR

  cell_type = ['B16']
  model_list = ['LGBM'] #Did not include SVR
  #model_list = ['RF', 'MLR', 'lasso', 'PLS', 'SVR', 'kNN', 'LGBM', 'XGB', 'DT']
  
  helper_list = ['DOTAP', 'DDAB', '14PA', '18PG', 'DOPE', 'DSPC']

  #Extracting SHAP Values
  for model_name in model_list:
    shap_values_list = []
    shap_inter_list = []
    for c in cell_type:
      with open(model_folder + f"{c}/{model_name}_{c}_Best_Model_Results.pkl", 'rb') as file: # import trained model
                best_results = pickle.load(file)
      input_param_names = best_results.loc['Feature names'][0]
      
      for lipid in helper_list:

        train_data = pd.read_csv(data_path)

        if size_zeta == True:
          cell_data = cell_data[cell_data.Size != 0] #Remove any rows where size = 0
          cell_data = cell_data[cell_data.Zeta != 0] #Remove any rows where zeta = 0
          cell_data = cell_data[cell_data.PDI <= PDI_cutoff] #Remove any rows where PDI > 0.45       

        #open from pkl file
        # with open(model_folder + f'/{c}/{model_name}_{c}_Best_Training_Data.pkl', "rb") as file:   # Unpickling
        #   train_data = pickle.load(file)
        X =  train_data.loc[train_data["Helper_lipid"] == lipid, input_param_names]


        print(f'\n################################################################\n\n{c} {model_name}:')
        with open(model_folder + f"{c}/{model_name}_{c}_Best_Model.pkl", 'rb') as file: # import trained model
          trained_model = pickle.load(file)
        if model_name == 'XGB':
          #explainer = shap.Explainer(trained_model.predict, train_data_list[cell_type.index(c)]) #XGB
          explainer = shap.Explainer(trained_model.predict, X) #XGB
          #explainer = shap.TreeExplainer(trained_model) #XGB
        else:
          #explainer = shap.Explainer(trained_model, train_data_list[cell_type.index(c)]) #for RF, LGBM
          explainer = shap.Explainer(trained_model) #for RF, LGBM
        #shap_values = explainer(train_data_list[cell_type.index(c)])

        #Get SHAP Values
        shap_values = explainer(X)
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

    # #save SHAP Interaction Values
    # with open(shap_save_path + f"{model_name}_SHAP_inter_value_list.pkl",  'wb') as file:
    #   pickle.dump(shap_inter_list, file)


if __name__ == "__main__":
    main()