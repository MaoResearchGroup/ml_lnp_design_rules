# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
import time
from utilities import get_mean_shap

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
def main(pipeline):
  print('\n###########################\n\n RUNNING SHAP EXPLANATIONS')
  start_time = time.time()


  #Config
  cell = pipeline['Cell']
  model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
  trained_model = pipeline['Feature_Reduction']['Refined_Model']
  X = pipeline['Feature_Reduction']['Refined_X']
  input_params = pipeline['Feature_Reduction']['Refined_Params']
  shap_save_path = pipeline['Saving']['SHAP']

  X = X[input_params].copy()

  #initialize SHAP Explainer
  if model_name == 'XGB':
    explainer = shap.Explainer(trained_model.predict, X) #XGB
    tree = False
  elif model_name in ['LGBM', 'RF', 'DT']: #for other tree based
    explainer = shap.TreeExplainer(trained_model) 
    tree = True
  else:
    explainer = shap.Explainer(trained_model)
    tree = False 

  #Get SHAP Values
  shap_values = explainer(X)

  #Get Best Feature Values based on Average SHAP Values
  best_feature_values, mean_shap = get_mean_shap(c = cell,
                                                  input_params=input_params,
                                                  shap_values= shap_values,
                                                  N_bins= 20)

  ######   SAVING RESULTS ######
  if os.path.exists(shap_save_path) == False:
    os.makedirs(shap_save_path, 0o666)


  #save SHAP Values
  with open(shap_save_path + f"{model_name}_SHAP_values.pkl",  'wb') as file:
    pickle.dump(shap_values, file)
  shap_values.values.tofile(shap_save_path + f"{model_name}_SHAP_values.csv",sep = ',')
  shap_values.data.tofile(shap_save_path + f"{model_name}_SHAP_data.csv",sep = ',')


  #Save average shap of the binned features as csv
  with open(shap_save_path + f"/{model_name}_{cell}_mean_shap.csv", 'w', encoding = 'utf-8-sig') as f:
      mean_shap.to_csv(f, index = False)
    

  #Save average shap of the features as csv
  with open(shap_save_path + f"/{model_name}_{cell}_best_feature_values.csv", 'w', encoding = 'utf-8-sig') as f:
      best_feature_values.to_csv(f, index = False)

  #Only can get interaction values if treeExplainer
  if tree:
    # Get SHAP Interaction Values
    shap_interaction_values = explainer.shap_interaction_values(X)
    
    #save SHAP Interaction Values
    with open(shap_save_path + f"{model_name}_SHAP_inter_values.pkl",  'wb') as file:
      pickle.dump(shap_interaction_values, file)
    
  else:
     shap_interaction_values = None

  #Update Pipeline                    
  pipeline['SHAP']['X'] = X
  pipeline['SHAP']['Input_Params'] = input_params
  pipeline['SHAP']['SHAP_Values'] = shap_values
  pipeline['SHAP']['SHAP_Interaction_Values'] = shap_interaction_values
  pipeline['SHAP']['Best_Feature_Values'] = best_feature_values
  pipeline['SHAP']['Mean_SHAP_Values'] = mean_shap

  print("\n\n--- %s minutes for SHAP explanation---" % ((time.time() - start_time)/60))

  return pipeline, shap_values, shap_interaction_values, best_feature_values, mean_shap
if __name__ == "__main__":
    main()