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
from sklearn.manifold import TSNE
from copy import deepcopy


"""**MAIN**"""
def main(pipeline, N_bins, refined = True):
  print('\n###########################\n\n RUNNING SHAP EXPLANATIONS')
  start_time = time.time()


  #Config
  cell = pipeline['Cell']
  model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
  trained_model = deepcopy(pipeline['Model_Selection']['Best_Model']['Model'])
  X = pipeline['Feature_Reduction']['Refined_X']
  y = pipeline['Data_preprocessing']['y']
  shap_save_path = pipeline['Saving']['SHAP']

  if refined:
    input_params = pipeline['Feature_Reduction']['Refined_Params']
  else:
    input_params = pipeline['Data_preprocessing']['Input_Params']
    
  print(input_params)
  X = X[input_params].copy()
  trained_model.fit(X,np.ravel(y))

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
                                                  N_bins= N_bins)
  #Only can get interaction values if treeExplainer
  if tree:
    # Get SHAP Interaction Values
    shap_interaction_values = explainer.shap_interaction_values(X)
    
    #save SHAP Interaction Values
    with open(shap_save_path + f"{model_name}_SHAP_inter_values.pkl",  'wb') as file:
      pickle.dump(shap_interaction_values, file)
    
  else:
     shap_interaction_values = None

  
  #Embed results using TSNE
  projections = TSNE(n_components=2, perplexity=50, random_state = 0).fit_transform(shap_values.values)
 
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


  #Update Pipeline                    
  pipeline['SHAP']['X'] = X
  pipeline['SHAP']['y'] = pipeline['Data_preprocessing']['y']
  pipeline['SHAP']['Input_Params'] = input_params
  pipeline['SHAP']['Explainer'] = explainer
  pipeline['SHAP']['SHAP_Values'] = shap_values
  pipeline['SHAP']['SHAP_Interaction_Values'] = shap_interaction_values
  pipeline['SHAP']['Best_Feature_Values'] = best_feature_values
  pipeline['SHAP']['Norm_Best_Feature_Values'] = best_feature_values
  pipeline['SHAP']['N_bins'] = N_bins
  pipeline['SHAP']['Mean_SHAP_Values'] = mean_shap
  pipeline['SHAP']['TSNE_Embedding'] = projections
  pipeline['STEPS_COMPLETED']['SHAP'] = True

  print("\n\n--- %s minutes for SHAP explanation---" % ((time.time() - start_time)/60))

  return pipeline, shap_values, shap_interaction_values, best_feature_values, mean_shap
if __name__ == "__main__":
    main()