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


  X = pipeline['Data_preprocessing']['X']
  y = pipeline['Data_preprocessing']['y']
  
  
  shap_save_path = pipeline['Saving']['SHAP']

  #check if feature reduction was conducted
  if refined:
    if pipeline['STEPS_COMPLETED']['Feature_Reduction']:
      input_params = pipeline['Feature_Reduction']['Refined_Params']
      input_param_type = 'refined'
  else:
    input_params = pipeline['Data_preprocessing']['Input_Params']
    refined = False
    input_param_type = 'original'
  
  ### TRAINING DATA
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

  new_save = f'{shap_save_path}{input_param_type}/'
  if os.path.exists(new_save) == False:
    os.makedirs(new_save, 0o666)


  #save SHAP Values
  with open(new_save + f"{model_name}_SHAP_values.pkl",  'wb') as file:
    pickle.dump(shap_values, file)
  shap_values.values.tofile(new_save + f"{model_name}_SHAP_values.csv",sep = ',')
  shap_values.data.tofile(new_save + f"{model_name}_SHAP_data.csv",sep = ',')


  #Save average shap of the binned features as csv
  with open(new_save + f"/{model_name}_{cell}_mean_shap.csv", 'w', encoding = 'utf-8-sig') as f:
      mean_shap.to_csv(f, index = False)
    

  #Save average shap of the features as csv
  with open(new_save + f"/{model_name}_{cell}_best_feature_values.csv", 'w', encoding = 'utf-8-sig') as f:
      best_feature_values.to_csv(f, index = False)


  #Update Pipeline
    
  outer_key = input_param_type
  inner_key_list = ['X', 'y', 'Input_Params','Explainer','SHAP_Values', 'SHAP_Interaction_Values','Best_Feature_Values', 'Norm_Best_Feature_Values', 'N_bins','Mean_SHAP_Values','TSNE_Embedding']
  for new_inner_key in inner_key_list:
      if outer_key in pipeline:
          pipeline['SHAP'][outer_key][new_inner_key] = None
      else:
          pipeline['SHAP'][outer_key] = {new_inner_key: None}


  pipeline['SHAP'][input_param_type]['X'] = X
  pipeline['SHAP'][input_param_type]['y'] = pipeline['Data_preprocessing']['y']
  pipeline['SHAP'][input_param_type]['Input_Params'] = input_params
  pipeline['SHAP'][input_param_type]['Explainer'] = explainer
  pipeline['SHAP'][input_param_type]['SHAP_Values'] = shap_values
  pipeline['SHAP'][input_param_type]['SHAP_Interaction_Values'] = shap_interaction_values
  pipeline['SHAP'][input_param_type]['Best_Feature_Values'] = best_feature_values
  pipeline['SHAP'][input_param_type]['Norm_Best_Feature_Values'] = best_feature_values
  pipeline['SHAP'][input_param_type]['N_bins'] = N_bins
  pipeline['SHAP'][input_param_type]['Mean_SHAP_Values'] = mean_shap
  pipeline['SHAP'][input_param_type]['TSNE_Embedding'] = projections
  pipeline['STEPS_COMPLETED']['SHAP'] = True

  print("\n\n--- %s minutes for SHAP explanation---" % ((time.time() - start_time)/60))

  return pipeline, shap_values, shap_interaction_values, best_feature_values, mean_shap
if __name__ == "__main__":
    main()