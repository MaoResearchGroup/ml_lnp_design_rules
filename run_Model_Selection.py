import pickle
import pandas as pd
import os
from Nested_CV import NESTED_CV
import time
from utilities import get_Model_Selection_Error


def run_Model_Selection(pipeline):

  """
  Function that:
  - runs the NESTED_CV for a desired cell typeand model panel for a given number of folds
  - prints status and progress of NESTED_CV
  - formats the results as a datafarme, and saves them locally
  - assigns the best HPs to the model, trains, and saves its locally
  - returns the results and updates the pipeline dictionary
  - prints time required for this step
  """
  print('\n###########################\n\n MODEL SELECTION AND OPTIMIZATION')
  start_time = time.time()

  #Config
  cell = pipeline['Cell']
  model_list = pipeline['Model_Selection']['Model_list']
  X = pipeline['Data_preprocessing']['X']
  y = pipeline['Data_preprocessing']['y']
  data = pipeline['Data_preprocessing']['all_proc_data']
  N_CV = pipeline['Model_Selection']['N_CV']
  save_path = pipeline['Saving']['Models']


  #Track optimized model results
  model_selection_results = pd.DataFrame(index = model_list, columns = ['Model', 'Hyper_Params'])

  #Iterate through model list and optimize using Nested CV
  for model_name in model_list:

    model_instance = NESTED_CV(model_name) #Initialize Model
    model_instance.input_target(X, y, data)
    model_instance.cross_validation(N_CV) #Hyperparameter Optimization
    model_instance.results() #collect results
    model_instance.overall_MAE(N_CV) #Rerun KFold using best parameters to evaluate performance
    model_instance.final_results() #Best model performance across Kfold
    model_instance.best_model_refit() #Fit best model to all training data


    # Check if save path exists (if not, then create path)
    if os.path.exists(save_path + f'{model_name}/') == False:
      os.makedirs(save_path + f'{model_name}/', 0o666)
      
    # Save Tuning Results CSV
    with open(save_path + f'{model_name}/HP_Tuning_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
      model_instance.CV_dataset.to_csv(f, index = False)
    
    # Save Tuning Results PKL
    model_instance.CV_dataset.to_pickle(save_path + f'{model_name}/HP_Tuning_Results.pkl', compression='infer', protocol=5, storage_options=None) 
    
    # Save the Model to pickle file
    with open(save_path + f'{model_name}/Trained_Model.pkl', 'wb') as file: 
          pickle.dump(model_instance.best_model, file)

    # Save best model Results CSV
    with open(save_path + f'{model_name}/Best_Model_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
      model_instance.best_model_dataset.to_csv(f, index = False)
    
    # Save best model Results PKL
    model_instance.best_model_dataset.to_pickle(save_path + f'{model_name}/Best_Model_Results.pkl', compression='infer', protocol=5, storage_options=None) 

    #update Model selection results
    model_selection_results.at[model_name, 'Model'] = model_instance.best_model
    model_selection_results.at[model_name, 'Hyper_Params'] =model_instance.best_model_params
    print(model_selection_results.at[model_name, 'Model'])

  #Saving Training data used
  with open(save_path + f'Training_Data.csv', 'w', encoding = 'utf-8-sig') as file:
        model_instance.cell_data.to_csv(file, index = False)

  #Update pipeline dictionary with results
  Model_AE, Best_AE = get_Model_Selection_Error(model_list= model_list,
                                      model_path=save_path,
                                      N_CV=N_CV)
  MAE = Model_AE.mean(axis = 0)


  pipeline['Model_Selection']['Results']['Absolute_Error'] = Model_AE
  pipeline['Model_Selection']['Results']['MAE'] = MAE
  pipeline['Model_Selection']['Best_Model']['Model_Name'] = MAE.index[0]
  pipeline['Model_Selection']['Best_Model']['Predictions'] = Best_AE
  pipeline['Model_Selection']['Best_Model']['MAE'] = MAE[0]
  pipeline['Model_Selection']['Best_Model']['Model'] = model_selection_results.at[MAE.index[0], 'Model']
  pipeline['Model_Selection']['Best_Model']['Hyper_Params'] = model_selection_results.at[MAE.index[0],'Hyper_Params']
  
  
  print('Selected Model: ')
  print(model_selection_results.at[MAE.index[0], 'Model'])
  print('Sucessfully save Model Selection Results and Update Pipeline')
  print("\n\n--- %s minutes for MODEL SELECTION---" % ((time.time() - start_time)/60))

  return pipeline, model_instance.CV_dataset, model_instance.best_model, model_instance.best_model_params
