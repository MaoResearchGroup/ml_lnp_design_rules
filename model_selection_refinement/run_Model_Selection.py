import pickle
import pandas as pd
import os
from model_selection_refinement.Nested_CV import NESTED_CV
import time
from copy import deepcopy

"""
Function that:
- runs the NESTED_CV for a desired cell typeand model panel for a given number of folds
- prints status and progress of NESTED_CV
- formats the results as a datafarme, and saves them locally
- assigns the best HPs to the model, trains, and saves its locally
- returns the results and updates the pipeline dictionary
- prints time required for this step
"""


def run_Model_Selection(pipeline):


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
  model_selection_results = pd.DataFrame(index = model_list, columns = ['Model', 'Hyper_Params', 'AE', 'MAE', 'Predictions'])
  #Iterate through model list and optimize using Nested CV
  for model_name in model_list:

    #Add model to dictionary
    pipeline['Model_Selection']['NESTED_CV'][model_name]= {'HP_tuning_df': None,
                                                            'Final_model': {'Model': None,
                                                                            'Hyper_Params': None,
                                                                            'Test_MAE': None,
                                                                            'Test_spear':None,
                                                                            'Test_pear': None,
                                                                            'Test_pred': None}
                                                        }

    model_instance = NESTED_CV(model_name) #Initialize Model
    model_instance.input_target(X, y, data)
    model_instance.cross_validation(N_CV) #Datasplitting and Hyperparameter Optimization
    model_instance.results() #collect results
    final_AE, final_acc, final_spear, final_pears, pred_df = model_instance.FINAL_TEST_MAE() #Train best selected model archetecture and hyperparameters on all training data and evaluate performance on test set
    model_instance.best_model_refit() #Fit best model to all training data

    # Check if save path exists (if not, then create path)
    if os.path.exists(save_path + f'{model_name}/') == False:
      os.makedirs(save_path + f'{model_name}/', 0o666)
      
    # Save Tuning Results CSV for users
    with open(save_path + f'{model_name}/HP_Tuning_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
      model_instance.CV_dataset.to_csv(f, index = False)
    
    # Save Tuning Results PKL #####OUTDATED
    model_instance.CV_dataset.to_pickle(save_path + f'{model_name}/HP_Tuning_Results.pkl', compression='infer', protocol=5, storage_options=None) 
    
    # Save the Model to pickle file
    with open(save_path + f'{model_name}/Trained_Model.pkl', 'wb') as file: 
          pickle.dump(model_instance.best_model, file)

    #Save model specific results to pipeline object
    pipeline['Model_Selection']['NESTED_CV'][model_name]['HP_tuning_df'] = model_instance.CV_dataset
    pipeline['Model_Selection']['NESTED_CV'][model_name]['Final_model']['Model'] = deepcopy(model_instance.best_model)
    pipeline['Model_Selection']['NESTED_CV'][model_name]['Final_model']['Hyper_Params'] = model_instance.best_model_params
    pipeline['Model_Selection']['NESTED_CV'][model_name]['Final_model']['Test_AE'] = final_AE
    pipeline['Model_Selection']['NESTED_CV'][model_name]['Final_model']['Test_MAE'] = final_acc
    pipeline['Model_Selection']['NESTED_CV'][model_name]['Final_model']['Test_spear'] = final_spear
    pipeline['Model_Selection']['NESTED_CV'][model_name]['Final_model']['Test_pear'] = final_pears
    pipeline['Model_Selection']['NESTED_CV'][model_name]['Final_model']['Test_pred'] = pred_df


    #Create a df for all Model selection results across different models
    model_selection_results.at[model_name, 'Model'] = model_instance.best_model
    model_selection_results.at[model_name, 'Hyper_Params'] =model_instance.best_model_params
    model_selection_results.at[model_name, 'AE'] = final_AE
    model_selection_results.at[model_name, 'MAE'] = final_acc
    model_selection_results.at[model_name, 'Predictions'] = pred_df

  #Saving Training data used
  with open(save_path + f'Training_Data.csv', 'w', encoding = 'utf-8-sig') as file:
        model_instance.cell_data.to_csv(file, index = False)

  #reorder all HP optimized models by MAE and select model with minimal test MAE for all future analysis.
  sorted_results = model_selection_results.sort_values(by='MAE', ascending=True)

  selected_model_name = sorted_results.index[0]
  selected_trained_model = sorted_results['Model'][0]
  selected_trained_model_HP = sorted_results['Hyper_Params'][0]
  selected_model_AE = sorted_results['AE'][0]
  selected_model_acc = sorted_results['MAE'][0]
  selected_model_predictions = sorted_results['Predictions'][0]


  #Save Name of best model to reference in future analysis
  pipeline['Model_Selection']['Best_Model']['Model_Name'] = selected_model_name
  pipeline['Model_Selection']['Best_Model']['Predictions'] = selected_model_predictions
  pipeline['Model_Selection']['Best_Model']['Test_Absolute_Error'] = selected_model_AE
  pipeline['Model_Selection']['Best_Model']['MAE'] = selected_model_acc
  pipeline['Model_Selection']['Best_Model']['Model'] = selected_trained_model
  pipeline['Model_Selection']['Best_Model']['Hyper_Params'] = selected_trained_model_HP
  pipeline['STEPS_COMPLETED']['Model_Selection'] = True
  
  print(f'Selected Model: {selected_model_name} with estimated error = {selected_model_acc}')
  print('Sucessfully save Model Selection Results and Update Pipeline')
  print("\n\n--- %s minutes for MODEL SELECTION---" % ((time.time() - start_time)/60))

  return pipeline, model_instance.CV_dataset, model_instance.best_model, model_instance.best_model_params
