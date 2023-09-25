import pickle
import os
#from Nested_CV import NESTED_CV
from Nested_CV_reformat import NESTED_CV_reformat

def run_NESTED_CV(model_name, cell, data_file_path, save_path, input_params, size_cutoff, PDI_cutoff, prefix, N_CV):

  """
  Function that:
  - runs the NESTED_CV for a desired model in the class, cell type, and for a given number of folds
  - default is 10-folds i.e., CV = None. CV = # Trials... # outerloop repeats
  - prints status and progress of NESTED_CV
  - formats the results as a datafarme, and saves them locally
  - assigns the best HPs to the model, trains, and saves its locally
  - then returns the results dataframe and the saved model
  """
  #model_instance = NESTED_CV(data_file_path, model_name)
  model_instance = NESTED_CV_reformat(model_name)
  model_instance.input_target(data_file_path = data_file_path, 
                              cell_type = cell, 
                              input_param_names= input_params,
                              prefix = prefix,
                              size_cutoff = size_cutoff, 
                              PDI_cutoff = PDI_cutoff)
  model_instance.cross_validation(N_CV)
  model_instance.results()
  model_instance.best_model() 


  # Check if save path exists (if not, then create path)
  if os.path.exists(save_path + f'{model_name}/{cell}/') == False:
    os.makedirs(save_path + f'{model_name}/{cell}/', 0o666)
    
  # Save Tuning Results CSV
  with open(save_path + f'{model_name}/{cell}/{model_name}_{cell}_HP_Tuning_Results.csv', 'w', encoding = 'utf-8-sig') as f: #Save file to csv
    model_instance.CV_dataset.to_csv(f)
  
  # Save Tuning Results PKL
  model_instance.CV_dataset.to_pickle(save_path + f'{model_name}/{cell}/{model_name}_{cell}_HP_Tuning_Results.pkl', compression='infer', protocol=5, storage_options=None) 
  
  # Save the Model to pickle file
  with open(save_path + f'{model_name}/{cell}/{model_name}_{cell}_Trained.pkl', 'wb') as file: 
        pickle.dump(model_instance.best_model, file)

  # Save the Training Data used to .pkl
  with open(save_path + f'{model_name}/{cell}/{cell}_Training_Data.pkl', 'wb') as file:
        pickle.dump(model_instance.cell_data, file)

  # Save the Training Data used to csv
  with open(save_path + f'{model_name}/{cell}/{cell}_Training_Data.csv', 'w', encoding = 'utf-8-sig') as file:
        model_instance.cell_data.to_csv(file)
  
  print('Sucessfully save NESTED_CV Results, Final Model, and Training dataset')
  return model_instance.CV_dataset, model_instance.best_model, model_instance.cell_data 
