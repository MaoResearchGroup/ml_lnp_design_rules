from utilities import extract_training_data, init_pipeline, save_pipeline
from run_Model_Selection import run_Model_Selection
import Feature_reduction
import get_shap_explainations
import pickle
import os



def main():
  ################ What parts of the pipeline to run ###############
  #Make New pipeline
  new_pipeline = False
  
  #Parts to Run/Update
  run_preprocessing     = True
  run_model_selection   = True
  run_feature_reduction = True
  run_SHAP_explain      = True
  
  #Cell types to Run
  cell_type_list = ['HepG2','HEK293', 'N2a', 'ARPE19','B16', 'PC3']

  
  ############### PARAMETERS ###############################
  
  model_list = ['RF','LGBM', 'XGB', 'DT', 'MLR', 'lasso', 'PLS', 'kNN']#Did not include SVR 
  ratiometric = True
  RLU_floor = 2
  size_cutoff = 100000
  PDI_cutoff = 1 #Use 1 to include all data

  N_CV = 5
  prefix = "RLU_" #WARNING: HARDCODED

  ################ SAVING, LOADING##########################
  RUN_NAME                  = f"Runs/Final_PDI{PDI_cutoff}_RLU{RLU_floor}/"
  #RUN_NAME                  = f"Runs/Models_Final_All_Size_PDI{PDI_cutoff}_keep_Zeta_RLU{RLU_floor}"
  data_file_path            = 'Raw_Data/Final_Master_Formulas_updated.csv' #Where to extract training data


  ################ Model Training and Analysis ##########################
  #Loop through Model Training and Analysis for each cell type of interest
  for c in cell_type_list:

    pipeline_path  = f'{RUN_NAME}{c}/Pipeline_dict.pkl'
    
    #Load previous pipeline if exists
    if os.path.exists(pipeline_path) and new_pipeline == False:
      print(f'\n\n########## LOADING PREVIOUS PIPELINE FOR {c} ##############\n\n')
      with open(pipeline_path, 'rb') as file:
        pipeline_dict = pickle.load(file)
    
    #Initialize new pipeline if wanted
    else: 
      pipeline_dict = init_pipeline(pipeline_path = pipeline_path,
                                    RUN_NAME=RUN_NAME,
                                    cell = c,
                                    model_list=model_list,
                                    ratiometric=ratiometric,
                                    data_file_path=data_file_path,
                                    size_cutoff=size_cutoff,
                                    PDI_cutoff=PDI_cutoff,
                                    prefix=prefix,
                                    RLU_floor=RLU_floor,
                                    N_CV=N_CV)
      #Run Whole Pipeline
      run_preprocessing     = True
      run_model_selection   = True
      run_feature_reduction = True
      run_SHAP_explain      = True

    ##################### Extract Training Data ###############################
    if run_preprocessing:
      pipeline_dict, _, _, _= extract_training_data(pipeline_dict)

      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'DATA PREPROCESSING')
      

    ##################### Model Selection #####################################
    if run_model_selection:
      #Timing (Estimate 30min)
      pipeline_dict, _, _, _ = run_Model_Selection(pipeline_dict)

      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'MODEL SELECTION')
      

    #################### Feature Reduction #####################################
    if run_feature_reduction:
      #Timing (Estimated 4-8hrs)
      pipeline_dict, _,_,_,_,_ = Feature_reduction.main(pipeline=pipeline_dict)

      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'FEATURE REDUCTION')

    #################### SHAP Analysis #####################################
    if run_SHAP_explain:
      #Timing (Estimated 3 min)
      pipeline_dict, _,_,_,_ = get_shap_explainations.main(pipeline_dict, 10)
      
      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'SHAP')                      

    #################### Saving Pipeline Config and Results #####################################
    save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'FINAL SAVE')  

if __name__ == "__main__":
    main()