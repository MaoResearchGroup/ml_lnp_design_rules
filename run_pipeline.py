import matplotlib.pyplot as plt
import pandas as pd
from utilities import extract_training_data, init_pipeline
from run_Model_Selection import run_Model_Selection
import Model_Selection_Results
import plot_feature_distribution
import Feature_reduction
import learning_curve_plot
import get_shap_explainations
import refined_model_shap_plots
import SHAP_radar_plot
import bump_plot
import pickle
import os



def main():
  ################ What parts of the pipeline to run ###############

  #Make New pipeline
  new_pipeline = False
  
  #Model Training, Optimization, and analysis
  run_preprocessing     = True
  run_model_selection   = True
  run_feature_reduction = True
  run_SHAP_explain      = True
  

  ############### PARAMETERS ###############################

  model_list = ['RF', 'MLR', 'lasso', 'PLS', 'kNN', 'LGBM', 'XGB', 'DT']#Did not include SVR 
  cell_type_list = ['HepG2'] #,'HEK293', 'N2a', 'ARPE19','B16', 'PC3']
  ratiometric = True
  RLU_floor = 2
  size_cutoff = 100000
  PDI_cutoff = 1 #Use 1 to include all data

  N_CV = 2
  prefix = "RLU_" #WARNING: HARDCODED

  ################ SAVING, LOADING##########################
  RUN_NAME                  = f"Runs/Test_PDI{PDI_cutoff}_keep_Zeta_RLU{RLU_floor}/"
  #RUN_NAME                  = f"Runs/Models_Final_All_Size_PDI{PDI_cutoff}_keep_Zeta_RLU{RLU_floor}"
  data_file_path            = 'Raw_Data/Final_Master_Formulas.csv' #Where to extract training data


  ################ Model Training and Analysis ##########################
  #Loop through Model Training and Analysis for each cell type of interest
  for c in cell_type_list:

    pipeline_path  = f'{RUN_NAME}{c}/Pipeline_dict.pkl'
    
    #Load previous pipeline if exists
    if os.path.exists(pipeline_path) and new_pipeline == False:
      print('\n\n########## LOADING PREVIOUS PIPELINE ##############\n\n')
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
      
    ##################### Extract Training Data ###############################
    if run_preprocessing:
      pipeline_dict, _, _, _= extract_training_data(pipeline_dict)
      
      pipeline_dict['STEPS_COMPLETED']['Preprocessing'] = True
      with open(pipeline_path , 'wb') as file:
          pickle.dump(pipeline_dict, file)
      print(f"\n\n--- SAVED Data Preprocessing Pipeline {c} CONFIG AND RESULTS ---")

    ##################### Model Selection #####################################
    if run_model_selection:
      #Timing (Estimate 30min)
      pipeline_dict, _, _, _ = run_Model_Selection(pipeline_dict)

      pipeline_dict['STEPS_COMPLETED']['Model_Selection'] = True
      with open(pipeline_path, 'wb') as file:
        pickle.dump(pipeline_dict, file)
      print(f"\n\n--- SAVED Model Selection Pipeline {c} CONFIG AND RESULTS ---")
      

    #################### Feature Reduction #####################################
    if run_feature_reduction:
      #Timing (Estimated 4-8hrs)
      pipeline_dict, _,_,_,_,_ = Feature_reduction.main(pipeline_dict)

      pipeline_dict['STEPS_COMPLETED']['Feature_Reduction'] = True
      with open(pipeline_path , 'wb') as file:
        pickle.dump(pipeline_dict, file)
      print(f"\n\n--- SAVED Feature Reduction Pipeline {c} CONFIG AND RESULTS ---")

    #################### SHAP Analysis #####################################
    if run_SHAP_explain:
      #Timing (Estimated 30min)
      pipeline_dict, _,_,_,_ = get_shap_explainations.main(pipeline_dict)
      
      pipeline_dict['STEPS_COMPLETED']['SHAP'] = True
      with open(pipeline_path , 'wb') as file:
        pickle.dump(pipeline_dict, file)
      print(f"\n\n--- SAVED Shap Analysis Pipeline {c} CONFIG AND RESULTS ---")                        

    #################### Saving Pipeline Config and Results #####################################
    with open(pipeline_path , 'wb') as file:
      pickle.dump(pipeline_dict, file)
    print(f"\n\n--- SAVED PIPELINE for {c} CONFIG AND RESULTS ---")

if __name__ == "__main__":
    main()