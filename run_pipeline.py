from utilities import extract_training_data, init_pipeline, save_pipeline
import plotting_functions as plotter
from run_Model_Selection import run_Model_Selection
import HL_1_performance
import Feature_reduction
import straw_model
import get_shap_explainations
import pickle
import os



"""
run_pipeline script

- Used to generate explainable ML models for LNP transfection prediction
- Can control which parts of the pipeline to run and related data saving procedures
- user input required for datapreprocessing, cell lists, and model lists

"""

def main():


  ################ What parts of the pipeline to run ###############
  #Make New pipeline
  new_pipeline = False
  
  #Parts to Run/Update
  run_preprocessing     = False
  run_model_selection   = False
  run_HL_1              = False
  run_feature_reduction = False
  run_straw_model       = False
  run_learning_curve    = True
  run_SHAP_explain      = False
  
  redo_learning_curve   = True
  #Cell types to Run
  cell_type_list = ['B16']
  # cell_type_list = ['B16', 'HepG2','HEK293', 'N2a', 'ARPE19', 'PC3']

  
  ############### PARAMETERS ###############################
  model_list = ['LGBM']

  # model_list = ['RF','LGBM', 'XGB', 'DT', 'MLR', 'lasso', 'PLS', 'kNN', 'MLP']
  formula_type = 'percent' #options: ratio, percent, weight

  chemical_features = 'HL'       # options: HL, OHE, blended

  RLU_floor = 1.5
  size_cutoff = 10000
  PDI_cutoff = 1 #Use 1 to include all data

  N_CV = 5
  prefix = "RLU_" # "RLU_"

  ################ SAVING, LOADING##########################
  RUN_NAME                  = f"Runs/example_{chemical_features}_Features_PDI{PDI_cutoff}_RLU{RLU_floor}_SIZE{size_cutoff}/"

  data_file_path            = 'Raw_Data/1080_LNP_OHE.csv' #Where to extract training data

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
                                    param_type = formula_type,
                                    chemical_type= chemical_features,
                                    data_file_path=data_file_path,
                                    size_cutoff=size_cutoff,
                                    PDI_cutoff=PDI_cutoff,
                                    prefix=prefix,
                                    RLU_floor=RLU_floor,
                                    N_CV=N_CV)
      #Run Whole Pipeline
      run_preprocessing     = True
      run_model_selection   = True
      run_HL_1              = True
      run_feature_reduction = True
      run_straw_model       = True
      run_learning_curve    = True
      run_SHAP_explain      = True

    ##################### Extract Training Data ###############################
    if run_preprocessing:
      pipeline_dict, _, _, _= extract_training_data(pipeline_dict)

      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'DATA PREPROCESSING')
      

    ##################### Model Selection #####################################
    if run_model_selection:
      #Timing (Estimate 5-10 min per cell)
      pipeline_dict, _, _, _ = run_Model_Selection(pipeline_dict)

      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'MODEL SELECTION')
      
    if run_HL_1:
       #Timing (Estimate 5 min per cell)
      pipeline_dict, _ = HL_1_performance.main(pipeline_dict)

      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'MODEL SELECTION')
    #################### Feature Reduction #####################################
    if run_feature_reduction:
      #Timing (Estimated 1-2hr per cell)
      pipeline_dict, _,_,_,_,_ = Feature_reduction.main(pipeline=pipeline_dict, repeats = 5)
      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'FEATURE REDUCTION')
    
    if run_straw_model:
      #Timing (Estimated 5-10min per cell)
      composition_features = ['HL_(IL+HL)',
                                  '(IL+HL)',
                                'PEG_(Chol+PEG)',
                                'NP_ratio']
      HL_features = ['Lipid_NA_ratio',
                                'P_charged_centers', 
                                'N_charged_centers', 
                                'cLogP', 
                                'Hbond_D', 
                                'Hbond_A', 
                                'Total_Carbon_Tails', 
                                'Double_bonds']
      polar_features = ['P_charged_centers', 
                                'N_charged_centers', 
                                'Hbond_D', 
                                'Hbond_A']
      
      apolar_features = ['cLogP', 
                                'Total_Carbon_Tails', 
                                'Double_bonds']

      NP_features = ['Size', 
                                'PDI',
                                'Zeta']

      pipeline_dict = straw_model.main(pipeline=pipeline_dict,
                                        params_to_test= [polar_features] + [apolar_features]+ [HL_features] + [NP_features],
                                        NUM_TRIALS= 20,
                                        new_run=True)

      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'STRAW MODEL')
    if run_learning_curve:      
      #Timing (10 minutes per cell)
      if pipeline_dict['STEPS_COMPLETED']['Learning_Curve'] == False or redo_learning_curve:
              pipeline_dict = plotter.get_learning_curve(pipeline_dict, refined = False)
              
              save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                            step= 'LEARNING CURVE')
    
    
    
    #################### SHAP Analysis #####################################
    if run_SHAP_explain:
      #Timing (Estimated 1-2 minute per cell)

      #refined features
      pipeline_dict, _,_,_,_ = get_shap_explainations.main(pipeline_dict, N_bins = 10, refined = True)

      #All features
      pipeline_dict, _,_,_,_ = get_shap_explainations.main(pipeline_dict, N_bins = 10, refined = False)
      
      save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'SHAP')                      

    #################### Saving Pipeline Config and Results #####################################
    save_pipeline(pipeline=pipeline_dict, path = pipeline_path, 
                    step = 'FINAL SAVE')  

if __name__ == "__main__":
    main()