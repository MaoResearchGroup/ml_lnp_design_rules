import matplotlib.pyplot as plt
import pandas as pd
import run_NESTED_CV
import Model_Selection_Results
import plot_feature_distribution
import Feature_reduction
import learning_curve_plot
import refined_model_shap_explainations
import refined_model_shap_plots
import SHAP_radar_plot
import refined_model_shap_plots
import utilities
import SHAP_clustering
import bump_plot
import time
import os

def main():
  ################ What parts of the pipeline to run ###############
  plot_f_distribution   = False
  run_model_selection   = False
  plot_model_selection  = True
  run_learning_curve    = False
  run_feature_reduction = False
  run_SHAP_explain      = False
  run_SHAP_plots        = False
  plot_SHAP_cluster     = False
  plot_Rose             = False
  plot_bump             = False
  
  

  ############### PARAMETERS ###############################

  model_list = ['RF', 'MLR', 'lasso', 'PLS', 'kNN', 'LGBM', 'XGB', 'DT']#Did not include SVR 
  cell_type_list = ['HepG2','HEK293','B16', 'PC3', 'N2a', 'ARPE19']
  #cell_type_list = ['HepG2']
  #List helper lipids

  # #Testing
  # best_cell_model = [('HepG2', 'RF')]
  # model_list = ['LGBM']#Did not include SVR 
  # cell_type_list = ['HepG2']

  ratiometric = True
  RLU_floor = 2
  size = True
  zeta = True
  keep_PDI = True
  size_cutoff = 100000
  PDI_cutoff = 1 #Use 1 to include all data
  N_CV = 5
  prefix = "RLU_" #WARNING: HARDCODED
  input_param_names = utilities.select_input_params(size, zeta, ratio = ratiometric)
  print(f"INITIAL INPUT PARAMS: {input_param_names}")
  ################ SAVING, LOADING##########################
  #RUN_NAME                  = f"Runs/wt_HepG2_All_Size_PDI{PDI_cutoff}_keep_Zeta_RLU{RLU_floor}"

  RUN_NAME                  = f"Runs/Models_Final_All_Size_PDI{PDI_cutoff}_keep_Zeta_RLU{RLU_floor}"

  data_file_path            = 'Raw_Data/Final_Master_Formulas.csv' #Where to extract training data
  model_save_path           = f"{RUN_NAME}/Trained_Models/" # Where to save model, results, and training data 
  refined_model_save_path   = f"{RUN_NAME}/Feature_Reduction/" #where to save refined model and results
  shap_value_save_path      = f'{RUN_NAME}/SHAP_Values/'
  figure_save_path          = f"{RUN_NAME}/Figures/" #where to save figures
  features_save_path        = f"{figure_save_path}Features/"
  model_selection_save_path = f"{figure_save_path}Model_Selection/"
  learning_curve_save_path  = f"{figure_save_path}Learning_Curve/" #where to save learning curve results
  shap_plot_save_path       = f"{figure_save_path}SHAP/"
  shap_cluster_save_path    = f"{figure_save_path}SHAP_Clustering/"
  radar_plot_save_path      = f"{figure_save_path}Radar/"
  bump_plot_save_path       = f"{figure_save_path}Bump/"




  ##################### Screen and Optimize Model #####################################

  #Model Optimization and Selection
  if plot_f_distribution:
    plot_feature_distribution.main(cell_type_list=cell_type_list, 
                                   data_file_path= data_file_path, 
                                   input_param_names=input_param_names, 
                                   save_path=features_save_path, 
                                   size=size_cutoff,
                                   PDI = PDI_cutoff,
                                   keep_PDI = keep_PDI,
                                   RLU_floor= 0,
                                   prefix=prefix)
    

  if run_model_selection:
    print('\n###########################\n\n MODEL SELECTION AND OPTIMIZATION')
    #Timing (Estimated 2-3hrs)
    start_time = time.time()
    for cell in cell_type_list:

      #Helper lipids (Extract the training data for the cell type.
      # Loop through the unique helper lipids within that dataset

      for model_name in model_list:
        print("#"*10 + f"  TRAINING:{model_name} for {cell}  " + "#"*10 )
        run_NESTED_CV.run_NESTED_CV(model_name = model_name,
                      cell = cell, 
                      data_file_path = data_file_path, 
                      save_path = model_save_path, 
                      input_params = input_param_names,
                      size_cutoff = size_cutoff, 
                      PDI_cutoff = PDI_cutoff,
                      keep_PDI = keep_PDI,
                      prefix = prefix, 
                      RLU_floor = RLU_floor,
                      N_CV = N_CV)
    print("\n\n--- %s minutes for MODEL SELECTION---" % ((time.time() - start_time)/60))


  #Plotting Model Selection
  if plot_model_selection:
    print('\n###########################\n\n MODEL SELECTION PLOTTING')
    Model_Selection_Results.main(model_folder=model_save_path, 
                                figure_save_path=model_selection_save_path,
                                cell_type_list=cell_type_list,
                                model_list= model_list,
                                N_CV=N_CV)
  
  #Extract best models and cell types
  print('\n###########################\n\n SELECTING BEST MODELS')

  #Check whether selection has been made before
  if os.path.exists(model_save_path + "Best_Model_Cell.csv"):
    cell_model = pd.read_csv(model_save_path + "Best_Model_Cell.csv")
  else:
    cell_model = utilities.get_best_model_cell(model_selection_save_path, 
                                              model_folder=model_save_path,
                                              cell_type_list=cell_type_list)      
  best_cell_model = list(cell_model.drop(cell_model.columns[2], axis=1).itertuples(index=False, name=None))
  print("Best models/cell pairs:",best_cell_model)

  ##################### Learning Curve #####################################
  if run_learning_curve:
    print('\n###########################\n\n RUNNING LEARNING CURVE')
    #Timing
    start_time = time.time()
    learning_curve_plot.main(cell_model_list=best_cell_model, 
                            data_file_path=data_file_path, 
                            save_path=learning_curve_save_path, 
                            model_folder=model_save_path, 
                            input_param_names=input_param_names, 
                            size_cutoff=size_cutoff, 
                            PDI_cutoff=PDI_cutoff, 
                            keep_PDI = keep_PDI,
                            prefix=prefix, 
                            NUM_ITER=10)
    print("\n\n--- %s minutes for Learning Curve---" % ((time.time() - start_time)/60))

  #################### Feature Reduction #####################################
  if run_feature_reduction:
    print('\n###########################\n\n RUNNING FEATURE REDUCTION')
    #Timing (Estimated 4-8hrs)
    start_time = time.time()
    Feature_reduction.main(cell_model_list=best_cell_model,
                            model_save_path= model_save_path,
                            refined_model_save_path= refined_model_save_path, 
                            input_param_names=input_param_names, 
                            keep_PDI=keep_PDI,
                            prefix=prefix, 
                            N_CV = N_CV)
    print("\n\n--- %s minutes for feature reduction---" % ((time.time() - start_time)/60))
  #################### SHAP Analysis #####################################
  if run_SHAP_explain:
    print('\n###########################\n\n RUNNING SHAP EXPLANATIONS')
        #Timing
    start_time = time.time()
    refined_model_shap_explainations.main(cell_model_list=best_cell_model, 
                                     model_folder = refined_model_save_path, 
                                     shap_save_path = shap_value_save_path)
    print("\n\n--- %s minutes for SHAP explanation---" % ((time.time() - start_time)/60))
  if run_SHAP_plots:
    print('\n###########################\n\n PLOTTING SHAP')
    refined_model_shap_plots.main(cell_model_list=best_cell_model, 
                             model_folder = refined_model_save_path, 
                             shap_value_path = shap_value_save_path, 
                             plot_save_path = shap_plot_save_path)
  
  if plot_SHAP_cluster:
    print('\n###########################\n\n PLOTTING SHAP CLUSTER PLOT')
    SHAP_clustering.main(cell_model_list=best_cell_model,
                         shap_value_path=shap_value_save_path, 
                         model_save_path=refined_model_save_path,
                         figure_save_path=shap_cluster_save_path)
  if plot_Rose:
    print('\n###########################\n\n PLOTTING ROSE PLOT')
    SHAP_radar_plot.main(cell_model_list=best_cell_model, 
                         model_folder=refined_model_save_path, 
                         shap_value_path=shap_value_save_path, 
                         figure_save_path=radar_plot_save_path, 
                         N_bins = 10, 
                         comp_features = ['NP_ratio', 
                                          'Dlin-MC3_Helper lipid_ratio',
                                          'Dlin-MC3+Helper lipid percentage', 
                                          'Chol_DMG-PEG_ratio'],
                         lipid_features = ['P_charged_centers', 
                                           'N_charged_centers', 
                                           'Double_bonds'],
                         phys_features= ['Zeta'])
  if plot_bump:
    print('\n###########################\n\n PLOTTING BUMP PLOT')
    bump_plot.main(cell_model_list=best_cell_model,
                    model_folder=refined_model_save_path,
                    shap_value_path=shap_value_save_path,
                    plot_save_path= bump_plot_save_path,
                    N_bins=10)
if __name__ == "__main__":
    main()