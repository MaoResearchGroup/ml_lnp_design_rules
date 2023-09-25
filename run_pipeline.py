import matplotlib.pyplot as plt
import run_NESTED_CV
import Model_Selection_Results
import Feature_reduction
import learning_curve_plot
import refined_model_shap_explainations
import refined_model_shap_plots
import SHAP_radar_plot
import refined_model_shap_plots
import utilities



def main():
  ################ What parts of the pipeline to run ###############
  run_model_selection   = True
  plot_model_selection  = False
  run_learning_curve    = True
  run_feature_reduction = False
  run_SHAP_explain      = False
  run_SHAP_plots        = False
  plot_Rose             = False
  
  
  ################ SAVING, LOADING##########################
  RUN_NAME = "Models_Final_All_Size_PDI_Zeta"
  data_file_path = 'Raw_Data/Final_Master_Formulas.csv' #Where to extract training data
  figure_save_path = f"BMES_Figures/" #where to save figures
  model_save_path = f"Trained_Models/{RUN_NAME}/" # Where to save model, results, and training data
  learning_curve_save_path = f"{figure_save_path}Training_size/{RUN_NAME}/" #where to save learning curve results
  refined_model_save_path = f"Feature_Reduction/{RUN_NAME}/" #where to save refined model and results
  shap_value_save_path = f'SHAP_Values/{RUN_NAME}/'
  shap_plot_save_path = f"{figure_save_path}SHAP/{RUN_NAME}/"
  radar_plot_save_path = f"{figure_save_path}Radar/{RUN_NAME}/"

  ############## CELLS, ALGORITHMS, PARAMETERS ####################################
  #model_list = ['RF', 'MLR', 'lasso', 'PLS', 'kNN', 'LGBM', 'XGB', 'DT']#Did not include SVR 
  model_list = ['LGBM']
  best_model_list = ['LGBM'] 
  # cell_type_list = ['HepG2','HEK293','N2a', 'ARPE19', 'B16', 'PC3']
  #model_list = ['LGBM'] 
  cell_type_list = ['B16']
  size = True
  zeta = True
  size_cutoff = 100000
  PDI_cutoff = 1 #Use 1 to include all data
  N_CV = 5
  prefix = "RLU_" #WARNING: HARDCODED
  input_param_names = utilities.select_input_params(size, zeta)
  print(f"INITIAL INPUT PARAMS: {input_param_names}")


  ##################### Screen and Optimize Model #####################################
  #Model Optimization and Selection
  if run_model_selection:

    print('\n###########################\n\n MODEL SELECTION AND OPTIMIZATION')
    for cell in cell_type_list:

      for model_name in model_list:

        print("#"*10 + f"  TRAINING:{model_name} for {cell}  " + "#"*10 )
        run_NESTED_CV.run_NESTED_CV(model_name = model_name,
                      cell = cell, 
                      data_file_path = data_file_path, 
                      save_path = model_save_path, 
                      input_params = input_param_names,
                      size_cutoff = size_cutoff, 
                      PDI_cutoff = PDI_cutoff,
                      prefix = prefix, 
                      N_CV = N_CV)
  
  # #Plotting Model Selection
  if plot_model_selection:
    print('\n###########################\n\n MODEL SELECTION PLOTTING')
    Model_Selection_Results.main(RUN_NAME = RUN_NAME, 
                                model_folder=model_save_path, 
                                figure_save_path=figure_save_path,
                                cell_type_list=cell_type_list,
                                model_list= model_list,
                                N_CV=N_CV)
  ##################### Learning Curve #####################################
  if run_learning_curve:
    print('\n###########################\n\n RUNNING LEARNING CURVE')
    learning_curve_plot.main(model_list=best_model_list, 
                            cell_type_list= cell_type_list, 
                            data_file_path=data_file_path, 
                            save_path=learning_curve_save_path, 
                            model_folder=model_save_path, 
                            input_param_names=input_param_names, 
                            size_cutoff=size_cutoff, 
                            PDI_cutoff=PDI_cutoff, 
                            prefix=prefix, 
                            NUM_ITER=10)

  #################### Feature Reduction #####################################
  if run_feature_reduction:
    print('\n###########################\n\n RUNNING FEATURE REDUCTION')
    Feature_reduction.main(cell_names= cell_type_list,
                            model_list= best_model_list,
                            model_save_path= model_save_path,
                            refined_model_save_path= refined_model_save_path, 
                            input_param_names=input_param_names, 
                            prefix=prefix, 
                            N_CV = N_CV)
  
  #################### SHAP Analysis #####################################
  if run_SHAP_explain:
    print(input_param_names)
    print('\n###########################\n\n RUNNING SHAP EXPLANATIONS')
    refined_model_shap_explainations.main(model_list = best_model_list, 
                                     cell_type_list = cell_type_list, 
                                     model_folder = refined_model_save_path, 
                                     shap_save_path = shap_value_save_path)
  if run_SHAP_plots:
    print('\n###########################\n\n PLOTTING SHAP')
    refined_model_shap_plots.main(model_list=best_model_list, 
                             cell_type_list=cell_type_list, 
                             model_folder = refined_model_save_path, 
                             shap_value_path = shap_value_save_path, 
                             plot_save_path = shap_plot_save_path)
    
  if plot_Rose:
    print('\n###########################\n\n PLOTTING ROSE PLOT')
    SHAP_radar_plot.main(model_list=best_model_list, 
                         cell_type_list=cell_type_list, 
                         model_save_path=refined_model_save_path, 
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
if __name__ == "__main__":
    main()