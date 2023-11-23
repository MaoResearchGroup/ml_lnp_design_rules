import pickle
import plotting_functions as plotter
import os

def main():

        #Specify which plots to make
        plot_f_distribution    = True
        plot_model_selection   = True
        feature_reduction      = False
        run_learning_curve     = False
        run_SHAP_plots         = False
        plot_SHAP_cluster      = False #Fuse
        plot_Rose              = False #Fuse
        plot_bump              = False #Fuse



        file_path  = f"Runs/Test_PDI1_keep_Zeta_RLU2/"
        cell_type_list = ['HepG2', 'HEK293', 'N2a', 'ARPE19','B16', 'PC3']
        manuscript_save_path = file_path + 'Manuscript_Figures/'
        if not os.path.exists( manuscript_save_path):
                # Create the directory if it doesn't exist
                os.makedirs( manuscript_save_path)
                print(f"Directory '{manuscript_save_path}' created.")
        else:
                print(f"Directory '{manuscript_save_path}' already exists.")
        
        
        #### MANUSCRIPT WIDE FIGURES
        pipe_list = []
        for c in cell_type_list:
                #Import Pipeline of interest
                with open(file_path + f'{c}/Pipeline_dict.pkl', 'rb') as file:
                        pipeline_results = pickle.load(file)
                pipe_list.append(pipeline_results)
        if plot_f_distribution:
                plotter.plot_tfxn_dist_comp(pipeline_list = pipe_list,
                                                raw = True,
                                                save_path= manuscript_save_path)
                plotter.tfxn_heatmap(pipeline_list=pipe_list,
                                     save_path=manuscript_save_path)
        if plot_model_selection:
                plotter.plot_cell_comparision(pipeline_list = pipe_list,
                                              save_path = manuscript_save_path)
        
        
        
        ##### CELL TYPE SPECIFIC FIGURES ###########
        for c in cell_type_list:
                
                #Import Pipeline of interest
                with open(file_path + f'{c}/Pipeline_dict.pkl', 'rb') as file:
                        pipeline_results = pickle.load(file)

                RUN_NAME = pipeline_results['Saving']['RUN_NAME']
                pipeline_path             = f'{RUN_NAME}{c}/Pipeline_dict.pkl'

                # if run_SHAP_plots:
                #         print('\n###########################\n\n PLOTTING SHAP')
                #         refined_model_shap_plots.main(cell_model_list=best_cell_model, 
                #                         model_folder = refined_model_save_path, 
                #                         shap_value_path = shap_value_save_path, 
                #                         plot_save_path = shap_plot_save_path)

                # if plot_SHAP_cluster:
                #         print('\n###########################\n\n PLOTTING SHAP CLUSTER PLOT')
                #         SHAP_clustering.main(cell_model_list=best_cell_model,
                #                 shap_value_path=shap_value_save_path, 
                #                 model_save_path=refined_model_save_path,
                #                 figure_save_path=shap_cluster_save_path)
                # if plot_Rose:
                #         print('\n###########################\n\n PLOTTING ROSE PLOT')
                #         SHAP_radar_plot.main(cell_model_list=best_cell_model, 
                #                 model_folder=refined_model_save_path, 
                #                 shap_value_path=shap_value_save_path, 
                #                 figure_save_path=radar_plot_save_path, 
                #                 N_bins = 10, 
                #                 comp_features = ['NP_ratio', 
                #                                         'Dlin-MC3_Helper lipid_ratio',
                #                                         'Dlin-MC3+Helper lipid percentage', 
                #                                         'Chol_DMG-PEG_ratio'],
                #                 lipid_features = ['P_charged_centers', 
                #                                         'N_charged_centers', 
                #                                         'Double_bonds'],
                #                 phys_features= ['Zeta'])
                # if plot_bump:
                #         print('\n###########################\n\n PLOTTING BUMP PLOT')
                #         bump_plot.main(cell_model_list=best_cell_model,
                #                 model_folder=refined_model_save_path,
                #                 shap_value_path=shap_value_save_path,
                #                 plot_save_path= bump_plot_save_path,
                #                 N_bins=10,
                #                 feature_order=input_param_names)

                # ##################### PLOTTING #####################################

                        
                #Plotting Model Selection Results
                if plot_model_selection:
                        print('\n###########################\n\n MODEL SELECTION PLOTTING')
                        plotter.plot_AE_Box(pipeline_results)
                        plotter.plot_predictions(pipeline_results)

                #Plot Feature Reduction
                if feature_reduction:
                        print('\n###########################\n\n Feature Reduction PLOTTING')
                        plotter.plot_feature_reduction(pipeline_results)

                # Learning Curve 
                if run_learning_curve:
                        #Timing (10 minutes)
                        if pipeline_results['STEPS_COMPLETED']['Learning_Curve'] == False:
                                pipeline_results = plotter.get_learning_curve(pipeline_results)
                                pipeline_results['STEPS_COMPLETED']['Learning_Curve'] = True
                                with open(pipeline_path , 'wb') as file:
                                        pickle.dump(pipeline_results, file)
                                print(f"\n\n--- Updated Pipeline with Learning Curve {c} CONFIG AND RESULTS ---")

                        plotter.plot_learning_curve(pipeline_results)
if __name__ == "__main__":
    main()