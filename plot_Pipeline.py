import pickle
import plotting_functions as plotter
import os

def main():

        #Specify which plots to make
        prelim                 = False
        plot_f_distribution    = True
        plot_model_selection   = False
        feature_reduction      = False
        run_learning_curve     = False
        run_SHAP_plots         = False
        plot_bump              = False 


        RUN_NAME  = f"Runs/Final_PDI1_RLU2/"


        cell_type_list = ['HepG2', 'PC3', 'HEK293', 'B16',  'N2a', 'ARPE19']

        feature_plotting_order = ['Dlin-MC3+Helper lipid percentage',
                                'Dlin-MC3_Helper lipid_ratio',
                                'Chol_DMG-PEG_ratio',
                                'NP_ratio',
                                'P_charged_centers', 
                                'N_charged_centers', 
                                'cLogP', 
                                'Hbond_D', 
                                'Hbond_A', 
                                'Total_Carbon_Tails', 
                                'Double_bonds',
                                'Size', 
                                'PDI', 
                                'Zeta']
        manuscript_save_path = RUN_NAME + 'Manuscript_Figures/'
        if not os.path.exists(manuscript_save_path):
                # Create the directory if it doesn't exist
                os.makedirs( manuscript_save_path)
                print(f"Directory '{manuscript_save_path}' created.")
        else:
                print(f"Directory '{manuscript_save_path}' already exists.")
        
        
        #### MANUSCRIPT WIDE FIGURES
        pipe_list = []
        for c in cell_type_list:
                #Import Pipeline of interest
                with open(RUN_NAME + f'{c}/Pipeline_dict.pkl', 'rb') as file:
                        pipeline_results = pickle.load(file)
                pipe_list.append(pipeline_results)
        
        #Comparison of Transfection
        if plot_f_distribution:
                plotter.plot_tfxn_dist_comp(pipeline_list = pipe_list,
                                                raw = True,
                                                save= manuscript_save_path)
                plotter.tfxn_heatmap(pipeline_list=pipe_list,
                                     save=manuscript_save_path)
        #Comparison of cell-cell model MAE
        if plot_model_selection:
                plotter.plot_cell_comparision(pipeline_list = pipe_list,
                                              save= manuscript_save_path)
        
        #Design Feature Bump Plots
        if plot_bump:
                plotter.bumpplot(pipeline_list = pipe_list,
                                 lw = 3,
                                 feature_order= feature_plotting_order,
                                 save= manuscript_save_path)
        
        
        ##### CELL TYPE SPECIFIC FIGURES ###########
        for c in cell_type_list:
                print(f'\n\n ############# Plotting for {c} ##############')
                #Import Pipeline of interest
                with open(RUN_NAME + f'{c}/Pipeline_dict.pkl', 'rb') as file:
                        pipeline_results = pickle.load(file)

                RUN_NAME = pipeline_results['Saving']['RUN_NAME']
                pipeline_path             = f'{RUN_NAME}{c}/Pipeline_dict.pkl'

                #Prelimary Dataset
                if prelim:
                        prelim_save = pipeline_results['Saving']['Figures'] + 'Prelim/'
                        #Check save path
                        if os.path.exists(prelim_save) == False:
                                os.makedirs(prelim_save, 0o666)


                        plotter.tfxn_dist(pipeline = pipeline_results,
                                          raw = True,
                                          save = prelim_save)
                        plotter.feature_distribution(pipeline = pipeline_results,
                                             save = prelim_save)        
                #Plotting Model Selection Results
                if plot_model_selection:
                        print('\n########## MODEL SELECTION PLOTTING')
                        model_selection_save = pipeline_results['Saving']['Figures'] + 'Model_Selection/'
                        #Check save path
                        if os.path.exists(model_selection_save) == False:
                                os.makedirs(model_selection_save, 0o666)
                        

                        plotter.plot_AE_Box(pipeline_results, model_selection_save)
                        plotter.plot_predictions(pipeline_results, model_selection_save)

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
                #SHAP Plots
                if run_SHAP_plots:
                        SHAP_save = pipeline_results['Saving']['Figures'] + 'SHAP/'
                        #Check save path
                        if os.path.exists(SHAP_save) == False:
                                os.makedirs(SHAP_save, 0o666)

                        #Beeswarm Summary Plot
                        plotter.plot_summary(pipeline = pipeline_results,
                                             cmap = 'viridis_r',
                                             feature_order=True,
                                             save = SHAP_save,
                                             order = feature_plotting_order)
                
                        #Feature Importance
                        plotter.plot_importance(pipeline = pipeline_results,
                                                feature_order=True,
                                                order = feature_plotting_order,
                                                save = SHAP_save)
                        
                        #Embedded Colored by Feature Value
                        #Transfection
                        plotter.plot_SHAP_cluster(pipeline = pipeline_results,
                                             feature_name='Transfection Efficiency',
                                             cmap = 'Reds',
                                             size = 3.5,
                                             save = SHAP_save,
                                             title = False)
                        #Formulation Features
                        for f in ['NP_ratio',
                                   'Chol_DMG-PEG_ratio',
                                   'Dlin-MC3+Helper lipid percentage',
                                   'Dlin-MC3_Helper lipid_ratio' ]: 
                                plotter.plot_SHAP_cluster(pipeline = pipeline_results,
                                                feature_name=f,
                                                cmap = 'viridis_r',
                                                size = 2,
                                                save = SHAP_save,
                                                title = False)
                        
                                #Embedded Colored by SHAP Value
                                plotter.plot_embedded(pipeline=pipeline_results, 
                                                feature_name= f,
                                                save = SHAP_save)

                        #Plot dependence
                        plotter.plot_dependence(pipeline=pipeline_results,
                                                feature_name = 'Dlin-MC3_Helper lipid_ratio',
                                                interaction_feature= 'P_charged_centers',
                                                save = SHAP_save)
                        #Plot Interaction
                        plotter.plot_interaction(pipeline = pipeline_results,
                                                 cmap = 'viridis_r',
                                                 save = SHAP_save)
                        
                        #Radar/Rose Plots
                        plotter.plot_Radar(pipeline= pipeline_results,
                                           save = SHAP_save)
                        plotter.plot_Rose(pipeline= pipeline_results,
                                          save = SHAP_save)
                        
                        # #Plot Force Plot (Not used for manuscript)
                        # plotter.plot_force(formulation = 1,
                        #                    pipeline = pipeline_results)

if __name__ == "__main__":
    main()