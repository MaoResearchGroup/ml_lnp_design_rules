import pickle
from utilities import save_pipeline, truncate_colormap, get_Model_Selection_performance
import plotting_functions as plotter
import os
import matplotlib as plt

def main():

        #Specify which plots to make
        manuscript             = True
        prelim                 = True
        plot_f_distribution    = True
        plot_model_selection   = True
        feature_reduction      = True
        straw_model            = True
        run_learning_curve     = False
        run_SHAP_plots         = True
        plot_bump              = True


        RUN_NAME  = f"Runs/Percentage_PDI1_RLU1.5_SIZE10000/"

        #cell_type_list = ['HEK293']
        cell_type_list = ['HepG2','B16',  'PC3', 'HEK293',  'N2a', 'ARPE19']

        shap_cmap = truncate_colormap(plt.cm.get_cmap('Greens'), 0.2, 0.8)
        
        feature_plotting_order = ['HL_(IL+HL)',
                                  '(IL+HL)',
                                'PEG_(Chol+PEG)',
                                'NP_ratio',
                                'Lipid_NA_ratio', ############
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
        
        # feature_plotting_order = ['Dlin-MC3_Helper lipid_ratio',
        #                           'Dlin-MC3+Helper lipid percentage',
        #                         'Chol_DMG-PEG_ratio',
        #                         'NP_ratio',
        #                         'P_charged_centers', 
        #                         'N_charged_centers', 
        #                         'cLogP', 
        #                         'Hbond_D', 
        #                         'Hbond_A', 
        #                         'Total_Carbon_Tails', 
        #                         'Double_bonds',
        #                         'Size', 
        #                         'PDI', 
        #                         'Zeta']

        manuscript_save_path = RUN_NAME + 'Manuscript_Figures/'


        if not os.path.exists(manuscript_save_path):
                # Create the directory if it doesn't exist
                os.makedirs( manuscript_save_path)
                print(f"Directory '{manuscript_save_path}' created.")
        else:
                print(f"Directory '{manuscript_save_path}' already exists.")
        
        
        #### MANUSCRIPT WIDE FIGURES/TABLES
        if manuscript:
                #Get list of pipeline to compare results from
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
                                                        save= manuscript_save_path,
                                                        new_order = ['HepG2', 'PC3', 'B16', 'HEK293',  'N2a', 'ARPE19'],
                                                        pipe_order = cell_type_list)
                        plotter.tfxn_heatmap(pipeline_list=pipe_list,
                                        save=manuscript_save_path,
                                        helper_lipid=True)
                #Comparison of cell-cell model MAE
                if plot_model_selection:
                        plotter.plot_cell_comparision(pipeline_list = pipe_list,
                                                save= manuscript_save_path)
                        plotter.tabulate_model_selection_results(pipeline_list=pipe_list,
                                                                save=manuscript_save_path)
                if feature_reduction:
                        plotter.plot_refined_model_comparisions(pipeline_list= pipe_list,
                                                                save = manuscript_save_path)
                        plotter.tabulate_refined_model_results(pipeline_list=pipe_list,
                                                        cell_type_list=cell_type_list,
                                                        save=manuscript_save_path)
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

                        feature_list = pipeline_results['Data_preprocessing']['Input_Params'].copy()
                        feature_list.insert(0, 'Transfection_Efficiency')

                        for feature in feature_list:
                                plotter.feature_dist(pipeline = pipeline_results,
                                                  feature_name = feature,
                                                  raw = True,
                                                  save = prelim_save)
   
                #Plotting Model Selection Results
                if plot_model_selection:
                        print('\n########## MODEL SELECTION PLOTTING')
                        model_selection_save = pipeline_results['Saving']['Figures'] + 'Model_Selection/'
                        #Check save path
                        if os.path.exists(model_selection_save) == False:
                                os.makedirs(model_selection_save, 0o666)
                        
                        get_Model_Selection_performance(pipeline_results,
                                                        loop= 'test',
                                                        save = model_selection_save + "Performance_table.xlsx",)
                        plotter.plot_AE_Box(pipeline_results, model_selection_save, loop= 'test')
                        plotter.plot_predictions(pipeline_results, model_selection_save)
                        

                #Plot Feature Reduction
                if feature_reduction:
                        print('\n########## FEATURE REDUCTION PLOTTING')
                        plotter.plot_feature_reduction(pipeline_results)
                        
                if straw_model:
                        print('\n########## STRAW MODEL PLOTTING')
                        straw_save = pipeline_results['Saving']['Figures'] + 'Straw_Model/'
                        #Check save path
                        if os.path.exists(straw_save) == False:
                                os.makedirs(straw_save, 0o666)
                        plotter.plot_bar_with_t_test(df = pipeline_results['Straw_Model']['Results'].copy(), save = straw_save, label_column= 'Feature', value_column= 'KFold Average MAE', feature_order=['Control'] + feature_plotting_order)
                # Learning Curve 
                if run_learning_curve:
                        print('\n########## LEARNING CURVE PLOTTING')


                        plotter.plot_learning_curve(pipeline_results)

                #SHAP Plots
                if run_SHAP_plots:
                        print('\n########## SHAP PLOTTING')
                        SHAP_save = pipeline_results['Saving']['Figures'] + 'SHAP/'
                        #Check save path
                        if os.path.exists(SHAP_save) == False:
                                os.makedirs(SHAP_save, 0o666)

                        #Beeswarm Summary Plot
                        plotter.plot_summary(pipeline = pipeline_results,
                                             cmap = shap_cmap,
                                        #      feature_order=feature_plotting_order,
                                                feature_order=None,
                                             save = SHAP_save)
                
                        #Feature Importance
                        plotter.plot_importance(pipeline = pipeline_results,
                                                # feature_order=feature_plotting_order,
                                                feature_order=None,
                                                save = SHAP_save)
                        
                        #Embedded Colored by Feature Value
                        #Transfection
                        plotter.plot_SHAP_cluster(pipeline = pipeline_results,
                                             feature_name='Transfection Efficiency',
                                             cmap = 'Reds',
                                             size = 2.5,
                                             save = SHAP_save,
                                             title = False)
                        #Formulation Features

                        plotter.plot_SHAP_cluster(pipeline = pipeline_results,
                                        feature_name='all',
                                        cmap = 'viridis_r',
                                        size = 1.8,
                                        save = SHAP_save,
                                        title = False)
                        ### EMBEDDED by SHAP VALUES
                        #Formulation Features

                        # plotter.plot_SHAP_cluster(pipeline = pipeline_results,
                        #                 feature_name='all',
                        #                 cmap = 'cool',
                        #                 size = 1.8,
                        #                 save = SHAP_save,
                        #                 shap_values = True,
                        #                 title = False)
                
                        # #Embedded Colored by SHAP Value
                        # plotter.plot_embedded(pipeline=pipeline_results, 
                        #                 feature_name= 'all',
                        #                 size = 2,
                        #                 save = SHAP_save)

                        #Plot dependence
                        # plotter.plot_dependence(pipeline=pipeline_results,
                        #                         feature_list = ['HL_(IL+HL)', 
                        #                                         '(IL+HL)', 
                        #                                         'PEG_(Chol+PEG)', 
                        #                                         'P_charged_centers'],
                        #                         interaction_feature_list= ['HL_(IL+HL)', 
                        #                                                 '(IL+HL)', 
                        #                                                    'P_charged_centers'],
                        #                         save = SHAP_save)
                        # #Plot Interaction
                        # plotter.plot_interaction(pipeline = pipeline_results,
                        #                          cmap = 'viridis_r',
                        #                          save = SHAP_save)
                        
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