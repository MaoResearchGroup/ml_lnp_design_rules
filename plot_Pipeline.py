import pickle
from utilities import truncate_colormap, get_Model_Selection_performance
from plotting import plotting_functions as plotter
import os
import matplotlib as plt

def main():

        """
        plot_pipeline script
        
        - Generate figures/tables to visualize model performance, feature refinement, SHAP analysis
        - User can control which plots to make and color schemes
        - Example code for helper lipid chemical feature models for the B16F10 cell type (used in the main manuscript). 
                - ML Pipeline will need to be run (run_pipeline.py) to general models and results for cell-wise comparison figures.
        """

        #Specify which plots to make
        cell_comparison        = True # Run cell-wise comparision figures
        cell_specific          = False  # cell type specific plots

        prelim                 = True  # Run initial data analysis of training data
        plot_f_distribution    = True  # Plot distributions of input features
        plot_model_selection   = True  # Model selection and HL-1 performance plots
        feature_reduction      = True  # Feature reduction plots
        straw_model            = True  # Straw model plots (plotted using Graphpad in the manuscript instead)
        run_learning_curve     = True  # Plot learning curves
        run_SHAP_plots         = True # SHAP plots
        refined_shap           = True # Whether to used feature refined models (True) or models trained on all provided data (False)
        plot_bump              = False # Compiled LNP Design rules

        RUN_NAME  = f"Runs/Final_HL_Features_PDI1_RLU1.5_SIZE10000/"

        
        cell_type_list = ['B16'] #must match naming on raw data files.

        shap_cmap = truncate_colormap(plt.cm.get_cmap('viridis_r'), 0.1, 1) #color map for SHAP plots
        
        feature_plotting_order = ['HL_(IL+HL)',
                                  '(IL+HL)',
                                'PEG_(Chol+PEG)',
                                'NP_ratio',
                                'P_charged_centers',
                                'N_charged_centers', 
                                'cLogP', 
                                'Hbond_D', 
                                'Hbond_A', 
                                'Total_Carbon_Tails', 
                                'Double_bonds',
                                '18PG', '14PA', 'DOPE', 'DSPC', 'DOTAP', 'DDAB',
                                'Size', 
                                'PDI', 
                                'Zeta']

        cell_comp_save_path = RUN_NAME + 'cell_comp_figures/'

        comp_type = ['prelim', 'model selection', 'feature reduction', 'SHAP']

        for t in comp_type:
                if not os.path.exists(cell_comp_save_path + t + '/'):
                        # Create the directory if it doesn't exist
                        os.makedirs(cell_comp_save_path + t + '/')
                        print(f"Directory '{cell_comp_save_path + t + '/'}' created.")
                else:
                        print(f"Directory '{cell_comp_save_path + t + '/'}' already exists.")
        
        
        #### Cell-wise comparison FIGURES/TABLES
        if cell_comparison:
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
                                                        save= cell_comp_save_path + 'prelim/',
                                                        new_order = ['HepG2', 'PC3', 'B16', 'HEK293',  'N2a', 'ARPE19'],
                                                        pipe_order = cell_type_list)
                        plotter.tfxn_heatmap(pipeline_list=pipe_list,
                                        save=cell_comp_save_path + 'prelim/',
                                        helper_lipid=True)
                #Comparison of cell-cell model MAE
                if plot_model_selection:
                        plotter.plot_cell_comparision(pipeline_list = pipe_list,
                                                save= cell_comp_save_path + 'model selection/')
                        plotter.tabulate_model_selection_results(pipeline_list=pipe_list,
                                                                save=cell_comp_save_path)
                if feature_reduction:
                        plotter.tabulate_refined_model_results(pipeline_list=pipe_list,
                                                        cell_type_list=cell_type_list,
                                                        save=cell_comp_save_path+ 'feature reduction/')
                #Design rule dot plot
                if plot_bump:
                        if refined_shap:
                                param_type = 'refined'
                        else:
                                param_type = 'original'
                        plotter.heat_dot(pipeline_list = pipe_list,
                                         param_type=param_type,
                                        cmap=shap_cmap,
                                        mk_size= 2e3,
                                        feature_order= feature_plotting_order,
                                        save= cell_comp_save_path+ 'SHAP/')
                        
            
        ##### CELL TYPE SPECIFIC FIGURES ###########
        elif cell_specific:
                for c in cell_type_list:
                        print(f'\n\n ############# Plotting for {c} ##############')
                        #Import Pipeline of interest
                        with open(RUN_NAME + f'{c}/Pipeline_dict.pkl', 'rb') as file:
                                pipeline_results = pickle.load(file)

                        RUN_NAME = pipeline_results['Saving']['RUN_NAME']
                        
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
                                                        save = prelim_save,
                                                        show_legend= False)
        
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
                        
                        # Learning Curve 
                        if run_learning_curve:
                                print('\n########## LEARNING CURVE PLOTTING')
                                plotter.plot_learning_curve(pipeline_results)

                        #SHAP Plots
                        if run_SHAP_plots:
                                print('\n########## SHAP PLOTTING')
                                if refined_shap:
                                        param_type = 'refined'
                                else:
                                        param_type = 'original'
                                SHAP_save = pipeline_results['Saving']['Figures'] + f'SHAP/{param_type}/'
                                #Check save path
                                if os.path.exists(SHAP_save) == False:
                                        os.makedirs(SHAP_save, 0o666)

                                #SHAP Beeswarm Summary Plot
                                plotter.plot_summary(pipeline = pipeline_results,
                                                param_type=param_type,
                                                cmap = shap_cmap,
                                                feature_order=feature_plotting_order,
                                                save = SHAP_save)
                        
                                #SHAP General Feature Importance
                                plotter.plot_importance(pipeline = pipeline_results,
                                                        param_type=param_type,
                                                        feature_order=feature_plotting_order,
                                                        save = SHAP_save)
                                
                                
                                ### Formulation Relative Feature values EMBEDDED onto all SHAP VALUES
                                plotter.plot_SHAP_cluster(pipeline = pipeline_results,
                                                        param_type=param_type,
                                                feature_name='all',
                                                cmap = 'cool',
                                                size = 1.8,
                                                save = SHAP_save,
                                                shap_values = True,
                                                title = False)
                        
                                ### Feature specific SHAP values EMBEDDED onto all SHAP VALUES
                                plotter.plot_embedded(pipeline=pipeline_results, 
                                                param_type=param_type,
                                                feature_name= 'all',
                                                size = 2,
                                                save = SHAP_save)

if __name__ == "__main__":
    main()