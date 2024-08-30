import pandas as pd
import numpy as np
from scipy import stats, interpolate
import seaborn as sns
import plotly.express as px
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
from utilities import get_spearman, run_tukey, extraction_all, get_Model_Selection_Error, run_anova
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from copy import deepcopy, copy
from sklearn.model_selection import learning_curve
import time
import shap
from itertools import chain
from matplotlib.colors import Normalize
from Scatter_map import scattermap


"""
plotting_functions

- contains all the custom plotting functions used for this study

"""


########## INITIAL FEATURE ANALYSIS PLOTS ##################
def tfxn_heatmap(pipeline_list, save, helper_lipid = False):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12

    cell_type_list = []
    cell_tfxn_list = []
    HL_list        = [] #Only used if helper lipid is true
    for pipe in pipeline_list:
        prefix = pipe['Data_preprocessing']['prefix']
        cell_type_list.append(pipe['Cell'])
        cell_tfxn_list.append(pipe['Data_preprocessing']['all_proc_data'][prefix + pipe['Cell']])
        

    training_data = pd.concat(cell_tfxn_list, axis = 1)


    #Initiate a correlation matrix of zeros
    all_corr = pd.DataFrame(np.zeros((len(cell_type_list),len(cell_type_list))), index = cell_type_list, columns = cell_type_list)
    lipid_corr = pd.DataFrame(np.zeros((len(cell_type_list),len(cell_type_list))), index = cell_type_list, columns = cell_type_list)

    #Get correlation between all cell types
    for cell1 in cell_type_list:
        for cell2 in cell_type_list:
            all_corr.loc[cell1, cell2] = get_spearman(training_data, cell1, cell2)
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    heatmap = sns.heatmap(round(np.abs(all_corr),2), vmin=.2, vmax=1, cmap='Blues', annot = True, annot_kws={"size": 8}, cbar = False)
    heatmap.invert_yaxis()
    plt.tick_params(axis='y', which='both', labelsize=10)
    plt.tick_params(axis='x', which='both', labelsize=10)


    cbar = heatmap.figure.colorbar(heatmap.collections[0])
    cbar.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 8)
    cbar.ax.tick_params(labelsize=6)

    plt.gca()
    plt.savefig(save + 'All_Data_Tfxn_Heatmap.svg', dpi=600, format = 'svg', transparent=True, bbox_inches='tight')
    plt.close()           
    
    # Save the all correlation data to csv
    with open(save + 'All_tfxn_correlation.csv', 'w', encoding = 'utf-8-sig') as file:
        all_corr.to_csv(file)

    ### PLOTTING THE HEATMAP FOR LIPID SPECIFIC FORMULATIONS ACROSS CELL TYPES   

    if helper_lipid:
        for pipe in pipeline_list:
            HL = pipe['Data_preprocessing']['all_proc_data']['Helper_lipid'].copy()
            HL = HL.rename(pipe['Cell'])
            HL_list.append(HL)

        HL_array = pd.concat(HL_list, axis = 1)

        #Iterate through subsets based on helper lipid used in formulations
        for lipid in HL_array.stack().unique():
            boolean_array = (HL_array == lipid).any(axis=1)
            lipid_data = training_data[boolean_array]
            for cell1 in cell_type_list:
                for cell2 in cell_type_list:
                    lipid_corr.loc[cell1, cell2] = get_spearman(lipid_data, cell1, cell2)
    
            fig, ax = plt.subplots(1,1,figsize=(3,3))    
            heatmap = sns.heatmap(round(np.abs(lipid_corr),2),vmin=-0.2, vmax=1,cmap='Blues', annot = True, annot_kws={"size": 8})
            heatmap.invert_yaxis()
            plt.gca()
            plt.title(lipid)
            plt.savefig(save + f'{lipid}_tfxn_Heatmap.svg', dpi=600, format = 'svg', transparent=True, bbox_inches='tight')
            plt.close()      
            # Save the all correlation data to csv
            with open(save + f'{lipid}_tfxn_correlation.csv', 'w', encoding = 'utf-8-sig') as file:
                lipid_corr.to_csv(file)




def plot_tfxn_dist_comp(pipeline_list, raw, save, new_order = None, pipe_order = None):

    #Plot parameters
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12

    
    #subplots
    fig, ax = plt.subplots(3,2, sharex = True, sharey='row', figsize = (3.3, 4))

    sns.set(font_scale = 1)

    #limits
    plt.xlim(0, 13)

    #tick labels
    ax[2, 0].set_xlabel('ln(RLU)')
    ax[2, 1].set_xlabel('ln(RLU)')
    ax[0, 0].set_yticks(np.arange(0, 500,100), fontsize = 8)
    ax[1, 0].set_yticks(np.arange(0, 300,50), fontsize = 8)
    ax[2, 0].set_yticks(np.arange(0, 150,50), fontsize = 8)
    #loop through subplots
    for i, ax in enumerate(ax.flatten()):

        #Get Training Data for each pipeline
        if new_order is None:
            pipeline = pipeline_list[i]
        else:
            pipeline = pipeline_list[pipe_order.index(new_order[i])]

        cell = pipeline['Cell']
        prefix = pipeline['Data_preprocessing']['prefix']
        if raw:
            data = pipeline['Data_preprocessing']['raw_data']
        else:
            data = pipeline['Data_preprocessing']['all_proc_data']
        
        RLU_floor = pipeline['Data_preprocessing']['RLU_floor']


        if i ==1:
            show_legend = True
        else:
            show_legend = False
        
        ax = sns.histplot(data=data, x=prefix + cell,
                          multiple="stack", 
                          hue="Helper_lipid", 
                          binwidth = 0.5,
                          hue_order= data.Helper_lipid.unique(), 
                          ax = ax, 
                          palette= "husl",
                          legend=show_legend,
                          line_kws={'linewidth': 2},
                          edgecolor='white') 
        
        ax.set_xticks(np.arange(0, 15, 3), fontsize = 6)


        ax.text(0.5, 0.85, cell, transform=ax.transAxes,
        fontsize=8, ha='center')

        #Add borders
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        #remove grid lines
        plt.grid(False)

        if show_legend:
            sns.move_legend(
                ax, "lower center",
                bbox_to_anchor=(-0.1, 1), 
                ncol=3, 
                title='Helper Lipid Choice', 
                frameon=False)
            plt.setp(ax.get_legend().get_texts(), fontsize='8')
            plt.setp(ax.get_legend().get_title(), fontsize='8') 

        #Vertical line to show floor used during processing
        if raw:
            ax.axvline(x=RLU_floor, color='gray', linestyle='--')
    #Save Transfection Distribution
    
    plt.savefig(save + f'tfxn_dist.svg', dpi = 600, transparent = True, bbox_inches = "tight")
    plt.close()


def feature_dist(pipeline, feature_name, raw, save, show_legend = True):
    
    # Configuration of the feature name and data source
    cell = pipeline['Cell']
    prefix = pipeline['Data_preprocessing']['prefix']
    feature = prefix + cell if feature_name == 'Transfection_Efficiency' else feature_name
    data = pipeline['Data_preprocessing']['raw_data' if raw else 'all_proc_data']

    # Plot parameters
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 20})
    sns.set_palette("husl", 6)
    sns.set(font_scale=2)

    # Creating the subplot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plotting
    ax = sns.histplot(data=data, 
                      x=feature, 
                      multiple="stack", 
                      ax = ax,
                      hue="Helper_lipid",
                      hue_order=data.Helper_lipid.unique(), 
                      palette="husl", 
                      legend=show_legend,
                      edgecolor='white')
    
    # Customize axes
    ax.set_xlabel(feature_name)

    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(colors='black', which='both')
    ax.tick_params(bottom=True, left=True)

    # # Move and format the legend
    if show_legend:
        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(.5, 1), 
            ncol=3, 
            title='Helper Lipid Choice', 
            frameon=False)
        plt.setp(ax.get_legend().get_texts(), fontsize='8')
        plt.setp(ax.get_legend().get_title(), fontsize='8') 

    # Save and close the plot
    plt.savefig(f'{save}/{feature_name}_dist.svg', dpi=600, transparent=True, bbox_inches="tight")
    plt.close()
    sns.reset_orig()
    return

def tfxn_clustering(X, Y, input_params, figure_save, cell):

    shap_pca50 = PCA(n_components=2).fit_transform(X)
    #shap_embedded = TSNE(n_components=2, perplexity=50, random_state = 0).fit_transform(shap_values.values)

    # Dimensionality Reduction: SHAP values on Transfection Efficiency
    f = plt.figure(figsize=(5,5))
    plt.scatter(shap_pca50[:,0],
            shap_pca50[:,1],
            c=Y.values,
            linewidth=0.1, alpha=1.0, edgecolor='black', cmap='YlOrBr')
    plt.title('Dimensionality Reduction: SHAP values on Transfection Efficiency in ' + cell)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    cb = plt.colorbar(label=None, aspect=40, orientation="horizontal")
    cb.set_alpha(1)
    cb.draw_all()
    cb.outline.set_linewidth(0.2)
    cb.ax.tick_params('x', length=2)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.get_title()
    cb.ax.set_title(label="Normalized Transfection Efficiency", fontsize=10, color="black", weight="bold")
    cb.set_ticks([0.1, 0.9])
    cb.set_ticklabels(['Low', 'High'])
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(), fontsize=10, color="black", weight="bold")

    plt.gca().axis("off") # no axis

    plt.savefig(figure_save + f'{cell}_clustered_tfxn.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
    plt.close()

########## MODEL SELECTION PLOTS ##################

def plot_AE_Box(pipeline, save, loop = 'test'):

    #Retreive Data
    cell = pipeline['Cell']
    
    df = get_Model_Selection_Error(pipeline=pipeline, loop= loop)

    #convert AE to percent error
    error_df = df*100

    #Save data with figure
    df.to_csv(save + f"{cell}_Boxplot_dataset.csv", index = False)


    #run one-way ANOVA
    _, p_anova = run_anova(df)


    #Run Statistical Analysis
    if p_anova < 0.05:
        print('RUNNING TUKEY')
        run_tukey(error_df, save, cell)

    #Config Plot
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    f, boxplot = plt.subplots(figsize=(4, 2.5))

    # choose color scheme
    palette = sns.color_palette("husl", 8, as_cmap=False)

    # # set boxplot style
    # boxplot = sns.set_style("white")
    # boxplot = sns.set_theme(font='Arial')

    # boxplot set up and box-whis style
    boxplot = sns.boxplot(palette=palette, 
                        data=error_df, saturation = 0.8,
                        boxprops = dict(linewidth=1.0, edgecolor='black', alpha = 0.8),
                        whiskerprops = dict(linewidth=1.0, color='black'),
                        capprops = dict(linewidth=1.0, color='black'),
                        flierprops=dict(marker="d", markerfacecolor= "black", markeredgecolor="black", 
                                        markersize =0.5, alpha=0.2),
                        medianprops=dict(color="black", linewidth=1.0, linestyle= '--'), 
                        showmeans=True,
                        meanprops=dict(marker="^", markerfacecolor="red", alpha=0.8,
                                        markeredgecolor="black", markersize=6, linewidth=0.05, zorder=10))

    # include each datapoint
    boxplot = sns.stripplot(data=error_df, marker="o", edgecolor='white', 
                            alpha=0.5, size=6, linewidth=0.3, palette='dark:black', jitter = True, zorder=0)

    # Title
    # boxplot.axes.set_title("Model Performance Ranked by Percent Error", font = "Arial",fontsize=20, color="Black", weight="bold")

    # Title - x-axis/y-axis
    boxplot.set_ylabel("Percent Error", font = "Arial", fontsize=12, color='black')
    
    boxplot.set(ylim=(-2, 90), yticks=np.arange(0,120,20))

    # x-axis rotation and text color
    boxplot.set_xticklabels(boxplot.get_xticklabels(),rotation = 30, ha = "right", color='black', fontsize=12)

    # x-axis and y-axis tick color
    boxplot.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes

    # x-axis and y-axis label color
    boxplot.axes.yaxis.label.set_color('black')
    boxplot.axes.xaxis.label.set_color('black')

    # format graph outline (color)
    boxplot.spines['left'].set_color('black')
    boxplot.spines['bottom'].set_color('black')
    boxplot.spines['right'].set_color('black')
    boxplot.spines['top'].set_color('black')

    # add tick marks on x-axis or y-axis
    boxplot.tick_params(bottom=False, left=True)
    boxplot.set_yticklabels(boxplot.get_yticklabels(), size = 12)
    plt.grid(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])


    plt.savefig(save + f"{cell} Model Selection Boxplot.svg", dpi=600, format = 'svg', transparent=True, bbox_inches = "tight")
    plt.close()


def plot_predictions(pipeline, save, pred = None, exp = None, normalized = True, model_name = None, correlations = True, correlation_pos = 0.9):
    
    ######### Hold Out Validation Pred vs Exp. Plots ########
    cell = pipeline['Cell']


    if pred is None and exp is None:
        #Get Predictions
        model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
        data = pipeline['Model_Selection']['Best_Model']['Predictions']

        experimental = data['Experimental_Transfection']
        predicted = data['Predicted_Transfection']
    else:
        experimental = exp
        predicted = pred

    
    
    if normalized:
        lower_lim = -0.05
        upper_lim = 1.15
    else:
        upper_lim = 12 #max(predicted.max(), experimental.max())*1.05
        lower_lim = 0 #min(predicted.min(), experimental.min())*1.05


    #Config Plot
    # sns.set_theme(font='Arial', font_scale= 2)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    #Plot
    fig = plt.figure(figsize=(2.5,2.5))
    cmap = sns.color_palette("hls", 8, as_cmap=False)


    reg = sns.regplot(x = experimental, 
                        y = predicted,
                        marker='.',
                        scatter_kws={"color": cmap[0], 
                                    "alpha": 0.6}, 
                        line_kws={"color": "black", "linestyle":'--'})

    #Annotate with correlations
    if correlations:
        pearsons = stats.pearsonr(predicted, experimental)
        spearman = stats.spearmanr(predicted, experimental)

        plt.annotate('Pearsons r = {:.2f}'.format(pearsons[0]), xy=(0.1, correlation_pos), xycoords='axes fraction')
        plt.annotate('Spearmans r = {:.2f}'.format(spearman[0]), xy=(0.1, correlation_pos-0.1), xycoords='axes fraction')

    #Labels
    plt.ylabel('Predicted Transfection')
    plt.xlabel('Experimental Transfection')

    if normalized:
        reg.set(xlim=(lower_lim, upper_lim), xticks=np.linspace(0,1,5), ylim=(lower_lim, upper_lim), yticks=np.linspace(0,1,5))

    else:
        reg.set(xlim=(lower_lim, upper_lim), xticks=np.linspace(0,20,11), ylim=(lower_lim, upper_lim), yticks=np.linspace(0,20,11))
    plt.xlim(lower_lim, upper_lim)
    plt.ylim(lower_lim, upper_lim)

    #Ticks
    reg.tick_params(colors='black', which='both')
    reg.tick_params(bottom=True, left=True)
    reg.axes.yaxis.label.set_color('black')
    reg.axes.xaxis.label.set_color('black')
    # reg.set_title("Hold-out Set Performance",weight="bold", fontsize = 15)

    reg.set_yticklabels(reg.get_yticklabels())
    reg.set_xticklabels(reg.get_xticklabels())
    # plt.tick_params(axis='both', which='major', labelsize=10)

    reg.spines['left'].set_color('black')
    reg.spines['bottom'].set_color('black')
    reg.spines['right'].set_visible(False)
    reg.spines['top'].set_visible(False)

    plt.grid(False)
    plt.savefig(save + f'{model_name}_{cell}_predictions.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
    plt.close()


def tabulate_refined_model_results(pipeline_list, cell_type_list, save):
    
    df_best_cell_model = pd.DataFrame(index = ['Model_Name','Retained_Features','# of Features', 'MAE', 'Spearman', 'Pearson'], columns = cell_type_list)
    for pipe in pipeline_list:
        df = pipe['Feature_Reduction']['Final_Results'].copy()
        cell = pipe['Cell']
        df_best_cell_model.at['Model_Name', cell] = pipe['Model_Selection']['Best_Model']['Model_Name']
        df_best_cell_model.at['Retained_Features',cell] = df.at[0,'Feature names']
        df_best_cell_model.at['# of Features',cell] = df.at[0,'# of Features']
        df_best_cell_model.at['MAE',cell] = df.at[0,'MAE']
        df_best_cell_model.at['Spearman', cell] =df.at[0,'Spearman']
        df_best_cell_model.at['Pearson',cell] = df.at[0,'Pearson']

    
    with open(save + "Feature_Red_Overall_Best_Results.csv", 'w', encoding = 'utf-8-sig') as f:
        df_best_cell_model.to_csv(f) 


def plot_cell_comparision(pipeline_list, save):
    #extract data
    best_AE = pd.DataFrame()
    for pipe in pipeline_list:
        #Extract best model AE for each cell
        best_AE[pipe['Cell']] = pipe['Model_Selection']['Best_Model']['Test_Absolute_Error']
    #convert data to percent error
    best_AE = best_AE*100

    #Sort DF by MAE
    sorted_col = best_AE.mean().sort_values()
    best_AE = best_AE[sorted_col.index]
    
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12


    #Plot
    fig = plt.figure(figsize=(2.5,1.7))
    # sns.set_theme(font='Arial', font_scale= 2)
    palette = sns.color_palette("Set2", 6, as_cmap=False)
    
    bar = sns.barplot(data=best_AE, palette=palette, saturation=0.5, ci = None)

    # Adding a pointplot to mark the mean without error bars
    mean_marker = sns.pointplot(data=best_AE, 
                                join=False,  # Ensures no lines are drawn between points
                                markers='o',  # Marker style, can choose other symbols
                                linestyles='',  # No line
                                color='black',  # Color for the mean marker
                                ci=None,  # Disable confidence interval
                                scale=0.75)  # Adjust the size of the markers
    
    plt.ylabel('Mean Percent Error', font = "Arial")
    bar.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
    bar.set(ylim=(0, 12), yticks=np.arange(0,15,5))
    # add tick marks on x-axis or y-axis
    bar.tick_params(bottom=True, left=True)
    # x-axis and y-axis label color
    bar.axes.yaxis.label.set_color('black')
    bar.axes.xaxis.label.set_color('black')
    # bar.set_title("Cell to Cell Comparision", weight="bold", fontsize=15)

    bar.set_yticklabels(bar.get_yticklabels())
    bar.set_xticklabels(bar.get_xticklabels(), rotation = 45, ha = 'right')
    # plt.tick_params(axis='both', which='major', labelsize=10)

    bar.spines['left'].set_color('black')
    bar.spines['bottom'].set_color('black')  
    bar.spines['right'].set_visible(False)
    bar.spines['top'].set_visible(False)      # x-axis and y-axis tick color
    x_min, x_max = bar.get_xlim()
    #plot dotted line at o
    plt.grid(False)
    plt.plot([x_min, x_max], [0,0], '--r')
    plt.savefig(save + f'Cell_wise_model_Comparision.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
    plt.close()

def tabulate_model_selection_results(pipeline_list,save):
    ##########  Collect all Results ###############

    cell_type_list = []
    all_model_list = []
    list_model_list = []
    for pipe in pipeline_list:
        cell_type_list.append(pipe['Cell'])
        list_model_list.append(pipe['Model_Selection']['Model_list'])
    
    # Flatten the list of lists using chain
    flattened_list = list(chain(*list_model_list))

    # Get unique values using set
    unique_values = set(flattened_list)

    # Convert the set back to a list if needed
    all_model_list = list(unique_values)

    #initialize dataframes
    df_MAE = pd.DataFrame(index = all_model_list, columns = cell_type_list)
    df_AE = pd.DataFrame(columns = cell_type_list)
    df_spearman = pd.DataFrame(index = all_model_list, columns = cell_type_list)
    df_pearson = pd.DataFrame(index = all_model_list, columns = cell_type_list)

    df_best_cell_model = pd.DataFrame(index = ['Model_Name','Hyper_Params', 'n_samples', 'MAE', 'MAE_std', 'Spearman', 'Pearson'], columns = cell_type_list)
    

    for pipe in pipeline_list:
        cell = pipe['Cell']
        model_list = pipe['Model_Selection']['Model_list']
        N_CV = pipe['Model_Selection']['N_CV']
        model_path = pipe['Saving']['Models']

        best_model_name = pipe['Model_Selection']['Best_Model']['Model_Name']

        

        for model_name in model_list:
            
            #Combine the predictions and experimental transfection data into a single row
            data = extraction_all(pipeline=pipe, model_name=model_name, loop = 'test')
            predicted    = data['Predicted_Transfection']
            experimental = data['Experimental_Transfection']
            AE           = data['Absolute_Error']

            #calculate
            pearson = stats.pearsonr(predicted, experimental)[0]
            spearman = stats.spearmanr(predicted, experimental)[0]
            MAE = AE.mean()
            MAE_std = AE.std()
            n_samples = len(AE)

            #update DF
            df_pearson.at[model_name, cell] = pearson
            df_spearman.at[model_name, cell] = spearman
            df_MAE.at[model_name, cell] = MAE

            #Update best model table
            if model_name == best_model_name:
                df_best_cell_model.at['Model_Name', cell] = model_name
                df_best_cell_model.at['n_samples', cell] = n_samples
                df_best_cell_model.at['MAE', cell] = MAE
                df_best_cell_model.at['MAE_std', cell] = MAE_std
                df_best_cell_model.at['Spearman', cell] =spearman
                df_best_cell_model.at['Pearson', cell] =pearson
                df_best_cell_model.at['Hyper_Params', cell] = pipe['Model_Selection']['Best_Model']['Hyper_Params']

                df_AE[cell] = AE

    #Save the AE results for additional plotting (outside of script)

    with open(save + f"Best_Cell_and_Model_AE.csv", 'w', encoding = 'utf-8-sig', newline='') as f:
        df_AE.to_csv(f)
    ########## Tabulate Results ##################
    with open(save + "Model_Selection_MAE.csv", 'w', encoding = 'utf-8-sig',newline='') as f:
        df_MAE.to_csv(f)
    with open(save + "Model_Selection_spearman.csv", 'w', encoding = 'utf-8-sig',newline='') as f:
        df_spearman.to_csv(f)
    with open(save + "Model_Selection_pearson.csv", 'w', encoding = 'utf-8-sig',newline='') as f:
        df_pearson.to_csv(f)   
    with open(save + "Model_Selection_Overall_Best_Results.csv", 'w', encoding = 'utf-8-sig',newline='') as f:
        df_best_cell_model.to_csv(f)   
    
    print('\nSaved All Model Selection Tables')

    return df_MAE, df_spearman, df_pearson


########## FEATURE REDUCTION PLOTS ##################
def plot_feature_reduction(pipeline):
    # Config
    stats_df = pipeline['Feature_Reduction']['Reduction_Results']
    cell_type = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    save = pipeline['Saving']['Figures']

    # Adjust MAE into percent error
    stats_df['Error'] = stats_df['MAE'] * 100
    stats_df['Error_std'] = stats_df['MAE_std'] * 100

    # Plot configurations
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12})

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(2.5, 1.5), facecolor='white')

    ax2 = ax1.twinx()

    # Define plotting style elements
    markers = {'Error': 'o', 'Spearman': 'v', 'Pearson': '^'}
    colors = sns.color_palette("husl", 5)
    marker_size = 5
    line_width = 3
    cap_size = 2

    # Plot Error
    ax1.errorbar(stats_df['# of Features'], stats_df['Error'], yerr=stats_df['Error_std'], fmt=markers['Error'], 
                  markersize=marker_size, color=colors[0], ecolor='darkgray', elinewidth=line_width, 
                  capsize=cap_size, capthick=line_width, alpha=0.6)
    ax1.plot(stats_df['# of Features'], stats_df['Error'], color=colors[0], marker=markers['Error'], 
             markersize=marker_size, markerfacecolor='black', markeredgecolor='black', alpha=0.8, 
             label='Percent Error', linewidth=line_width)

    # Plot Spearman and Pearson
    for metric, color, marker in zip(['Spearman', 'Pearson'], [colors[2], colors[4]], [markers['Spearman'], markers['Pearson']]):
        ax2.errorbar(stats_df['# of Features'], stats_df[metric], yerr=stats_df[metric + '_std'], fmt=marker, 
                     markersize=marker_size, color=color, ecolor='darkgray', elinewidth=line_width, 
                     capsize=cap_size, capthick=line_width, alpha=0.6)
        ax2.plot(stats_df['# of Features'], stats_df[metric], color=color, marker=marker, 
                 markersize=marker_size, markerfacecolor='black', markeredgecolor='black', alpha=0.8, 
                 label=metric, linewidth=line_width)

    # Labels and axis settings
    ax1.set_xlabel('Number of Remaining Features')
    ax1.set_ylabel('Percent Error')
    ax2.set_ylabel('Correlation')
    ax1.invert_xaxis()

    # Set y-axis limits
    ax1.set_ylim(0, 20)
    ax2.set_ylim(0.6, 1)

    # Customize ticks and spines
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
    
    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize='small', loc='lower left', bbox_to_anchor=(-0.3, 1.00), 
               ncol=3, columnspacing=1, handletextpad=0.2, framealpha=0)

    plt.grid(visible=False)
    plt.savefig(save + f'{cell_type}_{model_name}_Feature_Reduction_Plot.svg', dpi=600, transparent=True, bbox_inches='tight')
    plt.close()


def perform_t_tests(df, label_column, value_column, control_name, test_type='one-way', p_value_threshold=0.05, output_excel='t_test_results.xlsx'):
    # Retrieve control group values
    control_values = df[df[label_column] == control_name][value_column]
    results = {}
    statistics = []

    # Append statistics for the control group
    statistics.append({
        'Group': control_name,
        'Average Value': control_values.mean(),
        'Standard Deviation': control_values.std(),
        'P-Value': np.nan  # NaN because it does not compete against itself
    })

    # Process other groups and compare against control
    for group in df[label_column].unique():
        if group != control_name:
            group_values = df[df[label_column] == group][value_column]
            t_stat, p_value = stats.ttest_ind(control_values, group_values, nan_policy='omit')

            # Adjust p-value based on the specified test type
            if test_type == 'one-way':
                p_value = p_value / 2 if t_stat < 0 else 1 - (p_value / 2)

            # Collect results and statistics
            results[group] = {'p_value': p_value, 'is_significant': p_value < p_value_threshold}
            statistics.append({
                'Group': group,
                'Average Value': group_values.mean(),
                'Standard Deviation': group_values.std(),
                'P-Value': p_value
            })

    # Save results to Excel
    df_stats = pd.DataFrame(statistics)
    df_stats.to_excel(output_excel, index=False)

    return results

def plot_bar_with_t_test(df, plot_save, t_test_save, feature_order, label_column, value_column, palette='husl', annotate = True, p_value_threshold=0.05, test_type = 'one-way'):

    # Convert list values in the label column to strings, rename empty lists to 'Control'
    df[label_column] = df[label_column].apply(lambda x: 'Control' if isinstance(x, list) and not x else ', '.join(x) if isinstance(x, list) else x)

    # Perform one-way T-tests against the 'Control'
    results = perform_t_tests(df, label_column, value_column, 'Control', test_type, p_value_threshold, t_test_save)

    # Filter the feature_order to include only items present in the DataFrame
    filtered_feature_order = [feature for feature in feature_order if feature in df[label_column].unique()]

    # Append any features found in the DataFrame but not in the filtered feature_order
    unique_features = df[label_column].unique()
    missing_features = [feature for feature in unique_features if feature not in filtered_feature_order]
    final_feature_order = filtered_feature_order + missing_features

    # Create the bar plot with the final feature order
    fig = plt.figure(figsize=(4, 3))
    ax = sns.barplot(data=df, x=label_column, y=value_column, order=final_feature_order, errorbar='sd', capsize=.15, palette=palette)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    sns.stripplot(data=df, x=label_column, y=value_column, order=final_feature_order, color="black", jitter=True, size=6, alpha=0.6)

    plt.ylim(0, 0.18)

    # Statistical Annotation (Student T test)
    if annotate:
        annotations = annotate_barplot_with_pvalues(ax, df, results, label_column, value_column, p_value_threshold)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.xaxis.set_tick_params(which='both', direction='out')
    ax.yaxis.set_tick_params(which='both', direction='out')


    plt.savefig(plot_save, dpi=600, format='svg', transparent=True, bbox_inches='tight')
    plt.close()

def annotate_barplot_with_pvalues(ax, df, results, x_col, y_col, threshold):
    annotation_buffer = .05
    unique_features = df[x_col].unique().tolist()
    y_max = df[y_col].max()

    for group, result in results.items():
        p_value = result['p_value']
        if p_value < threshold:
            x1 = unique_features.index('Control')
            x2 = unique_features.index(group)
            y = y_max + annotation_buffer
            star = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
            ax.plot([x1, x1, x2, x2], [y - 0.02, y, y, y - 0.02], lw=1.5, c='black')
            ax.text((x1 + x2) * 0.5, y, star, ha='center', va='bottom')
            annotation_buffer += 0.05

    # Adding legend for significance levels
    handles, labels = ax.get_legend_handles_labels()
    labels += ['* p < 0.05', '** p < 0.01', '*** p < 0.001']
    handles += [plt.Line2D([], [], color='black', marker='_', linestyle='None')]*3
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)


############ LEARNING CURVE ##################
def get_learning_curve(pipeline, refined = False, NUM_ITER =5, num_splits =5, num_sizes= 50):

    start_time = time.time()
    #Initialize
    pipeline['Learning_Curve'] = {'NUM_ITER': NUM_ITER,
                        'num_splits': num_splits,
                        'num_sizes': num_sizes,
                        'Train_Error': None,
                        'Valid_Error': None
                        }
    #Config
    save_path = pipeline['Saving']['Figures']
    trained_model = pipeline['Model_Selection']['Best_Model']['Model']

    #Whether to used feature refined dataset or not
    if refined:
        X = pipeline['Feature_Reduction']['Refined_X']
        pipeline['Learning_Curve']['Dataset_Type']  = 'Refined'
    else:
        X = pipeline['Data_preprocessing']['X']
        pipeline['Learning_Curve']['Dataset_Type']  = 'All'


    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    c = pipeline['Cell']
    Y = pipeline['Data_preprocessing']['y']


    #Copy Trained Model for learning curve
    model = deepcopy(trained_model)


    #initialize training sizes
    train_size= np.linspace(0.005, 1, num_sizes)*len(X)*(num_splits-1)/num_splits
    train_size = np.floor(train_size).astype(int)
    train_scores_mean = pd.DataFrame(index=train_size)
    validation_scores_mean = pd.DataFrame(index=train_size)
    
    print(f"\n############ Calculating Learning Curve: {model_name}_{c}############ ")
    #Train model and record performance
    for i in range(NUM_ITER):
        cross_val = KFold(n_splits= num_splits, random_state= i+10, shuffle=True)
        train_sizes, train_scores, validation_scores = learning_curve(estimator = model, 
                                                                        X = X, 
                                                                        y = np.ravel(Y), 
                                                                        cv = cross_val, 
                                                                        train_sizes= train_size,
                                                                        scoring = 'neg_mean_absolute_error', shuffle= True,
                                                                        random_state = 42,
                                                                        n_jobs= -1)

        train_scores_mean[i] = -train_scores.mean(axis = 1)
        validation_scores_mean[i] = -validation_scores.mean(axis = 1)
        
    #Calculate overall results
    train_scores_mean['Train_size'] = train_scores_mean.index
    train_scores_mean["Score_Type"] = "Train"
    train_scores_mean["Mean_MAE"] = train_scores_mean.iloc[:,:NUM_ITER].mean(axis=1)
    train_scores_mean["sd"] = train_scores_mean.iloc[:,:NUM_ITER].std(axis=1)
    train_scores_mean['Percent_Error'] = train_scores_mean['Mean_MAE'] * 100
    train_scores_mean['Percent_SD'] = train_scores_mean['sd'] * 100   

    validation_scores_mean['Train_size'] = validation_scores_mean.index
    validation_scores_mean["Score_Type"] = "Validation"
    validation_scores_mean["Mean_MAE"] = validation_scores_mean.iloc[:,:NUM_ITER].mean(axis=1)
    validation_scores_mean["sd"] = validation_scores_mean.iloc[:,:NUM_ITER].std(axis=1)
    validation_scores_mean['Percent_Error'] = validation_scores_mean['Mean_MAE'] * 100
    validation_scores_mean['Percent_SD'] = validation_scores_mean['sd'] * 100  

    #Save Data
    with open(save_path + f'{model_name}_{c}_training_results.csv', 'w', encoding = 'utf-8-sig') as f:
        train_scores_mean.to_csv(f)

    with open(save_path + f'{model_name}_{c}_val_results.csv', 'w', encoding = 'utf-8-sig') as f:
        validation_scores_mean.to_csv(f)

    #Update Pipeline
    pipeline['Learning_Curve']['Model_used']  = trained_model
    pipeline['Learning_Curve']['Train_Error'] = train_scores_mean
    pipeline['Learning_Curve']['Valid_Error'] = validation_scores_mean
    pipeline['STEPS_COMPLETED']['Learning_Curve'] = True

    print('\n######## Learning_Curve Results Saved')
    print("\n\n--- %s minutes for Learning Curve---" % ((time.time() - start_time)/60))  
    return pipeline
    
def plot_learning_curve(pipeline):
    print('Plotting Learning Curve')
    #Extract Results
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    save = pipeline['Saving']['Figures']

    mean_train_scores = pipeline['Learning_Curve']['Train_Error']
    mean_validation_scores = pipeline['Learning_Curve']['Valid_Error']
    train_valid_scores = pd.concat([mean_validation_scores, mean_train_scores], 
                                   ignore_index= True)

    #Plotting
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize = (2.5, 1.5))
    # sns.set_theme(font='Arial', font_scale= 1)

    
    line = sns.lineplot(data =  train_valid_scores,
                            x = "Train_size", 
                            y = 'Percent_Error', 
                            hue = "Score_Type", 
                            errorbar = "sd", 
                            linewidth = 3, 
                            palette=sns.color_palette("Set2" , 2))

    line.set(xlim=(0, 900), xticks=np.linspace(0,900,6), ylim=(0, 20), yticks=np.arange(0,25,5))
    line.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
    # add tick marks on x-axis or y-axis
    line.tick_params(bottom=True, left=True)
    # x-axis and y-axis label color
    line.axes.yaxis.label.set_color('black')
    line.axes.xaxis.label.set_color('black')


    line.set_yticklabels(line.get_yticklabels())
    line.set_xticklabels(line.get_xticklabels())
    # plt.tick_params(axis='both', which='major', labelsize=10)

    line.spines['left'].set_color('black')
    line.spines['bottom'].set_color('black')        # x-axis and y-axis spines
    line.spines['right'].set_visible(False)
    line.spines['top'].set_visible(False)

    line.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
    # line.set_title("Learning Curve",weight="bold", fontsize=15)
    plt.grid(visible = False)
    plt.xlabel('Training size')
    plt.ylabel('Percent Error')
    plt.legend(fontsize = 'small', loc='upper right', framealpha = 0)
    plt.savefig(save + f'{model_name}_{cell}_learning_curve.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
    plt.close()


#################### SHAP PLOTS #########################
def plot_summary(pipeline, param_type, cmap, save, feature_order = None):

    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    shap_values = pipeline['SHAP'][param_type]['SHAP_Values']
    X = pipeline['SHAP'][param_type]['X']

    #Plotting
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12


    #Plot Beeswarm
    fig, ax1 = plt.subplots(1, 1, figsize = (5.5, 5))    

    if feature_order is not None:

        input_params = pipeline['SHAP'][param_type]['Input_Params']
        result_order = [item for item in feature_order if item in input_params]
        col2num = {col: i for i, col in enumerate(input_params)}
        final_order = list(map(col2num.get, result_order))

        shap.plots.beeswarm(shap_values, 
                            max_display=15,
                            show=False,
                            color_bar=False, 
                            order=final_order, 
                            color=plt.get_cmap(cmap))
    else:
        shap.plots.beeswarm(shap_values, 
                            max_display=15,
                            show=False,
                            color_bar=False, 
                            color=plt.get_cmap(cmap))

    #Format Y axis
    ax1.tick_params(axis='y', labelsize=12, color = 'black')
    #Format X axis
    ax1.tick_params(axis='x', labelsize=12, color = 'black')
    ax1.set_xlabel("SHAP Value (impact on model output)", fontsize = 12)

    #Colorbar
    cbar = plt.colorbar(ax = ax1, ticks = [], aspect= 15)
    cbar.ax.text(0.5, -0.02, 'Low', fontsize = 12, transform=cbar.ax.transAxes, 
        va='top', ha='center')
    cbar.ax.text(0.5, 1.02, 'High',fontsize = 12, transform=cbar.ax.transAxes, 
        va='bottom', ha='center')
    cbar.set_label(label = "Relative Feature Value", size = 12)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_color('black')        # x-axis and y-axis spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.gcf().set_size_inches(5.2,3.5)
    plt.grid(visible=False)
    plt.savefig(save + f'Summary.svg', dpi = 600, transparent = True, bbox_inches = 'tight')   
    plt.close()


def plot_importance(pipeline, param_type,save, feature_order = None):
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    shap_values = pipeline['SHAP'][param_type]['SHAP_Values']

    #Plotting
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12  

    #Plot
    fig, ax1 = plt.subplots(1, 1, figsize = (5, 6))    

    if feature_order is not None:
        input_params = pipeline['SHAP'][param_type]['Input_Params']
        result_order = [item for item in feature_order if item in input_params]
        col2num = {col: i for i, col in enumerate(input_params)}
        final_order = list(map(col2num.get, result_order))
        shap.plots.bar(shap_values, 
                max_display=15,
                show=False,
                order=final_order)
    else:
        shap.plots.bar(shap_values, 
                max_display=15,
                show=False)
     
    #Set X axis limis
    ax1.set_xlim(xmin = 0, xmax = 0.15)

    #Format Y axis
    ax1.tick_params(axis='y')

    #Save plot
    plt.gcf().set_size_inches(4,3.5)
    plt.grid(visible=False)
    plt.savefig(save + f'importance.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
    plt.close()

def plot_interaction(pipeline,param_type, save, cmap = 'viridis_r'):
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    shap_values = pipeline['SHAP'][param_type]['SHAP_Interaction_Values']
    X = pipeline['SHAP'][param_type]['X']
 
    
    if shap_values is None:
        print('\n\n ##### ERROR: NO INTERACTION VALUES, CANNOT PLOT INTERACTION PLOT ####')
        return
    #Plotting
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12  

    #Plot
    shap.summary_plot(shap_values, 
                      X, 
                      max_display=15, 
                      show = False, 
                      color=plt.get_cmap(cmap))
    plt.gcf().set_size_inches(30, 30)
    plt.colorbar()
    #Save plot
    plt.savefig(save + f'inter_summary.svg', bbox_inches = 'tight')
    plt.close()


def plot_force(formulation, param_type,pipeline, save, cmap = 'viridis_r'):
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    X = pipeline['SHAP'][param_type]['X']   
    shap_values = pipeline['SHAP'][param_type]['SHAP_Values']
    explainer = pipeline['SHAP'][param_type]['Explainer']
    input_params = pipeline['SHAP'][param_type]['Input_Params'] 


 
    #Plotting
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12  

    #Plot
    shap.plots.force(base_value = explainer.expected_value, 
                     shap_values = shap_values.values[formulation, :],
                     features = X.iloc[formulation, :],
                     feature_names = input_params,
                     matplotlib = True,
                     plot_cmap = cmap,
                     figsize = (4,4),
                     show = False
                     )
    
    #Save plot
    plt.savefig(save + f'{formulation}_Force.png', bbox_inches = 'tight')  
    plt.close()

def plot_dependence(pipeline, param_type,feature_list, save, interaction_feature_list = None, cmap = 'viridis_r'):
    
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    shap_values = pipeline['SHAP'][param_type]['SHAP_Values']
    X = pipeline['SHAP'][param_type]['X']
    input_params = pipeline['SHAP'][param_type]['Input_Params']  

    for feature_name in feature_list:
        for interaction_feature in interaction_feature_list:
            if feature_name == interaction_feature:
                continue
            shap.dependence_plot(ind=feature_name, 
                                shap_values=shap_values.values, 
                                features=X, 
                                feature_names=input_params,
                                interaction_index = interaction_feature,
                                show =False)
            #Save plot
            if os.path.exists(f'{save}Dependence/') == False:
                os.makedirs(f'{save}Dependence/', 0o666)
            plt.savefig(f'{save}Dependence/{feature_name}_{interaction_feature}_dep.png', bbox_inches = 'tight')
            plt.close() 

#wrapper for SHAP_Clusterplot
def plot_SHAP_cluster(pipeline, param_type,save, feature_name = 'all', cmap = 'viridis_r', size = 3,  title = True,shap_values = False): 
    input_params = pipeline['SHAP'][param_type]['Input_Params']
    if feature_name == 'all':
            for feature_name in input_params:
                SHAP_clusterplot(pipeline, param_type, feature_name, cmap, size, save, title, shap_values)
    else:
        SHAP_clusterplot(pipeline, param_type, feature_name, cmap, size, save, title, shap_values)

def SHAP_clusterplot(pipeline, param_type,feature_name, cmap, size, save, title, shap_values = False):
    
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    projections = pipeline['SHAP'][param_type]['TSNE_Embedding']

    if shap_values:
        input_params = pipeline['SHAP'][param_type]['Input_Params'].copy()
        shap = pipeline['SHAP'][param_type]['SHAP_Values'].values.copy()
        shap_df = pd.DataFrame(shap, columns = input_params)
        feature_values = shap_df[feature_name].values
        min = -0.3
        max = 0.3
        

    else:
        if feature_name == 'Transfection Efficiency':
            feature_values = pipeline['SHAP'][param_type]['y'].values
            min = 0
            max = 1
        else:
            feature_values = pipeline['SHAP'][param_type]['X'].loc[:,feature_name]
            min = feature_values.min()
            max = feature_values.max()
    norm = Normalize(vmin = min, vmax = max)
    
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12

    f = plt.figure(figsize=(size*.8,size))
    plt.scatter(projections[:,0],
            projections[:,1],
            c = feature_values,
            norm = norm,
            marker = 'o',
            s = size*5,
            linewidth=0.1, 
            alpha=0.6, 
            edgecolor='black', 
            cmap=cmap)
    # plt.title(f"{title}")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    #Colorbar
    cb = plt.colorbar(label=None, aspect=40, orientation="horizontal")
    cb.set_alpha(1)
    cb.draw_all()
    cb.outline.set_linewidth(0.2)
    cb.ax.tick_params('x', length=2)
    cb.ax.xaxis.set_label_position('top')
    if shap_values:
        ticks = [min, (min + max) / 2, max]
    else:
        ticks = [min, max]
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(), fontsize=10, color="black", weight="bold")

    if title:
        cb.ax.get_title()
        cb.ax.set_title(label=f"{feature_name}", fontsize=10, color="black", weight="bold")

    plt.gca().axis("off") # no axis

    # y-axis limits and interval
    plt.gca().set(ylim=(-30, 30), yticks=np.arange(-40, 40, 10))
    plt.gca().set(xlim=(-40, 60), xticks=np.arange(-50, 70, 10))

    if shap_values:
        #Check save path
        if os.path.exists(f'{save}Embedded_SHAP/') == False:
                os.makedirs(f'{save}Embedded_SHAP/', 0o666)
        
        plt.savefig(f'{save}Embedded_SHAP/{feature_name}_SHAP_cluster.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
    else:
        if os.path.exists(f'{save}Embedded_Features/') == False:
                os.makedirs(f'{save}Embedded_Features/', 0o666)
        plt.savefig(f'{save}Embedded_Features/{feature_name}_cluster.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
    plt.close()

def plot_embedded(pipeline, param_type,save, size, feature_name = 'all', method = 'TSNE'):
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    shap_values = pipeline['SHAP'][param_type]['SHAP_Values']
    input_params = pipeline['SHAP'][param_type]['Input_Params']

    if method == 'TSNE':
        projections = pipeline['SHAP'][param_type]['TSNE_Embedding']
    else:
        projections = 'pca'

    if feature_name == 'all': #Plot all features
        for feature_name in input_params:
            shap.embedding_plot(ind = feature_name,
                                shap_values = shap_values.values,
                                feature_names = input_params,
                                method = projections,
                                # marker_size = size*5,
                                alpha = 0.5,
                                show = False)
            plt.gcf().set_size_inches(size, size)
            plt.savefig(f'{save}{feature_name}_{method}_shap.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
            plt.close()
    else: 
        shap.embedding_plot(ind = feature_name,
                    shap_values = shap_values.values,
                    feature_names = input_params,
                    method = projections,
                    alpha = 0.5,
                    show = False)
        plt.gcf().set_size_inches(size, size)
        plt.savefig(f'{save}{feature_name}_{method}_shap.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
        plt.close()


def plot_Radar(pipeline, param_type,save):
    #config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    best_values = pipeline['SHAP'][param_type]['Best_Feature_Values']
    
    fig = px.line_polar(best_values, r=f"{cell}_Norm_Feature_Value", theta="Feature", line_close=True, start_angle= 0, width=800, height=400)
    fig.update_traces(fill='toself')
    fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))
    fig.update_layout(
        font_family="Arial",
        font_color="black",
        font = dict(size=12)
        )
    fig.update_layout({
        'plot_bgcolor':'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
    fig.write_image(save + f'{model_name}_{cell}_Radar.svg')


def plot_Rose(pipeline, param_type,save):
    #config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    best_values = pipeline['SHAP'][param_type]['Best_Feature_Values']

    fig = px.bar_polar(best_values, r=f"{cell}_Norm_Feature_Value", theta="Feature",
                        width=800, height=400)
    
    fig.update_layout({
        'plot_bgcolor':'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
    fig.write_image(save + f'{model_name}_{cell}_Rose.svg')


def get_SHAP_feature_importance(features, shap_values):
    # Calculate feature importance
    shap_values_combined = np.mean(np.abs(shap_values), axis=0)

    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': shap_values_combined
    })

    return feature_importance



def heat_dot(pipeline_list, param_type, save, cmap, mk_size = 2e3, feature_order=None):
    cell_names = []
    combined_norm_best_values = pd.DataFrame()
    combined_best_values = pd.DataFrame()
    
    list_of_importances = []
    for pipe in pipeline_list:
        cell = pipe['Cell']
        cell_names.append(cell)

        # Get SHAP Feature Importance
        feature_importance = get_SHAP_feature_importance(
            pipe['SHAP'][param_type]['X'].columns,
            pipe['SHAP'][param_type]['SHAP_Values'].values
        )

        feature_importance = feature_importance.rename(columns={'Importance': cell})
        list_of_importances.append(feature_importance.set_index('Feature'))

        norm_best_values = pipe['SHAP'][param_type]['Best_Feature_Values'].loc[:, ['Feature', f'{cell}_Norm_Feature_Value']]
        best_values = pipe['SHAP'][param_type]['Best_Feature_Values'].loc[:, ['Feature', f'{cell}_Feature_Value']]
        if combined_norm_best_values.empty:
            combined_norm_best_values = norm_best_values
            combined_best_values = best_values
        else:
            combined_norm_best_values = pd.merge(combined_norm_best_values, 
                                                 norm_best_values, 
                                                 on='Feature', 
                                                 how='outer')
            combined_best_values = pd.merge(combined_best_values, 
                                            best_values, 
                                            on='Feature', 
                                            how='outer')
    # Concatenate the list of DataFrames into a single DataFrame
    combined_feature_importance = pd.concat(list_of_importances, axis=1)
    
    #sort feature order
    if feature_order == None:
        sorted_index = combined_norm_best_values.index
    else:
        sorted_index = []
        for feature in feature_order:
            if feature in combined_norm_best_values["Feature"].tolist():
                sorted_index.append(combined_norm_best_values.index[combined_norm_best_values["Feature"] == feature][0])


    combined_norm_best_values = combined_norm_best_values.set_index(combined_norm_best_values.columns[0])

    final_feature_order = [index for index in feature_order if index in combined_norm_best_values.index]
    sorted_best = combined_norm_best_values.reindex(final_feature_order)
    sorted_importance = combined_feature_importance.reindex(final_feature_order)

    importance_array = sorted_importance.to_numpy()
    best_array = sorted_best.to_numpy()

    importance_array[np.isnan(importance_array)] = 0
    best_array[np.isnan(best_array)] = 0


    sns.set_style("white")
    # sns.set_context("notebook", rc={"grid.linewidth": 0.1,
    #                                 "font_size": 12,
    #                                 'font': 'Arial'})
    fig = plt.figure(figsize=(5, 4))

    ax = scattermap(best_array, 
                    marker_size=(importance_array)*mk_size, 
                    xticklabels= cell_names, 
                    yticklabels= final_feature_order,
                    cmap=cmap,
                    cbar=False)
    
    # Create a custom colorbar axis at the top right corner
    # Get the position of the current axes
    pos = ax.get_position()

    # Create a custom colorbar axis at the top right corner
    cax = fig.add_axes([pos.x1 + 0.05, pos.y1 - 0.45, 0.05, 0.25]) # [left, bottom, width, height]

    # Add colorbar to the new axis
    cbar = plt.colorbar(ax.collections[0], cax=cax, orientation='vertical')
    # cbar.set_label('Relative Feature Value')

    # # Set the label of the colorbar
    # cbar.set_label('Custom Label', labelpad=10, fontsize=12)

    # Adjust the label position to be horizontal on top
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel('Relative \nFeature \nValue', labelpad=5)
    cbar.ax.xaxis.set_ticks_position('top')
    # Adjust the position of the colorbar to the top right corner
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    mk_size = mk_size/10
    
    # Create a dot size legend using off-axis scatter calls and legend
    size_data = importance_array
    ax.scatter(-1, -1, label=f"{np.amax(size_data):0.2f}", marker="o", c="black", s=mk_size)
    ax.scatter(-1, -1, label=f"{np.amax(size_data)*0.5:0.2f}", marker="o", c="black", s=mk_size * 0.5)
    ax.scatter(-1, -1, label=f"{np.amax(size_data)*0.05:0.2f}", 
                marker="o", c="black", s=mk_size * 0.05)
    size_legend = ax.legend(loc="upper left", bbox_to_anchor=(0.98, 0.4), frameon = False)
    size_legend.set_title('Feature \nImportance')

    plt.savefig(f'{save}{param_type}_dot_heat_plot.svg', dpi = 600, transparent = True, bbox_inches = "tight")
    plt.close()
    return


def bumpplot(pipeline_list, param_type, lw, save, feature_order = None):
    
    combined_norm_best_values = pd.DataFrame()
    combined_best_values = pd.DataFrame()
    cell_names = []


    #Extracting Data
    for pipe in pipeline_list:
        cell = pipe['Cell']
        cell_names.append(cell)
        norm_best_values = pipe['SHAP'][param_type]['Best_Feature_Values'].loc[:, ['Feature', f'{cell}_Norm_Feature_Value']]
        best_values      = pipe['SHAP'][param_type]['Best_Feature_Values'].loc[:, ['Feature', f'{cell}_Feature_Value']]
        if combined_norm_best_values.empty:
            combined_norm_best_values = norm_best_values
            combined_best_values      = best_values
        else:
            combined_norm_best_values = pd.merge(combined_norm_best_values, 
                                            norm_best_values, 
                                            on='Feature', 
                                            how='outer')
            combined_best_values = pd.merge(combined_best_values, 
                                best_values, 
                                on='Feature', 
                                how='outer')
    #Save dataset with figure
    combined_norm_best_values.to_csv(save + f'{param_type}_best_norm_feature_value.csv', index=False)
    combined_best_values.to_csv(save + f'{param_type}_best_feature_value.csv', index=False)

    #sort feature order
    if feature_order == None:
        sorted_index = combined_norm_best_values.index
    else:
        sorted_index = []
        for feature in feature_order:
            if feature in combined_norm_best_values["Feature"].tolist():
                sorted_index.append(combined_norm_best_values.index[combined_norm_best_values["Feature"] == feature][0])

    sorted_df = combined_norm_best_values.reindex(sorted_index).reset_index(drop = True)
    sorted_df = sorted_df.iloc[::-1] #Reverse so the features are ordered from top to bottom on plot
    
    
    # Plotting
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12 
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    #Define feature colors
    feature_colors = {feature: sns.color_palette("husl", n_colors=combined_norm_best_values.shape[0])[i]
                  for i, feature in enumerate(combined_norm_best_values.index)}
    


    #add tick mark positions for all parameters
    tick_locations = combined_norm_best_values.index/combined_norm_best_values.index.max()
    # print(sorted_df)
    sorted_df.insert(1,'Features', tick_locations)
    sorted_df.fillna(-0.1, inplace = True)


    #Plot line for each feature
    for i in sorted_df.index:
        x = np.arange(sorted_df.shape[1]-1)
        y = sorted_df.iloc[i,1:].values
        color = feature_colors[i]
        x, y = add_widths(x, y)
        xs = np.linspace(0, x[-1], num=1024)
        plt.plot(xs, interpolate.PchipInterpolator(x, y)(xs), color=color, linewidth=lw, alpha = 0.75)

    #Setting y-ticks to values from the 'Feature' column
    feature_names = sorted_df['Feature'].unique()
    plt.yticks(sorted_df.Features, feature_names,rotation=45, ha='right', fontsize = 12)


    #Add Feature column to cell names
    cell_names.insert(0,'Feature')


    plt.xticks(np.arange(len(cell_names)), cell_names, fontsize = 12)
    
    sns.despine(left=True, bottom=True)  # Updated to keep the left axis
    ax.set_ylim(-0.20,1.02)
    ylim = ax.get_ylim()
    ax.set_xlim(-0.1,len(cell_names)-0.5)
    plt.grid(axis = 'x', color = 'black', alpha = 0.8)
    plt.grid(axis = 'y',visible=False)


    # Adding right axis labels
    magnitude_labels = ['N/A', 'Low', 'Med', 'High']
    region_ticks = np.linspace(0,1,len(magnitude_labels)).tolist()
    region_ticks[-1] = ylim[1]
    region_ticks[0] = -0.02
    region_ticks.insert(0, ylim[0])


    magnitude_ticks = []
    for i in range(1,len(region_ticks)):
        magnitude_ticks.append((region_ticks[i]+region_ticks[i-1])/2)
    ax2 = ax.twinx() 
    ax2.set_ylabel('Relative Feature Value', color = 'black', fontsize = 15) 
    ax2.tick_params(axis ='y', labelcolor = 'black') 
    ax2.set_yticks(magnitude_ticks, magnitude_labels, fontsize = 12, rotation = 90, ha='left')
    
    ax2.set_ylim(ylim)

    magnitude_colors = sns.light_palette("seagreen", len(region_ticks)-1)
    magnitude_colors.insert(0,'lightcoral')


    # Calculate regions
    regions = [
        (region_ticks[0], region_ticks[1]),
        (region_ticks[1], region_ticks[2]),
        (region_ticks[2], region_ticks[3]),
        (region_ticks[3], region_ticks[4])
        ]

    # Color the regions
    for (start, end), color in zip(regions, magnitude_colors):
        ax2.axhspan(start, end, facecolor=color, alpha=0.25)

    plt.savefig(f'{save}{param_type}_Bump_Plot.svg', dpi = 600, transparent = True, bbox_inches = "tight")
    plt.close()
    return

# Function to add widths
def add_widths(x, y, width=0.1):
    new_x = []
    new_y = []
    for i,j in zip(x,y):
        new_x += [i-width, i, i+width]
        new_y += [j, j, j]
    return new_x, new_y



############## IV VALIDATION ##############
def plot_validation_predictions(cell, validation_set, palette, save):
    data = copy(validation_set)
    data = data.reset_index()

    #Config Plot
    # sns.set_theme(font='Arial', font_scale= 2)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    #Plot
    fig = plt.figure(figsize=(5,4))
    cmap = sns.color_palette("hls", 8, as_cmap=False)
    bar = sns.barplot(
        data=data, 
        y="Normalized_y",
        x = 'index',
        hue = 'Helper_lipid',
        palette= palette,
        alpha = 0.8,
        dodge=False)
    # Add borders to bars
    for patch in bar.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)

    #ylim
    plt.ylim(0, 1)

    #Labels
    plt.ylabel('Predicted Transfection', fontsize = 12)
    plt.xlabel('Formulation Number',fontsize = 12)
    
    #Ticks
    bar.tick_params(colors='black', which='both')
    bar.tick_params(bottom=True, left=True)
    bar.axes.yaxis.label.set_color('black')
    bar.axes.xaxis.label.set_color('black')
   
    bar.xaxis.set_major_locator(MultipleLocator(10))
    bar.set_xticklabels(bar.get_xticklabels(), fontsize = 12)
    bar.set_yticklabels(bar.get_yticklabels(), fontsize = 12)


    #spines
    bar.spines['left'].set_color('black')
    bar.spines['bottom'].set_color('black')
    bar.spines['right'].set_visible(False)
    bar.spines['top'].set_visible(False)


    #adjust legend
    legend = bar.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, 
                 frameon=False, fontsize = '12')
    plt.setp(legend.get_texts(), fontsize='12')  # for legend text
    plt.setp(legend.get_patches(), linewidth=0, width=12)  # for legend markers 


    plt.grid(False)
    plt.savefig(save, dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
    # plt.show()
    plt.close()
