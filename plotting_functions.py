import pandas as pd
import numpy as np
from scipy import stats, interpolate
import seaborn as sns
import plotly.express as px
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
from utilities import get_spearman, extract_training_data, run_tukey, extraction_all
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from copy import deepcopy, copy
from sklearn.model_selection import learning_curve
import time
import shap
import pickle
from itertools import chain
from matplotlib.colors import Normalize


########## INITIAL FEATURE ANALYSIS PLOTS ##################
def tfxn_heatmap(pipeline_list, save, helper_lipid = False):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12
    sns.set(font='Arial')

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
    # ax[0, 0].set_ylim(0,300)
    # ax[1, 0].set_ylim(0,400)
    # ax[2, 0].set_ylim(0,120)
    # ax[0, 1].set_ylim(0,300)
    # ax[1, 1].set_ylim(0,400)
    # ax[2, 1].set_ylim(0, 120)

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

        # if i in [3, 4, 5]:
        #     ax.set_xlabel('ln(RLU)')


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


def feature_dist(pipeline, feature_name, raw, save):
    #Config
    cell = pipeline['Cell']
    prefix = pipeline['Data_preprocessing']['prefix']
    if feature_name == 'Transfection_Efficiency':
        feature = prefix + cell
    else:
        feature = feature_name

    if raw:
        data = pipeline['Data_preprocessing']['raw_data']
    else:
        data = pipeline['Data_preprocessing']['all_proc_data']

    #Plot parameters
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12
    sns.set_palette("husl", 6)
    sns.set(font_scale = 1)
    
    #subplots
    fig, ax = plt.subplots(1,1, figsize = (4, 3))


        
    ax = sns.histplot(data=data, x= feature,
                        multiple="stack", 
                        hue="Helper_lipid", 
                        hue_order= data.Helper_lipid.unique(), 
                        ax = ax,
                        palette= "husl", 
                        legend=True,
                        line_kws={'linewidth': 2},
                        edgecolor='white') 

    ax.set_xlabel(feature_name)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')

    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), 
        ncol=3, 
        title='Helper Lipid Choice', 
        frameon=False)
    plt.setp(ax.get_legend().get_texts(), fontsize='8')
    plt.setp(ax.get_legend().get_title(), fontsize='8') 
    

    plt.grid(visible=False)

    plt.savefig(save + f'{feature_name}_dist.svg', dpi = 600, transparent = True, bbox_inches = "tight")
    plt.close()

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

    # # y-axis limits and interval
    # plt.gca().set(ylim=(-30, 30), yticks=np.arange(-40, 40, 10))
    # plt.gca().set(xlim=(-30, 30), xticks=np.arange(-40, 40, 10))

    plt.savefig(figure_save + f'{cell}_clustered_tfxn.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
    plt.close()

########## MODEL SELECTION PLOTS ##################
def plot_AE_Box(pipeline, save):

    #Retreive Data
    cell = pipeline['Cell']
    df = pipeline['Model_Selection']['Results']['Absolute_Error']

    #convert AE to percent error
    error_df = df*100



    #Save data with figure
    df.to_csv(save + f"{cell}_Boxplot_dataset.csv", index = False)

    #Run Statistical Analysis
    run_tukey(error_df, save, cell)

    #Config Plot
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    f, boxplot = plt.subplots(figsize=(6, 3))

    # choose color scheme
    #palette = sns.color_palette("Paired")
    #palette = sns.color_palette("pastel")
    #palette = sns.color_palette("tab10")
    palette = sns.color_palette("husl", 8, as_cmap=False)

    # set boxplot style
    boxplot = sns.set_style("white")
    boxplot = sns.set_theme(font='Arial')

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
    #boxplot.set_xlabel("Model index", fontsize=12)
    boxplot.set_ylabel("Percent Error", font = "Arial", fontsize=12, color='black')
    
    boxplot.set(ylim=(-2, 105), yticks=np.arange(0,120,20))

    # x-axis rotation and text color
    boxplot.set_xticklabels(boxplot.get_xticklabels(),rotation = 0, color='black', fontsize=12)

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


def plot_predictions(pipeline, save, pred = None, exp = None, normalized = True, model_name = None, correlations = True):
    
    ######### Hold Out Validation Pred vs Exp. Plots ########
    cell = pipeline['Cell']


    if pred is None and exp is None:
        #Get Predictions
        model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
        data = pipeline['Model_Selection']['Best_Model']['Predictions']


        experimental = data['Experimental_Transfection']
        predicted = data['Predicted_Transfection']

        if normalized:
            lower_lim = -0.05
            upper_lim = 1.05
        else:
            lower_lim = None
            upper_lim = None
    else:
        experimental = exp
        predicted = pred
        if normalized:
            lower_lim = -0.05
            upper_lim = 1.15
        else:
            lower_lim = None
            upper_lim = None




    #Config Plot
    sns.set_theme(font='Arial', font_scale= 2)
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

        plt.annotate('Pearsons r = {:.2f}'.format(pearsons[0]), xy=(0.1, 0.9), xycoords='axes fraction')
        plt.annotate('Spearmans r = {:.2f}'.format(spearman[0]), xy=(0.1, 0.8), xycoords='axes fraction')

    #Labels
    plt.ylabel('Predicted Transfection', fontsize = 12)
    plt.xlabel('Experimental Transfection',fontsize = 12)

    if normalized:
        reg.set(xlim=(lower_lim, upper_lim), xticks=np.linspace(0,1,5), ylim=(lower_lim, upper_lim), yticks=np.linspace(0,1,5))
    
    #Ticks
    reg.tick_params(colors='black', which='both')
    reg.tick_params(bottom=True, left=True)
    reg.axes.yaxis.label.set_color('black')
    reg.axes.xaxis.label.set_color('black')
    # reg.set_title("Hold-out Set Performance",weight="bold", fontsize = 15)

    reg.set_yticklabels(reg.get_yticklabels(), fontsize = 12)
    reg.set_xticklabels(reg.get_xticklabels(), fontsize = 12)
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

def plot_refined_model_comparisions(pipeline_list, save):
    MAE_list = []
    cell_list = []
    error = pd.DataFrame(columns=['Cell', 'MAE', 'std'])
    for pipe in pipeline_list:

        df = pipe['Feature_Reduction']['Reduction_Results'].copy()
        N_CV = pipe['Feature_Reduction']['N_CV']
        predictions = np.array(df.iloc[-1, 9])
        experimental = np.array(df.iloc[-1, 10])

        #Calculate MAE for each repeat
        AE = abs(predictions-experimental)
        MAE = np.mean(AE, axis=1)
        MAE_list.append(MAE)
        cell_list.append(pipe['Cell'])
    
    #Convert to df
    error = pd.DataFrame(MAE_list, index = cell_list, columns=range(N_CV))

    # Calculate row-wise averages excluding the 'Category' column
    row_averages = np.mean(error, axis=1)

    # Get the sorted indices based on row averages
    sorted_ind = np.argsort(row_averages)
    sorted = error.index[sorted_ind].to_list()

    #convert to percent error
    error = error*100

    #convert to long form for plotting
    error_long = error.reset_index().melt(id_vars='index', var_name='columns', value_name='MAE')
    
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    fig = plt.subplots(1,1, figsize=(2.5,1.5))
    sns.set_theme(font='Arial', font_scale= 1)
    palette = sns.color_palette("Set2", 6, as_cmap=False)
    fontsize = 12

    bar = sns.barplot(
        data=error_long, x="index", y="MAE",
        palette= palette, order = sorted,
        alpha = 0.8,
        dodge=False,
        errorbar='sd', capsize = 0.5,
        errwidth=1.5)
    # Add borders to bars
    for patch in bar.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    sns.stripplot(
        data=error_long, x="index", y="MAE", color = 'black',
        dodge=True, alpha=0.8,
        marker = 'o', 
        edgecolor='black',
        linewidth=1,
        facecolor='none',
        legend=False,
        order = sorted,
        size = 4
        )
    plt.ylim(0,10)
    plt.gca().set_xlabel('')
    plt.legend().set_visible(False)
    plt.ylabel('Percent Error', font = "Arial", fontsize=fontsize, color = 'black')
    # add tick marks on x-axis or y-axis
    bar.tick_params(bottom=True, left=True)
    # x-axis and y-axis label color
    bar.axes.yaxis.label.set_color('black')
    bar.axes.xaxis.label.set_color('black')

    bar.set_yticklabels(bar.get_yticklabels(), size = fontsize, color = 'black')
    bar.set_xticklabels(bar.get_xticklabels(), size = fontsize, rotation = 45, ha='right', color = 'black')

    bar.spines['left'].set_color('black')
    bar.spines['bottom'].set_color('black')  
    bar.spines['right'].set_visible(False)
    bar.spines['top'].set_visible(False)      # x-axis and y-axis tick color

    plt.grid(False)
    plt.savefig(save + f'Cell_wise_Refined_model_Comparision.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
    plt.close()


def plot_cell_comparision(pipeline_list, save):
    #extract data
    best_AE = pd.DataFrame()
    for pipe in pipeline_list:
        #Extract best model AE for each cell
        best_AE[pipe['Cell']] = pipe['Model_Selection']['Results']['Absolute_Error'].iloc[:,0] 
    
    #convert data to percent error
    best_AE = best_AE*100

    #Sort DF by MAE
    sorted_col = best_AE.mean().sort_values()
    best_AE = best_AE[sorted_col.index]

    #Plot
    fig = plt.figure(figsize=(2.5,2.5))
    sns.set_theme(font='Arial', font_scale= 2)
    palette = sns.color_palette("Set2", 6, as_cmap=False)
    # sns.barplot(best_MAE)

    fontsize = 12
    # bar = sns.barplot(data = best_AE, errorbar = 'sd', palette=palette, capsize = 0.15,errwidth=0.5,saturation = 0.5)
    bar = sns.boxplot(data = best_AE, 
                      palette=palette, 
                      saturation = 0.5, 
                      fliersize = 2, 
                      boxprops=dict(alpha=0.8),
                      showmeans=True,
                      meanprops=dict(marker="^", 
                                     markerfacecolor="red", 
                                     alpha=0.8,
                                     markeredgecolor="black", 
                                     markersize=4, 
                                     linewidth=0.05, 
                                     zorder=10))
    
    plt.ylabel('Percent Error', font = "Arial", fontsize=fontsize)
    bar.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
    bar.set(ylim=(0, 60), yticks=np.arange(0,70,10))
    # add tick marks on x-axis or y-axis
    bar.tick_params(bottom=True, left=True)
    # x-axis and y-axis label color
    bar.axes.yaxis.label.set_color('black')
    bar.axes.xaxis.label.set_color('black')
    # bar.set_title("Cell to Cell Comparision", weight="bold", fontsize=15)

    bar.set_yticklabels(bar.get_yticklabels(), size = fontsize)
    bar.set_xticklabels(bar.get_xticklabels(), size = fontsize, rotation = 45, ha='right')
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
    df_spearman = pd.DataFrame(index = all_model_list, columns = cell_type_list)
    df_pearson = pd.DataFrame(index = all_model_list, columns = cell_type_list)

    df_best_cell_model = pd.DataFrame(index = ['Model_Name','Hyper_Params', 'MAE', 'Spearman', 'Pearson'], columns = cell_type_list)

    for pipe in pipeline_list:
        cell = pipe['Cell']
        model_list = pipe['Model_Selection']['Model_list']
        N_CV = pipe['Model_Selection']['N_CV']
        model_path = pipe['Saving']['Models']

        best_model_name = pipe['Model_Selection']['Best_Model']['Model_Name']

        for model_name in model_list:
            
            #Combine the predictions and experimental transfection data into a single row
            data = extraction_all(model_name, model_path, N_CV)
            predicted    = data['Predicted_Transfection']
            experimental = data['Experimental_Transfection']
            AE           = data['Absolute_Error']

            #calculate
            pearson = stats.pearsonr(predicted, experimental)[0]
            spearman = stats.spearmanr(predicted, experimental)[0]
            MAE = AE.mean()

            #update DF
            df_pearson.at[model_name, cell] = pearson
            df_spearman.at[model_name, cell] = spearman
            df_MAE.at[model_name, cell] = MAE

            #Update best model table
            if model_name == best_model_name:
                df_best_cell_model.at['Model_Name', cell] = model_name
                df_best_cell_model.at['MAE', cell] = MAE
                df_best_cell_model.at['Spearman', cell] =spearman
                df_best_cell_model.at['Pearson', cell] =pearson
                df_best_cell_model.at['Hyper_Params', cell] = pipe['Model_Selection']['Best_Model']['Hyper_Params']
    
    ########## Tabulate Results ##################
    with open(save + "Model_Selection_MAE.csv", 'w', encoding = 'utf-8-sig') as f:
        df_MAE.to_csv(f)
    with open(save + "Model_Selection_spearman.csv", 'w', encoding = 'utf-8-sig') as f:
        df_spearman.to_csv(f)
    with open(save + "Model_Selection_pearson.csv", 'w', encoding = 'utf-8-sig') as f:
        df_pearson.to_csv(f)   
    with open(save + "Model_Selection_Overall_Best_Results.csv", 'w', encoding = 'utf-8-sig') as f:
        df_best_cell_model.to_csv(f)   
    
    print('\nSaved All Model Selection Tables')

    return df_MAE, df_spearman, df_pearson


########## FEATURE REDUCTION PLOTS ##################
def plot_feature_reduction(pipeline):
    
    #Config
    stats_df= pipeline['Feature_Reduction']['Reduction_Results']
    cell_type= pipeline['Cell']
    model_name= pipeline['Model_Selection']['Best_Model']['Model_Name']
    save = pipeline['Saving']['Figures']

    #Adjust MAE into percent error
    stats_df['Error'] = stats_df['MAE']*100
    stats_df['Error_std'] = stats_df['MAE_std']*100

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(2, 1.5), facecolor='white')
    # Adjust font size and style


    ax2 = ax1.twinx()

    # Plot the points with error bars for Average MAE
    m_size = 5
    lw = 3
    cap = 2
    palette = sns.husl_palette()
    ax1.errorbar(stats_df['# of Features'], stats_df['Error'], yerr=stats_df['Error_std'], fmt='o',markersize = m_size, color='black',
                ecolor='darkgray', elinewidth=lw, capsize=cap, capthick=cap, alpha = 0.6)

    # Draw a line connecting the points for Average MAE
    ax1.plot(stats_df['# of Features'], stats_df['Error'], color=palette[0], marker = 'o', 
             markersize = m_size, markerfacecolor='black', markeredgecolor='black', alpha = 0.8, label='Error', linewidth=lw)

    # Plot error bars for Spearman correlation coefficient
    ax2.errorbar(stats_df['# of Features'], stats_df['Spearman'], yerr=stats_df['Spearman_std'], fmt='v', markersize = m_size, color='black',
                ecolor='darkgray', elinewidth=lw, capsize=cap, capthick=cap, alpha = 0.6)

    # Draw a line connecting the points for Spearman correlation coefficient
    ax2.plot(stats_df['# of Features'], stats_df['Spearman'], color=palette[2], marker = 'v',
             markersize = m_size, markerfacecolor='black', markeredgecolor='black',alpha = 0.8, label='Spearman', linewidth=lw)

    # Plot error bars for Pearson correlation coefficient
    ax2.errorbar(stats_df['# of Features'], stats_df['Pearson'], yerr=stats_df['Pearson_std'], fmt='^', markersize = m_size, color='black',
                ecolor='darkgray', elinewidth=lw, capsize=cap, capthick=cap, alpha = 0.6)

    # Draw a line connecting the points for Pearson correlation coefficient
    ax2.plot(stats_df['# of Features'], stats_df['Pearson'], color=palette[4], marker = '^',
             markersize = m_size, markerfacecolor='black', markeredgecolor='black',alpha = 0.8,label='Pearson', linewidth=lw)


    # Set labels for the x-axis and y-axes
    label_size = 12

    ax1.set_xlabel('Number of Remaining Features', fontsize = label_size, color = 'black')
    ax1.set_ylabel('Percent Error', fontsize = label_size, color = 'black')
    ax2.set_ylabel('Correlation', fontsize = label_size, color = 'black')
    # ax1.set_title("Feature Reduction", weight="bold", fontsize=15)
    # Reverse the x-axis
    ax1.invert_xaxis()

    # Set labels on x and y axis
    ax1.set_xticks(np.arange(int(stats_df['# of Features'].min()), int(stats_df['# of Features'].max()) + 1, 2), color = 'black')
    ax1.set_yticks(np.arange(0, 20, 5), color = 'black')
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize = label_size)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = label_size)

    # Set right axis y limits
    ax2.set_ylim (0.6, 1)
    ax2.set_yticks(np.linspace(0.6, 1, 3), color = 'black')
    ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
    # ax2.tick_params(axis = 'y', labelsize=label_size)
    ax2.tick_params(axis='y', which='minor', size=3, width=1)
    #ax2.tick_params(axis =which='minor', length=3, width=1)


    # Combine the legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2

    ax2.spines['left'].set_color('black')
    ax2.spines['bottom'].set_color('black') 
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_color('black')

    # Update the legend titles
    ax1.legend(lines, ['MAE', 'Spearman', 'Pearson'],
            fontsize = 'small', 
            loc='lower left', 
            bbox_to_anchor= (-0.35 , 1.00),
            ncol = 3,
            columnspacing=1, 
            handletextpad=0.2,
            framealpha = 0)

    plt.grid(visible=False)
    plt.savefig(save + f'{cell_type}_{model_name}_Feature_Reduction_Plot.svg', dpi=600, transparent = True, bbox_inches='tight')
    plt.close()

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
    model = copy.deepcopy(trained_model)


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
    plt.rcParams['font.size'] = 8

    fig, ax = plt.subplots(figsize = (2.5,2.5))
    sns.set_theme(font='Arial', font_scale= 1)

    
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


    line.set_yticklabels(line.get_yticklabels(), fontsize = 12)
    line.set_xticklabels(line.get_xticklabels(), fontsize = 12)
    # plt.tick_params(axis='both', which='major', labelsize=10)

    line.spines['left'].set_color('black')
    line.spines['bottom'].set_color('black')        # x-axis and y-axis spines
    line.spines['right'].set_visible(False)
    line.spines['top'].set_visible(False)

    line.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
    # line.set_title("Learning Curve",weight="bold", fontsize=15)
    plt.grid(visible = False)
    plt.xlabel('Training size', fontsize = 12)
    plt.ylabel('Percent Error', fontsize = 12)
    plt.legend(fontsize = 'small', loc='upper right', framealpha = 0)
    plt.savefig(save + f'{model_name}_{cell}_learning_curve.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
    plt.close()


#################### SHAP PLOTS #########################
def plot_summary(pipeline, cmap, save, feature_order = False):

    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    shap_values = pipeline['SHAP']['SHAP_Values']
    X = pipeline['SHAP']['X']

    #Plotting
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12


    #Plot Beeswarm
    fig, ax1 = plt.subplots(1, 1, figsize = (5.5, 5))    

    if feature_order is not None:

        input_params = pipeline['SHAP']['Input_Params']
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
    plt.savefig(save + f'{model_name}_{cell}_Summary.svg', dpi = 600, transparent = True, bbox_inches = 'tight')   
    plt.close()


def plot_importance(pipeline, save, feature_order = None):
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    shap_values = pipeline['SHAP']['SHAP_Values']

    #Plotting
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12  

    #Plot
    fig, ax1 = plt.subplots(1, 1, figsize = (5, 6))    

    if feature_order is not None:
        input_params = pipeline['SHAP']['Input_Params']
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
                show=False,
                order=final_order)
     
    #Set X axis limis
    ax1.set_xlim(xmin = 0, xmax = 0.15)

    #Format Y axis
    ax1.tick_params(axis='y')

    #Save plot
    plt.gcf().set_size_inches(4,3.5)
    plt.grid(visible=False)
    plt.savefig(save + f'{model_name}_{cell}_Bar.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
    plt.close()

def plot_interaction(pipeline, save, cmap = 'viridis_r'):
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    shap_values = pipeline['SHAP']['SHAP_Interaction_Values']
    X = pipeline['SHAP']['X']
 
    
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
    # f = plt.gcf()
    plt.gcf().set_size_inches(30, 30)
    plt.colorbar()
    #Save plot
    plt.savefig(save + f'{model_name}_{cell}_inter_summary.svg', bbox_inches = 'tight')
    plt.close()

#Fix
def plot_force(formulation, pipeline, save, cmap = 'viridis_r'):
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    X = pipeline['SHAP']['X']   
    shap_values = pipeline['SHAP']['SHAP_Values']
    explainer = pipeline['SHAP']['Explainer']
    input_params = pipeline['SHAP']['Input_Params'] 


 
    #Plotting
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12  

    #Plot
    # fig, ax = plt.subplots(1, 1, figsize = (6, 8))   
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
    plt.savefig(save + f'{model_name}_{cell}_{formulation}_Force.png', bbox_inches = 'tight')  
    # plt.close()

def plot_dependence(pipeline, feature_list, save, interaction_feature_list = None, cmap = 'viridis_r'):
    
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    shap_values = pipeline['SHAP']['SHAP_Values']
    X = pipeline['SHAP']['X']
    input_params = pipeline['SHAP']['Input_Params']  

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
            plt.savefig(f'{save}Dependence/{model_name}_{cell}_{feature_name}_{interaction_feature}_dependence.png', bbox_inches = 'tight')
            plt.close() 

#wrapper for SHAP_Clusterplot
def plot_SHAP_cluster(pipeline, save, feature_name = 'all', cmap = 'viridis_r', size = 3,  title = True,shap_values = False): 
    input_params = pipeline['SHAP']['Input_Params']
    if feature_name == 'all':
            for feature_name in input_params:
                SHAP_clusterplot(pipeline, feature_name, cmap, size, save, title, shap_values)
    else:
        SHAP_clusterplot(pipeline, feature_name, cmap, size, save, title, shap_values)

def SHAP_clusterplot(pipeline, feature_name, cmap, size, save, title, shap_values = False):
    
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    projections = pipeline['SHAP']['TSNE_Embedding']

    if shap_values:
        input_params = pipeline['SHAP']['Input_Params'].copy()
        shap = pipeline['SHAP']['SHAP_Values'].values.copy()
        shap_df = pd.DataFrame(shap, columns = input_params)
        feature_values = shap_df[feature_name].values
        min = -0.3
        max = 0.3
        

    else:
        if feature_name == 'Transfection Efficiency':
            feature_values = pipeline['SHAP']['y'].values
            min = 0
            max = 1
        else:
            feature_values = pipeline['SHAP']['X'].loc[:,feature_name]
            min = feature_values.min()
            max = feature_values.max()
    norm = Normalize(vmin = min, vmax = max)
        # #Normalize
        # color_values = (feature_values-min)/(max-min)
        # norm = None  
    
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
    
    # cb.set_ticks([0, 1])
    # cb.set_ticklabels([min, max])
    

    plt.gca().axis("off") # no axis

    # y-axis limits and interval
    plt.gca().set(ylim=(-30, 30), yticks=np.arange(-40, 40, 10))
    plt.gca().set(xlim=(-40, 60), xticks=np.arange(-50, 70, 10))

    if shap_values:
        #Check save path
        if os.path.exists(f'{save}Embedded_SHAP_values/') == False:
                os.makedirs(f'{save}Embedded_SHAP_values/', 0o666)
        plt.savefig(f'{save}Embedded_SHAP_values/{model_name}_{cell}_{feature_name}_feature_SHAP_cluster.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
    else:
        if os.path.exists(f'{save}Embedded_Features/') == False:
                os.makedirs(f'{save}Embedded_Features/', 0o666)
        plt.savefig(f'{save}Embedded_Features/{model_name}_{cell}_{feature_name}_feature_cluster.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
    plt.close()

def plot_embedded(pipeline, save, size, feature_name = 'all', method = 'TSNE'):
    #Config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    shap_values = pipeline['SHAP']['SHAP_Values']
    input_params = pipeline['SHAP']['Input_Params']

    if method == 'TSNE':
        projections = pipeline['SHAP']['TSNE_Embedding']
    else:
        projections = 'pca'

    if feature_name == 'all': #Plot all features
        for feature_name in input_params:
            shap.embedding_plot(ind = feature_name,
                                shap_values = shap_values.values,
                                feature_names = input_params,
                                method = projections,
                                marker_size = size*5,
                                alpha = 0.5,
                                show = False)
            plt.gcf().set_size_inches(size, size)
            plt.savefig(f'{save}{model_name}_{cell}_{feature_name}_{method}_shap.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
            plt.close()
    else: 
        shap.embedding_plot(ind = feature_name,
                    shap_values = shap_values.values,
                    feature_names = input_params,
                    method = projections,
                    alpha = 0.5,
                    show = False)
        plt.gcf().set_size_inches(size, size)
        plt.savefig(f'{save}{model_name}_{cell}_{feature_name}_{method}_shap.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
        plt.close()


def plot_Radar(pipeline, save):
    #config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    best_values = pipeline['SHAP']['Best_Feature_Values']
    
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


def plot_Rose(pipeline, save):
    #config
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    best_values = pipeline['SHAP']['Best_Feature_Values']

    fig = px.bar_polar(best_values, r=f"{cell}_Norm_Feature_Value", theta="Feature",
                        width=800, height=400)
    
    fig.update_layout({
        'plot_bgcolor':'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
    fig.write_image(save + f'{model_name}_{cell}_Rose.svg')

def bumpplot(pipeline_list, lw, save, feature_order = None):
    
    combined_norm_best_values = pd.DataFrame()
    combined_best_values = pd.DataFrame()
    cell_names = []
    #Extracting Data
    for pipe in pipeline_list:
        cell = pipe['Cell']
        cell_names.append(cell)
        norm_best_values = pipe['SHAP']['Best_Feature_Values'].loc[:, ['Feature', f'{cell}_Norm_Feature_Value']]
        best_values      = pipe['SHAP']['Best_Feature_Values'].loc[:, ['Feature', f'{cell}_Feature_Value']]
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
    combined_norm_best_values.to_csv(save + 'best_norm_feature_value.csv', index=False)
    combined_best_values.to_csv(save + 'best_feature_value.csv', index=False)

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
    
    sorted_df.insert(1,'Features', combined_norm_best_values.index/combined_norm_best_values.index.max())
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

    #plt.title('Design Parameters for Cell Type-Specific LNP Formulations', fontsize = 20, weight = 'bold')
    
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

    plt.savefig(f'{save}Bump_Plot.svg', dpi = 600, transparent = True, bbox_inches = "tight")
    plt.close()


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
    sns.set_theme(font='Arial', font_scale= 2)
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




    # sns.move_legend(
    #                 bar, "lower center",
    #                 bbox_to_anchor=(0.5, 1), 
    #                 ncol=3,
    #                 title='Helper Lipid', 
    #                 frameon=False)

    plt.grid(False)
    plt.savefig(save, dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
    plt.show()
    plt.close()
