import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from utilities import get_spearman, extract_training_data, run_tukey
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import copy
from sklearn.model_selection import learning_curve
import time


########## INITIAL FEATURE ANALYSIS PLOTS ##################
def tfxn_heatmap(pipeline_list, save_path):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12

    cell_type_list = []
    cell_tfxn_list = []
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
    fig = plt.subplot()
    plt.figure(figsize=(3,3))
    heatmap = sns.heatmap(round(np.abs(all_corr),2), vmin=.2, vmax=1, cmap='Blues', annot = True, annot_kws={"size": 6}, cbar = False)
    heatmap.invert_yaxis()
    plt.tick_params(axis='y', which='both', labelsize=8)
    plt.tick_params(axis='x', which='both', labelsize=8)
    cbar = heatmap.figure.colorbar(heatmap.collections[0])
    cbar.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 4)
    cbar.ax.tick_params(labelsize=6)

    plt.gca()
    plt.savefig(save_path + 'All_Data_Tfxn_Heatmap.svg', dpi=600, format = 'svg', transparent=True, bbox_inches='tight')
    plt.close()           


    #### PLOTTING THE HEATMAP FOR LIPID SPECIFIC FORMULATIONS ACROSS CELL TYPES        
    # #Iterate through subsets based on helper lipid used in formulations
    # print(training_data["Helper_lipid"].unique())
    # for lipid in training_data["Helper_lipid"].unique():
    #     lipid_data = training_data.loc[training_data["Helper_lipid"] == lipid]
    #     for cell1 in cell_type_list:
    #         for cell2 in cell_type_list:
    #             lipid_corr.loc[cell1, cell2] = get_spearman(lipid_data, cell1, cell2)
  
            
    #     sns.heatmap(lipid_corr,vmin=-0.2, vmax=1, annot = True)
    #     plt.gca()
    #     plt.title(lipid)
    #     plt.savefig(save_path + f'{lipid}_Heatmap.svg', dpi=600, format = 'svg', transparent=True, bbox_inches='tight')
    #     plt.close()      
    #     # Save the all correlation data to csv
    #     with open(save_path + f'{lipid}_tfxn_correlation.csv', 'w', encoding = 'utf-8-sig') as file:
    #         lipid_corr.to_csv(file)


    # Save the all correlation data to csv
    with open(save_path + 'All_tfxn_correlation.csv', 'w', encoding = 'utf-8-sig') as file:
          all_corr.to_csv(file)

def plot_tfxn_dist_comp(pipeline_list, raw, save_path):

    #Plot parameters
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12
    sns.set_palette("husl", 6)
    sns.set(font_scale = 1)
    
    #subplots
    fig, ax = plt.subplots(2,3, sharex = True, sharey=True, figsize = (4, 3))

    #limits
    # plt.ylim(0, 215)
    # plt.xlim(0, 12)

    #loop through subplots
    for i, ax in enumerate(ax.flatten()):

        #Get Training Data for each pipeline
        pipeline = pipeline_list[i]
        cell = pipeline['Cell']
        prefix = pipeline['Data_preprocessing']['prefix']
        RLU_floor = pipeline['Data_preprocessing']['RLU_floor']
        if raw:
            data = pipeline['Data_preprocessing']['raw_data']
            RLU_floor = 0


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
                          legend=show_legend,
                          line_kws={'linewidth': 2},
                          edgecolor='white') 
        # ax.set_yticks(np.arange(0, 300,50), fontsize = 6)
        ax.set_xticks(np.arange(RLU_floor, 15, 3), fontsize = 6)

        if i in [3, 4, 5]:
            ax.set_xlabel('ln(RLU)')


        ax.text(0.5, 0.85, cell, transform=ax.transAxes,
        fontsize=8, ha='center')

        if show_legend:
            sns.move_legend(
                ax, "lower center",
                bbox_to_anchor=(.5, 1), 
                ncol=3, 
                title='Helper Lipid Choice', 
                frameon=False)
            plt.setp(ax.get_legend().get_texts(), fontsize='8')
            plt.setp(ax.get_legend().get_title(), fontsize='8') 

    #Save Transfection Distribution
    plt.savefig(save_path + f'tfxn_dist.png', dpi = 600, transparent = True, bbox_inches = "tight")
    plt.close()

def tfxn_clustering(X, Y, input_params, figure_save_path, cell):

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

    plt.savefig(figure_save_path + f'{cell}_clustered_tfxn.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
    plt.close()



########## MODEL SELECTION PLOTS ##################
def plot_AE_Box(pipeline):

    #Retreive Data
    cell = pipeline['Cell']
    df = pipeline['Model_Selection']['Results']['Absolute_Error']
    save_path = pipeline['Saving']['Figures']

    #convert AE to percent error
    error_df = df*100

    #Check save path
    if os.path.exists(save_path) == False:
        os.makedirs(save_path, 0o666)

    #Save data with figure
    df.to_csv(save_path + f"{cell}_Boxplot_dataset.csv", index = False)

    #Run Statistical Analysis
    run_tukey(error_df, save_path, cell)

    #Config Plot
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    f, boxplot = plt.subplots(figsize=(6, 3))

    # choose color scheme
    #palette = sns.color_palette("Paired")
    #palette = sns.color_palette("pastel")
    #palette = sns.color_palette("tab10")
    palette = sns.color_palette("hls", 8, as_cmap=False)

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
    boxplot.set_ylabel("Percent Error", font = "Arial", fontsize=16, color='black')
    
    boxplot.set(ylim=(-2, 1), yticks=np.arange(0,120,20))

    # x-axis rotation and text color
    boxplot.set_xticklabels(boxplot.get_xticklabels(),rotation = 0, color='black', fontsize=14)

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
    boxplot.set_yticklabels(boxplot.get_yticklabels(), size = 14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])



    plt.savefig(save_path + f"{cell} Model Selection Boxplot.svg", dpi=600, format = 'svg', transparent=True, bbox_inches = "tight")
    plt.close()

    #plt.show()

def plot_predictions(pipeline):
    
    ######### Hold Out Validation Pred vs Exp. Plots ########
    cell = pipeline['Cell']
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    data = pipeline['Model_Selection']['Best_Model']['Predictions']
    save_folder = pipeline['Saving']['Figures']

    #Get Predictions
    experimental = data['Experimental_Transfection']
    predicted = data['Predicted_Transfection']

    #Calculate correlations
    pearsons = stats.pearsonr(predicted, experimental)
    spearman = stats.spearmanr(predicted, experimental)


    #Config Plot
    sns.set_theme(font='Arial', font_scale= 2)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    #Plot
    fig = plt.figure(figsize=(2.5,2.5))
    reg = sns.regplot(x = experimental, 
                        y = predicted,
                        marker='.', 
                        scatter_kws={"color": "m", 
                                    "alpha": 0.6}, 
                        line_kws={"color": "black", "linestyle":'--'})

    #Annotate with correlations
    plt.annotate('Pearsons r = {:.2f}'.format(pearsons[0]), xy=(0.1, 0.9), xycoords='axes fraction')
    plt.annotate('Spearmans r = {:.2f}'.format(spearman[0]), xy=(0.1, 0.8), xycoords='axes fraction')


    #Labels
    plt.ylabel('Predicted Transfection', fontsize = 12)
    plt.xlabel('Experimental Transfection',fontsize = 12)
    reg.set(xlim=(-0.05, 1.05), xticks=np.linspace(0,1,5), ylim=(-0.05, 1.05), yticks=np.linspace(0,1,5))
    
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
    reg.spines['bottom'].set_color('black')        # x-axis and y-axis tick color

    plt.savefig(save_folder + f'/{model_name}_{cell}_predictions.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
    plt.close()

def plot_cell_comparision(pipeline_list, save_path):
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

    ##### VIOLIN PLOT
    fontsize = 12
    # bar = sns.barplot(data = best_AE, errorbar = 'sd', palette=palette, capsize = 0.15,errwidth=0.5,saturation = 0.5)
    bar = sns.boxplot(data = best_AE, palette=palette,saturation = 0.5, fliersize = 2)
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
    bar.spines['bottom'].set_color('black')        # x-axis and y-axis tick color
    x_min, x_max = bar.get_xlim()
    #plot dotted line at o
    plt.plot([x_min, x_max], [0,0], '--r')
    plt.savefig(save_path + f'Cell_wise_model_Comparision.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
    plt.close()

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
    ax1.plot(stats_df['# of Features'], stats_df['Error'], color=palette[0], alpha = 0.8, label='Error', linewidth=lw)

    # Plot error bars for Spearman correlation coefficient
    ax2.errorbar(stats_df['# of Features'], stats_df['Spearman'], yerr=stats_df['Spearman_std'], fmt='v', markersize = m_size, color='black',
                ecolor='darkgray', elinewidth=lw, capsize=cap, capthick=cap, alpha = 0.6)

    # Draw a line connecting the points for Spearman correlation coefficient
    ax2.plot(stats_df['# of Features'], stats_df['Spearman'], color=palette[2], alpha = 0.8, label='Spearman', linewidth=lw)

    # Plot error bars for Pearson correlation coefficient
    ax2.errorbar(stats_df['# of Features'], stats_df['Pearson'], yerr=stats_df['Pearson_std'], fmt='^', markersize = m_size, color='black',
                ecolor='darkgray', elinewidth=lw, capsize=cap, capthick=cap, alpha = 0.6)

    # Draw a line connecting the points for Pearson correlation coefficient
    ax2.plot(stats_df['# of Features'], stats_df['Pearson'], color=palette[4], alpha = 0.8,label='Pearson', linewidth=lw)


    # Set labels for the x-axis and y-axes
    label_size = 12

    ax1.set_xlabel('Number of Remaining Features')
    ax1.set_ylabel('Percent Error')
    ax2.set_ylabel('Correlation')
    # ax1.set_title("Feature Reduction", weight="bold", fontsize=15)
    # Reverse the x-axis
    ax1.invert_xaxis()

    # Set labels on x and y axis
    ax1.set_xticks(np.arange(int(stats_df['# of Features'].min()), int(stats_df['# of Features'].max()) + 1, 2))
    ax1.set_yticks(np.arange(0, 20, 5))
    ax1.set_yticklabels(ax1.get_yticklabels())
    ax1.set_xticklabels(ax1.get_xticklabels())

    # Set right axis y limits
    ax2.set_yticks(np.linspace(0.5, 1, 3))
    ax2.tick_params(axis = 'y', labelsize=label_size)


    # Combine the legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')        # x-axis and y-axis spines
    ax1.spines['right'].set_color('black')
    ax1.spines['top'].set_color('black')


    # Update the legend titles
    ax1.legend(lines, ['MAE', 'Spearman', 'Pearson'],
            fontsize = 'small', 
            loc='lower left', 
            bbox_to_anchor= (-0.35 , 1.00),
            ncol = 3,
            columnspacing=1, 
            handletextpad=0.2,
            framealpha = 0)

    
    # Save the plot as a high-resolution image (e.g., PNG or PDF)
    plt.show()
    # plt.savefig(save + f'{cell_type}_{model_name}_Feature_Reduction_Plot.svg', dpi=600, transparent = True, bbox_inches='tight')

    plt.close()

############ LEARNING CURVE ##################
def get_learning_curve(pipeline, NUM_ITER =5, num_splits =5, num_sizes= 50):

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
    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    c = pipeline['Cell']
    X = pipeline['Data_preprocessing']['X']
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
                                                                        scoring = 'neg_mean_absolute_error', shuffle= True, n_jobs= -1)

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
    pipeline['Learning_Curve']['Train_Error'] = train_scores_mean
    pipeline['Learning_Curve']['Valid_Error'] = validation_scores_mean
    print('\n######## Learning_Curve Results Saved')
    print("\n\n--- %s minutes for Learning Curve---" % ((time.time() - start_time)/60))  
    return pipeline
    
def plot_learning_curve(pipeline):
    
    #Extract Results
    print('Plotting Learning Curve')
    mean_train_scores = pipeline['Learning_Curve']['Train_Error']
    mean_validation_scores = pipeline['Learning_Curve']['Valid_Error']
    train_valid_scores = pd.concat([mean_validation_scores, mean_train_scores], 
                                   ignore_index= True)

    #Plotting
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize = (2.5,1.5))
    sns.set_theme(font='Arial', font_scale= 1)

    
    line = sns.lineplot(data =  train_valid_scores,
                            x = "Train_size", 
                            y = 'Percent_Error', 
                            hue = "Score_Type", 
                            errorbar = "sd", 
                            linewidth = 3, 
                            palette=sns.color_palette("Set2" , 2))

    line.set(xlim=(0, 900), xticks=np.linspace(0,900,6), ylim=(0, 15), yticks=np.arange(0,20,5))
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

    plt.xlabel('Training size')
    plt.ylabel('Percent Error')
    plt.legend(fontsize = 'small', loc='upper right', framealpha = 0)
    # plt.savefig(save_path + f'{model_name}_{c}_learning_curve.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')