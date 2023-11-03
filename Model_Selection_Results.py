# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os
import utilities
from utilities import extraction_all


def plot_AE_Box(cell_type_names, model_path, save_path, N_CV):
    for cell in cell_type_names:
        #Extract data for all models
        ALL_MLR = extraction_all('MLR', cell, model_path,N_CV)
        ALL_lasso = extraction_all('lasso', cell, model_path,N_CV)
        ALL_kNN = extraction_all('kNN', cell, model_path,N_CV)
        ALL_PLS = extraction_all('PLS', cell, model_path,N_CV)
        ALL_DT = extraction_all('DT', cell, model_path,N_CV)
        ALL_RF = extraction_all('RF', cell, model_path,N_CV)
        ALL_LGBM = extraction_all('LGBM', cell, model_path,N_CV)
        ALL_XGB = extraction_all('XGB', cell, model_path,N_CV)

        #Append into a single dataframe
        ALL_AE = pd.DataFrame(ALL_MLR['Absolute_Error'], columns=['MLR'])
        ALL_AE['MLR'] = ALL_MLR['Absolute_Error']
        ALL_AE['lasso'] = ALL_lasso['Absolute_Error']
        ALL_AE['kNN'] = ALL_kNN['Absolute_Error']
        ALL_AE['PLS'] = ALL_PLS['Absolute_Error']
        #ALL_AE['SVR'] = ALL_SVR['Absolute_Error']
        ALL_AE['DT'] = ALL_DT['Absolute_Error']
        ALL_AE['RF'] = ALL_RF['Absolute_Error']
        ALL_AE['LGBM'] = ALL_LGBM['Absolute_Error']
        ALL_AE['XGB'] = ALL_XGB['Absolute_Error']
        #ALL_AE['NGB'] = ALL_NGB['Absolute_Error']
        #ALL_AE['NN'] = ALL_NN['Absolute_Error']
        sorted_index = ALL_AE.mean().sort_values().index
        df9=ALL_AE[sorted_index]

        if os.path.exists(save_path) == False:
            os.makedirs(save_path, 0o666)

        df9.to_csv(save_path + f"{cell}_Boxplot_dataset.csv", index = False)
        df9.describe()
        utilities.run_tukey(df9, save_path, cell)

        ############## PLOTTING
        # figure set-up - size
        f, boxplot = plt.subplots(figsize=(7, 3.5))

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
                            data=df9, saturation = 0.8,
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
        boxplot = sns.stripplot(data=df9, marker="o", edgecolor='white', 
                                alpha=0.5, size=6, linewidth=0.3, color='black', jitter = True, zorder=0)

        # Title
        boxplot.axes.set_title("Model Performance Ranked by MAE", font = "Arial",fontsize=20, color="Black", weight="bold")

        # Title - x-axis/y-axis
        #boxplot.set_xlabel("Model index", fontsize=12)
        boxplot.set_ylabel("Absolute error (AE)", font = "Arial", fontsize=16, color='black', 
                        weight="bold")
        
        boxplot.set(ylim=(-0.02, 1), yticks=np.linspace(0,1,6))

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
def plot_predictions(tuple_list, pred_transfection, exp_transfection, pearson_results, spearman_results, save_folder):
    ######### Hold Out Validation Pred vs Exp. Plots ########
    for best in tuple_list:
        #Define model and cell
        cell = best[0]
        model_name = best[1]

        fig = plt.figure(figsize=(3,3))
        predicted = pred_transfection.at[model_name, cell]
        experimental = exp_transfection.at[model_name, cell]

        sns.set_theme(font='Arial', font_scale= 2)
        reg = sns.regplot(x = experimental, y = predicted, color = "k")
        #plt.plot([0, 1], [0, 1], linestyle = 'dotted', color = 'r') #Ideal line
        plt.annotate('Pearsons r = {:.2f}'.format(pearson_results.at[model_name, cell]), xy=(0.2, 0.9), xycoords='axes fraction', fontsize=12)
        plt.annotate('Spearmans r = {:.2f}'.format(spearman_results.at[model_name, cell]), xy=(0.2, 0.8), xycoords='axes fraction', fontsize=12)
        plt.ylabel('Normalized Predicted RLU', font = "Arial", fontsize=10)
        plt.xlabel('Normalized Experimental RLU', font = "Arial", fontsize=10)
        reg.set(xlim=(-0.02, 1.02), xticks=np.linspace(0,1,5), ylim=(-0.02, 1.02), yticks=np.linspace(0,1,5))
        reg.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
        # add tick marks on x-axis or y-axis
        reg.tick_params(bottom=True, left=True)
        # x-axis and y-axis label color
        reg.axes.yaxis.label.set_color('black')
        reg.axes.xaxis.label.set_color('black')
        reg.set_title("Hold-out Set Performance",weight="bold", fontsize=12)

        reg.set_yticklabels(reg.get_yticklabels(), size = 8)
        reg.set_xticklabels(reg.get_xticklabels(), size = 8)
        # plt.tick_params(axis='both', which='major', labelsize=10)

        reg.spines['left'].set_color('black')
        reg.spines['bottom'].set_color('black')        # x-axis and y-axis tick color



        
        plt.savefig(save_folder + f'/{model_name}_{cell}_predictions.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
        plt.close()

def plot_cell_comparision(tuple_list, save_folder):
    #extract data
    best_AE = pd.DataFrame()
    for best in tuple_list:
        cell = best[0]
        data = pd.read_csv(save_folder + f"{cell}_Boxplot_dataset.csv")
        best_AE[cell] = data[data.columns[0]] #Extract first column of dataset
    
    #convert data to percent error
    best_AE = best_AE*100
    best_MAE = best_AE.mean(axis=0)
    #Plot
    fig = plt.figure(figsize=(3,3))
    sns.set_theme(font='Arial', font_scale= 2)
    palette = sns.color_palette("hls", 8, as_cmap=False)
    # sns.barplot(best_MAE)

    ##### VIOLIN PLOT
    bar = sns.barplot(data = best_AE, errorbar = 'sd', palette=palette)
    plt.ylabel('Average Percent Error', font = "Arial", fontsize=10)
    bar.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
    bar.set(ylim=(-5, 25), yticks=np.arange(-5,30,5))
    # add tick marks on x-axis or y-axis
    bar.tick_params(bottom=True, left=True)
    # x-axis and y-axis label color
    bar.axes.yaxis.label.set_color('black')
    bar.axes.xaxis.label.set_color('black')
    bar.set_title("Cell to Cell Comparision",weight="bold", fontsize=12)

    bar.set_yticklabels(bar.get_yticklabels(), size = 8)
    bar.set_xticklabels(bar.get_xticklabels(), size = 8, rotation = 45)
    # plt.tick_params(axis='both', which='major', labelsize=10)

    bar.spines['left'].set_color('black')
    bar.spines['bottom'].set_color('black')        # x-axis and y-axis tick color
    x_min, x_max = bar.get_xlim()
    #plot dotted line at o
    plt.plot([x_min, x_max], [0,0], '--r')

    plt.savefig(save_folder + f'/Cell_to_Cell_Comparision.svg', dpi=600, format = 'svg',transparent=True, bbox_inches = 'tight')
    plt.close()

def main(model_folder, figure_save_path, cell_type_list, model_list, N_CV):

    save_folder = figure_save_path
    
    ##########  Collect all Results ###############
    all_results = pd.DataFrame(columns = ['Model', 'Cell_Type', 'Valid Score', 'Test Score','Spearmans Rank','Pearsons Correlation','Model Parms', 'Experimental_Transfection','Predicted_Transfection'])

    for model in model_list:
        for cell in cell_type_list:
            result_file_path = model_folder + f'{model}/{cell}/{model}_{cell}_HP_Tuning_Results.pkl'
            # result_file_path = model_folder + f'{model}/{cell}/{model}_{cell}_Best_Model_Results.pkl'
            with open(result_file_path, 'rb') as file:
                results = pickle.load(file)
                results.drop(columns = ['Iter','Formulation_Index'], inplace = True)

                #Combine the predictions and experimental transfection data into a single row
                results = results.iloc[[0]] #keep only Best model, return dataframe type
                results.insert(0, 'Model', model) #Add model
                results.insert(1, 'Cell_Type', cell) #Add cell type
                all_results = pd.concat([results, all_results.loc[:]], ignore_index = True).reset_index(drop = True)
    
    #Save results
    if os.path.exists(model_folder) == False:
       os.makedirs(model_folder, 0o666)
    with open(model_folder + "Model_Selection_Results.csv", 'w', encoding = 'utf-8-sig') as f:
        all_results.to_csv(f)
    print('Saved Model Selection Results')
    
    ########## Extract Results for all models for all cell types##################
    MAE_results = pd.DataFrame(index = model_list, columns = cell_type_list)
    spearman_results = pd.DataFrame(index = model_list, columns = cell_type_list)
    pearson_results = pd.DataFrame(index = model_list, columns = cell_type_list)
    pred_transfection = pd.DataFrame(index = model_list, columns = cell_type_list)
    exp_transfection = pd.DataFrame(index = model_list, columns = cell_type_list)
    for model in model_list:
        for cell in cell_type_list:
            m1 = all_results["Model"] == model
            m2 = all_results["Cell_Type"] == cell
            MAE_results.at[model, cell] = all_results[m1&m2]['Test Score'].values[0]
            spearman_results.at[model, cell] = all_results[m1&m2]['Spearmans Rank'].values[0][0]
            pearson_results.at[model, cell] = all_results[m1&m2]['Pearsons Correlation'].values[0][0]
            pred_transfection.at[model, cell] = all_results[m1&m2]['Predicted_Transfection'].values[0]
            exp_transfection.at[model, cell] = all_results[m1&m2]['Experimental_Transfection'].values[0].transpose()[0] #Format as list
    
    ########## Tabulate Results ##################
    with open(model_folder + "Model_Selection_MAE.csv", 'w', encoding = 'utf-8-sig') as f:
        MAE_results.to_csv(f)
    with open(model_folder + "Model_Selection_spearman.csv", 'w', encoding = 'utf-8-sig') as f:
        spearman_results.to_csv(f)
    with open(model_folder + "Model_Selection_pearson.csv", 'w', encoding = 'utf-8-sig') as f:
        pearson_results.to_csv(f)   
    

    #### PLOT MODEL SELECTION RESULTS ###########
    if os.path.exists(save_folder) == False:
       os.makedirs(save_folder, 0o666)

    ###### Model Selection Box Plot ##############
    plot_AE_Box(cell_type_list, model_folder, save_folder, N_CV)
    
    ###### Select best cell/model pairs ##########
    cell_model = utilities.get_best_model_cell(figure_save_path = save_folder, 
                                              model_folder=model_folder,
                                              cell_type_list=cell_type_list)
    best_cell_model = list(cell_model.itertuples(index=False, name=None))


    ###### Hold-out set predictions plot ##############
    plot_predictions(tuple_list = best_cell_model,
                     pred_transfection=pred_transfection,
                     exp_transfection=exp_transfection,
                     pearson_results=pearson_results,
                     spearman_results=spearman_results,
                     save_folder=save_folder)
    

    ###### Cell-wise comparision plots ##############
    plot_cell_comparision(best_cell_model, save_folder=save_folder)
    



if __name__ == "__main__":
    main()