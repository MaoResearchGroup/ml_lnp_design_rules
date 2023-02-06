import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def extraction_all(name, cell, model_path):
    '''
    function that extracts and compiles a results dataframe as well as an 
    absolute error array for all modesl in NESTED_CV_results pickle files
    '''
    df = pd.read_pickle(model_path + f"{name}/{cell}/{name}_HP_Tuning_Results.pkl", compression='infer', storage_options=None)
    
    list_of_dataframes = []
    
    for n in range (5): #Range corresponds to number of outerloop iterations
        dataframe = pd.DataFrame(df['Formulation_Index'][n], columns=['Formulation_Index'])
        dataframe['Experimental_Transfection'] = df['Experimental_Transfection'][n]
        dataframe['Predicted_Transfection'] = df['Predicted_Transfection'][n]
        #dataframe['Formulation_Index'] = df['Formulation_Index'][n]
        dataframe['Absolute_Error'] = abs(dataframe['Experimental_Transfection'] - dataframe['Predicted_Transfection'])
        pd_series = dataframe['Absolute_Error']
        list_of_dataframes.append(dataframe)
    
    dataframe_all = pd.concat(list_of_dataframes, axis=0, ignore_index=True)
    
    return dataframe_all






def main():
    ################ Retreive/Store Data ##############################################
    datafile_path = "Raw_Data/7_Master_Formulas.csv"
    model_path = 'Trained_Models/230204_Models/'
    save_path = "Figures/Model_Selection/"
    ################ INPUT PARAMETERS ############################################
    cell_type_names = ['HEK293','HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']

    for cell in cell_type_names:
        #Extract data for all models
        ALL_MLR = extraction_all('MLR', cell, model_path)
        ALL_lasso = extraction_all('lasso', cell, model_path)
        ALL_kNN = extraction_all('kNN', cell, model_path)
        ALL_PLS = extraction_all('PLS', cell, model_path)
        ALL_DT = extraction_all('DT', cell, model_path)
        ALL_RF = extraction_all('RF', cell, model_path)
        ALL_LGBM = extraction_all('LGBM', cell, model_path)
        ALL_XGB = extraction_all('XGB', cell, model_path)

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
        df9.to_csv(save_path + f"{cell}_Figure_1_dataset.csv")
        df9.describe()

        ############## PLOTTING
        # figure set-up - size
        f, boxplot = plt.subplots(figsize=(12, 6))

        # choose color scheme
        #palette = sns.color_palette("Paired")
        #palette = sns.color_palette("pastel")
        #palette = sns.color_palette("tab10")
        palette = sns.color_palette("CMRmap")

        # set boxplot style
        boxplot = sns.set_style("white")

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
                            meanprops=dict(marker="o", markerfacecolor="black", alpha=1.0,
                                            markeredgecolor="black", markersize=4, linewidth=0.05, zorder=10))

        # include each datapoint
        boxplot = sns.stripplot(data=df9, marker="o", edgecolor='white', 
                                alpha=0.3, size=1.5, linewidth=0.3, color='black', jitter = True, zorder=0)

        # Title
        #boxplot.axes.set_title("ML model performance ranked by mean absolute error", fontsize=18, color="white", weight="bold")

        # Title - x-axis/y-axis
        #boxplot.set_xlabel("Model index", fontsize=12)
        boxplot.set_ylabel("Absolute error (AE)", fontsize=16, color='black', 
                        weight="bold")

        # y-axis limits and interval
        boxplot.set(ylim=(-0.02, 0.8), yticks=np.arange(0,0.8,0.05))
        #sns.despine(left=False, bottom=False)

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

        #statistical annotation
        #text you want to show in italics
        # x1, x2 = 0, 1  
        # y, h, col = 1.85, 0.02, 'black'
        # plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c=col)
        # plt.text((x1+x2)*.5, y+h+0.01, '$\it{p < 0.05}$', ha='center', va='bottom', color=col, fontsize=10)

        # statistical annotation
        #text you want to show in italics
        #x1, x2 = 0, 2  
        #y, h, col = 0.925, 0.02, 'black'
        #plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.75, c=col)
        #plt.text((x1+x2)*.5, y+h+0.01, '$\it{p < 0.05}$', ha='center', va='bottom', color=col)

        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12, weight = "bold")

        plt.tight_layout()

        plt.savefig(save_path + f"{cell} Model Selection Boxplot.png", dpi=600, format = 'png', transparent=True, bbox_inches='tight')

        #plt.show()`

if __name__ == "__main__":
    main()