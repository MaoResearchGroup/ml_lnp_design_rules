# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os

def extraction_all(name, cell, model_path):
    '''
    function that extracts and compiles a results dataframe as well as an 
    absolute error array for all modesl in NESTED_CV_results pickle files
    '''
    df = pd.read_pickle(model_path + f"{cell}/{name}_{cell}_Best_Model_Results.pkl", compression='infer', storage_options=None)
    
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

def plot_best_bar(results, save_path):
    if os.path.exists(save_path) == False:
        os.makedirs(save_path, 0o666)

    results.describe()

    ############## PLOTTING
    # figure set-up - size
    f, barplot = plt.subplots(figsize=(10, 6))

    # choose color scheme
    # palette = sns.color_palette("Paired")
    #palette = sns.color_palette("pastel")
    #palette = sns.color_palette("tab10")
    #palette = sns.color_palette("CMRmap")

    # set boxplot style
    barplot = sns.set_style("white")

    # boxplot set up and box-whis style
    #barplot = sns.barplot(results, x = "Cell_Type", y = "MAE", order=results.sort_values('MAE').Cell_Type)
    barplot = plt.bar(results.Cell_Type, results.MAE, yerr = results.MAE_std)
    #Add Error Bars (Standard deviation between the 10 repeats)
    #plt.errorbar(results.Cell_Type, results.MAE, yerr = results.MAE_std, order=results.sort_values('MAE').Cell_Type)

    # # Title
    # #boxplot.axes.set_title("ML model performance ranked by mean absolute error", fontsize=18, color="white", weight="bold")

    # # Title - x-axis/y-axis
    # #boxplot.set_xlabel("Model index", fontsize=12)
    # boxplot.set_ylabel("Absolute error (AE)", fontsize=16, color='black', 
    #                 weight="bold")
    
    # boxplot.set(ylim=(-0.02, 1), yticks=np.arange(0,1,0.1))

    # # x-axis rotation and text color
    # boxplot.set_xticklabels(boxplot.get_xticklabels(),rotation = 0, color='black', fontsize=12)

    # # x-axis and y-axis tick color
    # boxplot.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes

    # # x-axis and y-axis label color
    # boxplot.axes.yaxis.label.set_color('black')
    # boxplot.axes.xaxis.label.set_color('black')

    # # format graph outline (color)
    # boxplot.spines['left'].set_color('black')
    # boxplot.spines['bottom'].set_color('black')
    # boxplot.spines['right'].set_color('black')
    # boxplot.spines['top'].set_color('black')

    # # add tick marks on x-axis or y-axis
    # boxplot.tick_params(bottom=False, left=True)

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


    plt.savefig(save_path + f"Refined Model Selection Barplot.png", dpi=600, format = 'png', transparent=True, bbox_inches='tight')

    #plt.show()

def main():

    ################ Retreive Data ##############################################
    result_folder = "Feature_Reduction/Feature_reduction_SizeZeta_PDI_45/"
    save_folder = "Figures/Refined_Model_Selection/Feature_reduction_SizeZeta_PDI_45/" 
    cell_type_list = ['HepG2','HEK293','N2a', 'ARPE19', 'B16', 'PC3']
    model_list = ['LGBM', 'XGB','RF'] 
    
    ##########  Collect all Results ###############
    all_results = pd.DataFrame(columns = ['Model', 'Cell_Type', 
                                          '# of Features', 'Feature names', 
                                          'MAE', 'MAE_std', 
                                          'Spearman', 'Spearman_std', 
                                          'Pearson', 'Pearson_std',
                                          'linkage distance'])

    best_results = pd.DataFrame(columns = ['Model', 'Cell_Type', 
                                          '# of Features', 'Feature names', 
                                          'MAE', 'MAE_std', 
                                          'Spearman', 'Spearman_std', 
                                          'Pearson', 'Pearson_std',
                                          'linkage distance'])
    for cell in cell_type_list:
        best_model_MAE = 1
        for model in model_list:
            result_file_path = result_folder + f"{cell}/{model}_{cell}_Best_Model_Results.pkl"
            with open(result_file_path, 'rb') as file:
                results = pickle.load(file)

                #Add model and cell labels
                results.at["Cell_Type", 0] = cell
                results.at["Model", 0] = model

                all_results = pd.concat([results.T, all_results.loc[:]], ignore_index = True).reset_index(drop = True)
                #print(results)
                if results.loc["MAE", 0] <= best_model_MAE:
                    best_model_MAE = results.loc["MAE", 0]
                    best_model_results = results

        #Create DF with the best model results
        best_results = pd.concat([best_model_results.T, best_results.loc[:]], ignore_index = True).reset_index(drop = True)
    #Save results
    if os.path.exists(save_folder) == False:
       os.makedirs(save_folder, 0o666)
    with open(save_folder + "Refined_Model_Selection_Results.csv", 'w', encoding = 'utf-8-sig') as f:
        all_results.to_csv(f)
    print('Saved Results')

    #Save results
    with open(save_folder + "Best_Refined_Model_Selection_Results.csv", 'w', encoding = 'utf-8-sig') as f:
        best_results.to_csv(f)
    print('Saved Best Model Results')
        

    ### PLOT MODEL SELECTION RESULTS ###########
    plot_best_bar(best_results, save_folder)



if __name__ == "__main__":
    main()