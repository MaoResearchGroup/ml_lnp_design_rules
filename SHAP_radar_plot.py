import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
import pickle
import pandas as pd
import seaborn as sns
import os



import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = "notebook"

import plotly.express as px
import plotly.graph_objects as go

def mean_shap_bar(data, xaxis, yaxis):
    fig = plt.figure()
    sns.barplot(data, x = xaxis, y = yaxis)
    fig.gca()

    return fig 


def plot_scatter(f, data, xaxis, yaxis):
    fig= plt.figure()
    sns.scatterplot(data, x = xaxis, y = yaxis).set(title = f)
    sns.lineplot(data, x = xaxis, y = yaxis, color='red')
    fig.gca()
    return fig 
def plot_scatter_line(f, scatter_data, x1axis, y1axis, mean_data, x2axis, y2axis ):
    fig= plt.figure()
    sns.scatterplot(scatter_data, x = x1axis, y = y1axis).set(title = f)
    sns.lineplot(mean_data, x = x2axis, y = y2axis, color='red')
    fig.gca()
    return fig 


def plot_line(data, xaxis, yaxis):
    fig = plt.figure()
    #sns.scatterplot(data, x = xaxis, y = yaxis)
    sns.lineplot(data, x = xaxis, y = yaxis)
    fig.gca()
    plt.show()
    return fig 

def plot_hist(X, y, feature, N_bins):
    fig = plt.figure()
    y.rename("SHAP Value", inplace = True)
    datadf = pd.concat([X,y], axis = 1)
    print(datadf)
    bin_size = (X.max()-X.min())/N_bins
    sns.histplot(datadf, x = X.name, y = "SHAP Value", binwidth = bin_size)
    fig.gca()
def create_bins(lower_bound, upper_bound, quantity):

    bins = []
    boundaries = np.linspace(lower_bound, upper_bound, quantity)
    for i in range(len(boundaries)-1):
        bins.append((boundaries[i],boundaries[i+1] ))
    return bins

def find_bin(value, bins):
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1


def plot_Radar(data):
    data["Norm_Feature_Value"] = data["Norm_Feature_Value"]
    fig = px.line_polar(data, r="Norm_Feature_Value", theta="Feature", line_close=True, start_angle= 0, width=800, height=400)
    fig.update_traces(fill='toself')
    # # Defining the title of the plot
    # plt.title("Radar Chart")
    fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))
    fig.update_layout(
        font_family="Arial",
        font_color="black",
        font = dict(size=20)
        )
    fig.update_layout({
        'plot_bgcolor':'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        
    return fig

def plot_Rose(data):
    data["Norm_Feature_Value"] = data["Norm_Feature_Value"]
    ### pxplot
    fig = px.bar_polar(data, r="Norm_Feature_Value", theta="Feature",
                        color="Type", width=800, height=400)
    fig.update_layout({
        'plot_bgcolor':'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        

    # fig.update_layout(
    #     showlegend = False,
    #     polar = dict(
    #     sector = [0,270],
    #     ))

    return fig


def main(model_list, cell_type_list, model_save_path, shap_value_path, figure_save_path, N_bins, comp_features, lipid_features, phys_features):
    ################ Retreive/Store Data ##############################################
    # RUN_NAME = "Feature_reduction_Size_600_Zeta_PDI_0.45"
    # save_path = f"Figures/Radar/{RUN_NAME}" +"_Test/"
    # model_folder = f"Feature_Reduction/{RUN_NAME}/"
    # shap_value_path = f'SHAP_Values/{RUN_NAME}/'
    
    # cell_type = ['ARPE19','N2a','PC3','B16','HEK293','HepG2']

    # model_list = ['LGBM', 'XGB','RF']

    # # #testing
    # # cell_type = ['B16']
    # # model_list = ['LGBM']

    # # comp_features = ['NP_ratio', 
    # #                 'Dlin-MC3_Helper lipid_ratio',
    # #                 'Dlin-MC3+Helper lipid percentage', 
    # #                 'Chol_DMG-PEG_ratio']
    # comp_features = ['NP_ratio', 
    # #                'Dlin-MC3_Helper lipid_ratio',
    #                 'Dlin-MC3+Helper lipid percentage', 
    #                 'Chol_DMG-PEG_ratio']
    

    # lipid_features = ['P_charged_centers', 
    # #                  'N_charged_centers', 
    #                   'Double_bonds',
    #                   'cLogP', ]
    

    # phys_features = ['Size', 'Zeta']

    features = comp_features + lipid_features + phys_features
    ################ INPUT PARAMETERS ############################################
    #cell_type_names = ['HEK293','HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']

    N_bins = 10

    for c in cell_type_list:
        for model in model_list:

            #Check save paths          
            if os.path.exists(figure_save_path + f'/{c}/{model}/') == False:
                os.makedirs(figure_save_path + f'/{c}/{model}/', 0o666)

            #Get feature names used to train model
            with open(model_save_path + f"{c}/{model}_{c}_Best_Model_Results.pkl", 'rb') as file: # import best model results
                        best_results = pickle.load(file)
            input_param_names = best_results.loc['Feature names'][0]  
  
            mean_shap = pd.DataFrame(columns = input_param_names)
            #Get SHAP Values
            with open(shap_value_path + f"{model}_{c}_SHAP_values.pkl", "rb") as file:   # Unpickling
                shap_values = pickle.load(file)

            #Dataframe of input data to generate SHAP values
            df_input_data = pd.DataFrame(shap_values.data, columns = input_param_names)

            #Dataframe of shap values
            df_values = pd.DataFrame(shap_values.values, columns = input_param_names)

            #dataframe to store average SHAP values
            mean_storage = pd.DataFrame(columns = ["Feature", "Feature_Value_bin", "Feature_Value", "Avg_SHAP"])
            
            #List to store best feature values
            best_feature_values = []

            for f in features:
                #print(f"****************  {f}  ***********")
                #Iterate through all feature names to find unique feature values for composition or helper lipid parameters
                if f in comp_features:
                    f_type = "comp"
                elif f in lipid_features:
                    f_type = "lipid"
                elif f in phys_features:
                    f_type = "physio"
                else:
                    f_type = "N/A"
                if f in input_param_names:
                    combined = pd.DataFrame()
                    combined['Input'] = df_input_data[f]
                    combined["SHAP"] = df_values[f]
                    #check if a physiochemical feature (continous)
                    if f in phys_features:
                        #bins
                        feature_bins = create_bins(int(np.floor(combined['Input'].min())),int(np.ceil(combined['Input'].max())), N_bins)
                        print(feature_bins)
                        bin_means = []
                        for bin in feature_bins:
                            bin_means.append(np.mean(bin))

                        binned_inputs = []
                        for value in combined['Input']:
                            bin_index = find_bin(value, feature_bins)
                            binned_inputs.append(bin_index)
                        combined['bins'] = binned_inputs
                        unique_bins = combined['bins'].unique()
                        #Iterate through unique feature values to get average SHAP for that feature value
                        for bins in unique_bins:
                            #Get the mean shap value for the unique feature value
                            bin_mean = combined.loc[combined['bins'] == bins, 'SHAP'].mean()
                            #Store the mean shap value with the mean bin value
                            mean_storage.loc[len(mean_storage)] = [f, feature_bins[bins], np.mean(feature_bins[bins]), bin_mean]
                        #Create x,y plot of the feature value and 
                        scatter = plot_scatter_line(f, combined,'Input', "SHAP", mean_storage.loc[mean_storage["Feature"] == f, ["Feature_Value", "Avg_SHAP"]],
                                                    "Feature_Value", 
                                                    "Avg_SHAP")
                        scatter.savefig(figure_save_path + f'/{c}/{model}/{model}_{c}_{f}_scatter.png', bbox_inches = 'tight')
                        plt.close()


                    #composition or helper lipid features which are more categorical
                    else:
                        # #Create x,y plot of the feature value and 
                        # line = plot_line(combined,'Input',"SHAP")
                        unique_feature_values = df_input_data[f].unique()
                        #Iterate through unique feature values to get average SHAP for that feature value
                        for feature_value in unique_feature_values:
                            #Get the mean shap value for the unique feature value
                            feature_mean = df_values.loc[df_input_data[f] == feature_value, f].mean()

                            #Store the mean shap value
                            mean_storage.loc[len(mean_storage)] = [f, feature_value, feature_value, feature_mean]
                        
                        scatter = plot_scatter(f, combined,'Input', "SHAP")
                        scatter.savefig(figure_save_path + f'/{c}/{model}/{model}_{c}_{f}_scatter.png', bbox_inches = 'tight')
                        plt.close()




                    #Create a histogram of the avg_shap value for each feature value
                    feature_bar = mean_shap_bar(mean_storage.loc[mean_storage["Feature"] == f, ["Feature_Value", "Avg_SHAP"]],
                                                "Feature_Value", 
                                                "Avg_SHAP")
                    feature_bar.savefig(figure_save_path + f'/{c}/{model}/{model}_{c}_{f}_bar.png', bbox_inches = 'tight')
                    plt.close()
                    #Find the feature value with the max average shap value and save normalized fraction
                    best_feature_value = mean_storage['Feature_Value'][mean_storage.loc[mean_storage['Feature'] == f, "Avg_SHAP"].astype(float).idxmax()]
                    min = mean_storage.loc[mean_storage['Feature'] == f, 'Feature_Value'].min()
                    max = mean_storage.loc[mean_storage['Feature'] == f, 'Feature_Value'].max()
                    normalized_value = (best_feature_value - min)/(max-min)
                    best_feature_values.append((f,f_type, normalized_value))

                  
                # If feature is not in list, then populate feature value with "NA"
                else:
                     mean_storage.loc[len(mean_storage)] = [f, float("NaN"),float("NaN"), float("NaN")]
                     best_feature_values.append((f,f_type,float("NaN")))
                     print(f"{f} is not in input parameters")
                    
            #print(mean_storage)
            #print(best_feature_values)
            df_best_feature_values = pd.DataFrame(best_feature_values, columns = ["Feature", "Type","Norm_Feature_Value"])
            #print(df_best_feature_values)

                

            #Save average shap of the features as csv
            with open(figure_save_path + f"/{c}/{model}/{model}_{c}_mean_shap.csv", 'w', encoding = 'utf-8-sig') as f:
                mean_storage.to_csv(f)

            #Create radar plot of the feature value with highest average shap for each feature
            radar_plot = plot_Radar(df_best_feature_values)
            radar_plot.write_image(figure_save_path + f'{model}_{c}_comp_radar.svg')
            plt.close()   
            # #for type in ("comp", "lipid", "physio")
            # print(df_best_feature_values)
            # rose_plot = plot_Rose(df_best_feature_values)
            # rose_plot.write_image(figure_save_path + f'{model}_{c}_comp_rose.svg')

            # #Create radar plot of the feature value with highest average shap for each feature
            # radar_plot = plot_Radar(lipid_features, best_feature_values, comp_features, lipid_features, phys_features)
            # radar_plot.savefig(save_path + f'{model}_{c}_lipid_radar.png', bbox_inches = 'tight')
            # plt.close()           

                #df_values.loc[df_input_data[f] == ]
            #Get the mean SHAP value for
            # values = shap_values.values
            # print(input_data)




            # with open(model_folder + f'/{c}/{model}_{c}_Best_Training_Data.pkl', "rb") as file:   # Unpickling
            #     train_data = pickle.load(file)
            # X =  train_data[features]








    # cell_type_names = ['HEK293','HepG2', 'N2a', 'ARPE19', 'B16', 'PC3', 'overall']
    # for cell in cell_type_names:
    #     if cell == "B16":
    #         values = [1, 6, 1, 10, 7]
    #     elif cell =="HEK293":
    #         values = [1, 7, 10, 3, 10]
    #     elif cell == "N2a":
    #         values = [1, 1, 3, 3, 7]
    #     elif cell == "HepG2":
    #         values = [1, 10, 10, 3, 10]
    #     elif cell == "ARPE19":
    #         values = [1, 7, 10, 3, 10]
    #     elif cell == "PC3":
    #         values = [1, 1, 10, 3, 10]
    #     elif cell =="overall":
    #         values = [1, 7, 10, 3, 7]
    #     else:
    #         values = [0, 0, 0, 0, 0]
    #     plot_Radar(cell, features, values, save_path)


if __name__ == "__main__":
    main()