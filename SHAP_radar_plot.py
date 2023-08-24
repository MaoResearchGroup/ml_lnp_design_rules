import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
import pickle
import pandas as pd
import seaborn as sns
import os

def mean_shap_hist(data, xaxis, yaxis):
    fig = plt.figure()
    sns.barplot(data, x = xaxis, y = yaxis)
    fig.gca()

    return fig 


def plot_hist(X, y, feature, N_bins):
    fig = plt.figure()
    y.rename("SHAP Value", inplace = True)
    datadf = pd.concat([X,y], axis = 1)
    print(datadf)
    bin_size = (X.max()-X.min())/N_bins
    sns.histplot(datadf, x = X.name, y = "SHAP Value", binwidth = bin_size)
    fig.gca()


def plot_Radar(labels, values):

    #Create labels and correct angles
    labels=np.array(labels)
    values=np.array(values)
    angles=[n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
    
    # Add data to close plot
    labels = np.concatenate((labels,[labels[0]]))
    values=np.concatenate((values,[values[0]]))
    angles=np.concatenate((angles,[angles[0]]))

    #Plotting
    fig=plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.3)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.grid(True)
    return fig

def main():
    ################ Retreive/Store Data ##############################################
    RUN_NAME = "Feature_reduction_Size_600_Zeta_PDI_0.45"
    save_path = f"Figures/Radar/{RUN_NAME}/"
    model_folder = f"Feature_Reduction/{RUN_NAME}/"
    shap_value_path = f'SHAP_Values/{RUN_NAME}/'
    
    cell_type = ['ARPE19','N2a','PC3','B16','HEK293','HepG2']

    model_list = ['LGBM', 'XGB','RF']

    # #testing
    # cell_type = ['B16']
    # model_list = ['LGBM']


    features = ['NP_ratio', 
                'Dlin-MC3_Helper lipid_ratio',
                'Dlin-MC3+Helper lipid percentage', 
                'Chol_DMG-PEG_ratio']

    ################ INPUT PARAMETERS ############################################
    #cell_type_names = ['HEK293','HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']
    for c in cell_type:
        for model in model_list:

            #Check save paths          
            if os.path.exists(save_path + f'/{c}/{model}/') == False:
                os.makedirs(save_path + f'/{c}/{model}/', 0o666)

            #Get feature names used to train model
            with open(model_folder + f"{c}/{model}_{c}_Best_Model_Results.pkl", 'rb') as file: # import best model results
                        best_results = pickle.load(file)
            input_param_names = best_results.loc['Feature names'][0]  
            print(input_param_names)          
            mean_shap = pd.DataFrame(columns = input_param_names)
            #Get SHAP Values
            with open(shap_value_path + f"{model}_{c}_SHAP_values.pkl", "rb") as file:   # Unpickling
                shap_values = pickle.load(file)

            #Dataframe of input data to generate SHAP values
            df_input_data = pd.DataFrame(shap_values.data, columns = input_param_names)

            #Dataframe of shap values
            df_values = pd.DataFrame(shap_values.values, columns = input_param_names)

            #dataframe to store average SHAP values
            mean_storage = pd.DataFrame(columns = ["Feature", "Feature_Value", "Avg_SHAP"])
            
            #List to store best feature values
            best_feature_values = []
            #Iterate through all feature names to find unique feature values
            for f in features:
                unique_feature_values = df_input_data[f].unique()
                
                #Iterate through unique feature values to get average SHAP for that feature value
                for feature_value in unique_feature_values:
                    #Get the mean shap value for the unique feature value
                    feature_mean = df_values.loc[df_input_data[f] == feature_value, f].mean()

                    #Store the mean shap value
                    mean_storage.loc[len(mean_storage)] = [f, feature_value, feature_mean]

                #Find the feature value with the max average shap value and save normalized fraction
                best_feature_value = mean_storage['Feature_Value'][mean_storage.loc[mean_storage['Feature'] == f, "Avg_SHAP"].idxmax()]
                min = mean_storage.loc[mean_storage['Feature'] == f, 'Feature_Value'].min()
                max = mean_storage.loc[mean_storage['Feature'] == f, 'Feature_Value'].max()
                normalized_value = (best_feature_value - min)/(max-min)
                best_feature_values.append(normalized_value)

                #Create a histogram of the avg_shap value for each feature value
                feature_bar = mean_shap_bar(mean_storage.loc[mean_storage["Feature"] == f, ["Feature_Value", "Avg_SHAP"]],
                                            "Feature_Value", 
                                            "Avg_SHAP")
                plt.show()
                #feature_bar.savefig(save_path + f'/{c}/{model}/{model}_{c}_{f}_bar.png', bbox_inches = 'tight')
                plt.close()

                

            #Save average shap of the features as csv
            with open(save_path + f"/{c}/{model}/{model}_{c}_mean_shap.csv", 'w', encoding = 'utf-8-sig') as f:
                mean_storage.to_csv(f)

            #Create radar plot of the feature value with highest average shap for each feature
            radar_plot = plot_Radar(features, best_feature_values)
            radar_plot.savefig(save_path + f'{model}_{c}_radar.png', bbox_inches = 'tight')
            plt.close()            

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