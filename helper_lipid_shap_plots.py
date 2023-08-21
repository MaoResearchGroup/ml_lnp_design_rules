# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import os


def init_data(filepath,cell_type_names):
    """ Takes a full file path to a spreadsheet and an array of cell type names. 
    Returns a dataframe with 0s replaced with 1s."""
    df = pd.read_csv(filepath)
    for cell_type in df.columns[-len(cell_type_names):]:
      zero_rows = np.where(df[cell_type] == 0)
      for i in zero_rows:
        df[cell_type][i] = 1
    return df

def extract_training_data(data_path, cell_type_list, input_param_names):
    df = init_data(data_path, cell_type_list)
    training_data_array = []
    for cell in cell_type_list:
       training_data = df.loc[:,df.columns.isin(['Formula label', 'Helper_lipid'] + input_param_names + ['RLU_'+ cell])]
       training_data.dropna(inplace = True)
       training_data.reset_index(drop = True, inplace=True)
       training_data_array.append(training_data.loc[:, training_data.columns.isin(input_param_names)]) #Store only X training data
    return training_data_array
def main():

  ################ Retreive Data ##############################################
  shap_value_path = 'SHAP_Values/Refined_Models_NoSizeZeta/Helper_Lipid/'
  plot_save_path = "Figures/SHAP/Refined_Models_NoSizeZeta/Helper_lipid/"
  model_folder = "Feature_Reduction/Feature_reduction_NoSizeZeta/" 


##################### Run Predictions ###############################
  #Training Data
  helper_list = ['DOTAP', 'DDAB', '14PA', '18PG', 'DOPE', 'DSPC']
  model_list = ['LGBM', 'XGB','RF'] #Did not include SVR
  #model_list = ['RF', 'MLR', 'lasso', 'PLS', 'SVR', 'kNN', 'LGBM', 'XGB', 'DT']


  #Extracting SHAP Values
  for model_name in model_list:
  
  #   #Training Data For interaction plots
  #   with open(model_folder + f'{model_name}/{cell_type}/{cell_type}_Training_Data.pkl', "rb") as file:   # Unpickling
  #       train_data = pickle.load(file)
  #       train_data =  train_data[input_param_names]

    #SHAP Values
    with open(shap_value_path + f"{model_name}_SHAP_value_list.pkl", "rb") as file:   # Unpickling
      shap_values_list = pickle.load(file)

    # with open(shap_value_path + f"{model_name}_SHAP_inter_value_list.pkl", "rb") as file:   # Unpickling
    #   shap_inter_values_list = pickle.load(file)

    #Extract input parameters for each cell type
    with open(model_folder + f"B16/{model_name}_B16_Best_Model_Results.pkl", 'rb') as file: # import trained model
              best_results = pickle.load(file)
    input_param_names = best_results.loc['Feature names'][0]
    print(input_param_names) 

    # #Plot Beeswarm
    fig = plt.figure()               
    col2num = {col: i for i, col in enumerate(input_param_names)}
    feature_order = list(map(col2num.get, input_param_names))
    
    #Plots
    ax1 = fig.add_subplot(321)
    shap.plots.beeswarm(shap_values_list[0], max_display=15,show=False, color_bar=False, order=feature_order, color=plt.get_cmap('viridis'))
    ax1.set_title(helper_list[0])
    
    ax2 = fig.add_subplot(322)
    shap.plots.beeswarm(shap_values_list[1], max_display=15,show=False, color_bar=False, order=feature_order, color=plt.get_cmap('viridis'))
    ax2.set_title(helper_list[1])

    ax3 = fig.add_subplot(323)
    shap.plots.beeswarm(shap_values_list[2], max_display=15,show=False, color_bar=False, order=feature_order, color=plt.get_cmap('viridis'))
    ax3.set_title(helper_list[2])

    ax4 =fig.add_subplot(324)
    shap.plots.beeswarm(shap_values_list[3], max_display=15,show=False, color_bar=False, order=feature_order, color=plt.get_cmap('viridis'))
    ax4.set_title(helper_list[3])

    ax5 =fig.add_subplot(325)
    shap.plots.beeswarm(shap_values_list[4], max_display=15,show=False, color_bar=False, order=feature_order, color=plt.get_cmap('viridis'))
    ax5.set_title(helper_list[4])

    ax6 =fig.add_subplot(326)
    shap.plots.beeswarm(shap_values_list[5], max_display=15,show=False, color_bar=False, order=feature_order, color=plt.get_cmap('viridis'))
    ax6.set_title(helper_list[5])

    #Remove duplicate X axis labels
    ax1.set(xlabel=None)  
    ax2.set(xlabel=None)
    ax3.set(xlabel=None)
    ax4.set(xlabel=None)

    #Set X axis limis
    ax1.set_xlim( xmin = -0.2, xmax = 0.2)
    ax2.set_xlim( xmin = -0.2, xmax = 0.2)
    ax3.set_xlim( xmin = -0.2, xmax = 0.2)
    ax4.set_xlim( xmin = -0.2, xmax = 0.2)
    ax5.set_xlim( xmin = -0.2, xmax = 0.2)
    ax6.set_xlim( xmin = -0.2, xmax = 0.2)
    #Remove duplicate y axis labels
    ax2.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)

    #Format Y axis
    ax1.tick_params(axis='y', labelsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    ax5.tick_params(axis='y', labelsize=10)


    #Colorbar
    cbar = plt.colorbar(label = "Feature Value",  ax = [ax1, ax2, ax3, ax4, ax5, ax6], ticks = [], aspect=20)
    cbar.ax.text(0.5, -0.01, 'Low', transform=cbar.ax.transAxes, 
      va='top', ha='center')
    cbar.ax.text(0.5, 1.0, 'High', transform=cbar.ax.transAxes, 
      va='bottom', ha='center')
    # plt.tight_layout() 
    # plt.show()
 

    #Overall Plot Formats
    fig.suptitle(f'{model_name} SHAP Summary Plots' , horizontalalignment='right', verticalalignment='top', fontsize = 20)
    plt.gcf().set_size_inches(12, 15)

    #Save plot
    if os.path.exists(plot_save_path) == False:
        os.makedirs(plot_save_path, 0o666)
    plt.savefig(plot_save_path + f'{model_name}_Summary.png', bbox_inches = 'tight')


    #Feature importance Bar plot
    fig2 = plt.figure() 
    ax1 = fig2.add_subplot(321)
    shap.plots.bar(shap_values_list[0], max_display=15,show=False, order=feature_order)
    ax1.set_title(helper_list[0])
    
    ax2 = fig2.add_subplot(322)
    shap.plots.bar(shap_values_list[1], max_display=15,show=False, order=feature_order)
    ax2.set_title(helper_list[1])

    ax3 = fig2.add_subplot(323)
    shap.plots.bar(shap_values_list[2], max_display=15,show=False, order=feature_order)
    ax3.set_title(helper_list[2])

    ax4 =fig2.add_subplot(324)
    shap.plots.bar(shap_values_list[3], max_display=15,show=False,order=feature_order)
    ax4.set_title(helper_list[3])

    ax5 =fig2.add_subplot(325)
    shap.plots.bar(shap_values_list[4], max_display=15,show=False,  order=feature_order)
    ax5.set_title(helper_list[4])

    ax6 =fig2.add_subplot(326)
    shap.plots.bar(shap_values_list[5], max_display=15,show=False, order=feature_order)
    ax6.set_title(helper_list[5])

    #Remove duplicate X axis labels
    ax1.set(xlabel=None)  
    ax2.set(xlabel=None)
    ax3.set(xlabel=None)
    ax4.set(xlabel=None)

    #Set X axis limis
    fig2_x_min = 0
    fig2_x_max = .12
    ax1.set_xlim( xmin = fig2_x_min , xmax = fig2_x_max)
    ax2.set_xlim( xmin = fig2_x_min , xmax = fig2_x_max)
    ax3.set_xlim( xmin = fig2_x_min , xmax = fig2_x_max)
    ax4.set_xlim( xmin = fig2_x_min , xmax = fig2_x_max)
    ax5.set_xlim( xmin = fig2_x_min , xmax = fig2_x_max)
    ax6.set_xlim( xmin = fig2_x_min , xmax = fig2_x_max)
    #Remove duplicate y axis labels
    ax2.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)

    #Format Y axis
    ax1.tick_params(axis='y', labelsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    ax5.tick_params(axis='y', labelsize=10)
 

    #Overall Plot Formats
    fig2.suptitle(f'{model_name} SHAP Feature Importance Plots' , horizontalalignment='right', verticalalignment='top', fontsize = 20)
    plt.gcf().set_size_inches(12, 15)

    # shap.plots.beeswarm(shap_values_list[0], max_display=15,show=False, color_bar=True, color=plt.get_cmap('viridis'))
    #Save plot
    plt.savefig(plot_save_path + f'{model_name}_Bar.png', bbox_inches = 'tight')

    # plt.tight_layout()
    # plt.show()



    # #Interaction Plots
    # shap.summary_plot(shap_inter_values_list[0], train_data, max_display=15, show = False)
    # f = plt.gcf()
    # plt.colorbar()
    # plt.set_cmap('viridis')
    # #Save plot
    # f.savefig(plot_save_path + f'{model_name}_inter_summary.png', bbox_inches = 'tight')

    # # #dependance plot
    # # shap.dependence_plot(
    # # ('Dlin-MC3_Helper lipid_ratio', 'cLogP'),
    # # shap_inter_values_list[0], train_data,
    # # display_features=train_data)
if __name__ == "__main__":
    main()