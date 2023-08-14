# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import os

def plot_summary(shap_values, cell, model, feature_order, save):
# #Plot Beeswarm
  fig = plt.figure()    
  #Plots
  ax1 = fig.add_subplot(111)
  shap.plots.beeswarm(shap_values, max_display=15,show=False,
                      color_bar=False, order=feature_order, 
                      color=plt.get_cmap('viridis'))
  
  ax1.set_title(cell)

  #Set X axis limis
  ax1.set_xlim( xmin = -0.2, xmax = 0.2)


  #Format Y axis
  ax1.tick_params(axis='y', labelsize=10)


  #Colorbar
  cbar = plt.colorbar(label = "Feature Value",  ax = ax1, ticks = [], aspect= 5)
  cbar.ax.text(0.5, -0.01, 'Low', transform=cbar.ax.transAxes, 
    va='top', ha='center')
  cbar.ax.text(0.5, 1.0, 'High', transform=cbar.ax.transAxes, 
    va='bottom', ha='center')


  #Overall Plot Formats
  fig.suptitle(f'{model}_{cell} SHAP Summary Plots' , horizontalalignment='right', verticalalignment='top', fontsize = 20)
  plt.gcf().set_size_inches(12, 15)

  plt.savefig(save + f'{model}_{cell}_Summary.png', bbox_inches = 'tight')   

def plot_importance(shap_values, cell, model, feature_order,save):
  #Feature importance Bar plot
  fig = plt.figure() 
  ax1 = fig.add_subplot(111)
  shap.plots.bar(shap_values, max_display=15,show=False, order=feature_order)
  ax1.set_title(cell)
  
  #Remove duplicate X axis labels
  ax1.set(xlabel=None)  

  #Set X axis limis
  ax1.set_xlim(xmin = 0, xmax = 0.12)

  #Format Y axis
  ax1.tick_params(axis='y', labelsize=10)


  #Overall Plot Formats
  fig.suptitle(f'{model}_{cell} SHAP Feature Importance Plots' , horizontalalignment='right', verticalalignment='top', fontsize = 20)
  plt.gcf().set_size_inches(12, 15)

  #Save plot
  plt.savefig(save + f'{model}_{cell}_Bar.png', bbox_inches = 'tight')

def main():

  ################ Retreive Data ##############################################
  model_folder = "Feature_Reduction/Feature_reduction_NoSizeZeta/" 
  shap_value_path = 'SHAP_Values/Refined_Models_NoSizeZeta/'
  plot_save_path = "Figures/SHAP/Refined_Models_NoSizeZeta/"
 
  ##################### Run Predictions ###############################
  #Training Data
  cell_type_list = ['ARPE19','N2a','PC3','B16','HEK293','HepG2']
  model_list = ['LGBM', 'XGB','RF'] #Did not include SVR


  #Extracting SHAP Values
  for model_name in model_list:

    #SHAP Values
    with open(shap_value_path + f"{model_name}_SHAP_value_list.pkl", "rb") as file:   # Unpickling
      shap_values_list = pickle.load(file)

    for c in cell_type_list:

      #Extract input parameters for each cell type
      with open(model_folder + f"{c}/{model_name}_{c}_Best_Model_Results.pkl", 'rb') as file: # import trained model
                best_results = pickle.load(file)
      input_params = best_results.loc['Feature names'][0]
      print(input_params)
      

      #Create Feature Order
      col2num = {col: i for i, col in enumerate(input_params)}
      feature_order = list(map(col2num.get, input_params))
      
      #Check Figure save path
      if os.path.exists(plot_save_path) == False:
          os.makedirs(plot_save_path, 0o666)

      #Create and save plots
      plot_summary(shap_values_list[cell_type_list.index(c)], c, model_name, feature_order, plot_save_path)
      plot_importance(shap_values_list[cell_type_list.index(c)], c, model_name, feature_order, plot_save_path)
      




    # with open(shap_value_path + f"{model_name}_SHAP_inter_value_list.pkl", "rb") as file:   # Unpickling
    #   shap_inter_values_list = pickle.load(file)
  
   





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