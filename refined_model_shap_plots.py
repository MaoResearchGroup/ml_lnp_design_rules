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
                      color=plt.get_cmap('viridis_r'))
  

  #Plot Formatting
  # ax1.set_title(cell, fontsize = 30)


  #Set X axis limis
  ax1.set_xlim(xmin = -0.3, xmax = 0.3)


  #Format Y axis
  ax1.tick_params(axis='y', labelsize=20)
  #Format X axis
  ax1.tick_params(axis='x', labelsize=12)
  ax1.set_xlabel("SHAP Value (impact on model output)", font = "Arial", fontsize = 20)

  #Colorbar
  cbar = plt.colorbar(ax = ax1, ticks = [], aspect= 20)
  cbar.ax.text(0.5, -0.01, 'Low', fontsize = 20, transform=cbar.ax.transAxes, 
    va='top', ha='center')
  cbar.ax.text(0.5, 1.0, 'High',fontsize = 20, transform=cbar.ax.transAxes, 
    va='bottom', ha='center')
  cbar.set_label(label = "Relative Feature Value", size = 20)


  #Overall Plot Formats
  # fig.suptitle(f'{model}_{cell} SHAP Summary Plots' , horizontalalignment='right', verticalalignment='top', fontsize = 20)
  plt.gcf().set_size_inches(12, 8)

  plt.savefig(save + f'{model}_{cell}_Summary.svg', dpi = 600, transparent = True, bbox_inches = 'tight')   
  plt.close()


def plot_importance(shap_values, cell, model, feature_order,save):
  #Feature importance Bar plot
  fig = plt.figure() 
  ax1 = fig.add_subplot(111)
  shap.plots.bar(shap_values, max_display=15,show=False, order=feature_order)
  ax1.set_title(cell, fontsize = 30)
 
  #Set X axis limis
  ax1.set_xlim(xmin = 0, xmax = 0.15)

  #Format Y axis
  ax1.tick_params(axis='y', labelsize=15)


  #Overall Plot Formats
  fig.suptitle(f'{model}_{cell} SHAP Feature Importance Plots' , horizontalalignment='right', verticalalignment='top', fontsize = 20)
  plt.gcf().set_size_inches(6, 8)

  #Save plot
  plt.savefig(save + f'{model}_{cell}_Bar.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
  plt.close()

def plot_interaction(shap_values, data, cell, model, save):
  plt.close('all')
  plt.set_cmap('viridis')
  shap.summary_plot(shap_values, data, max_display=15, show = False, color=plt.get_cmap('viridis'))
  f = plt.gcf()
  plt.colorbar()
  #Save plot
  f.savefig(save + f'{model}_{cell}_inter_summary.png', bbox_inches = 'tight')
  plt.close()


def plot_force(formulation, shap_values, cell, model, feature_order,save):
   #Feature importance Bar plot
  fig = plt.figure() 
  ax3 = fig.add_subplot(111)
  shap.plots.force(shap_values[formulation])

  ax3.set_title(cell + str(formulation))

  plt.show()
  # #Remove duplicate X axis labels
  # ax3.set(xlabel=None)  

  # #Set X axis limis
  # ax3.set_xlim(xmin = 0, xmax = 0.12)

  # #Format Y axis
  # ax3.tick_params(axis='y', labelsize=10)


  #Overall Plot Formats
  fig.suptitle(f'{model}_{cell}_{formulation} SHAP Force Plots' , horizontalalignment='right', verticalalignment='top', fontsize = 20)
  plt.gcf().set_size_inches(6, 8)

  #Save plot
  plt.savefig(save + f'{model}_{cell}_{formulation}_Force.png', bbox_inches = 'tight')  


def main(cell_model_list, model_folder, shap_value_path, plot_save_path):
  print("BUMP PLOT")
  for cell_model in cell_model_list:
      c = cell_model[0]
      model_name = cell_model[1]
      #SHAP Values
      with open(shap_value_path + f"{model_name}_{c}_SHAP_values.pkl", "rb") as file:   # Unpickling
        shap_values = pickle.load(file)
        
    # #SHAP interaction Values
    # with open(shap_value_path + f"{model_name}_{c}_SHAP_inter_value_list.pkl", "rb") as file:   # Unpickling
    #   shap_inter_values_list = pickle.load(file)

      #Extract input parameters for each cell type
      with open(model_folder + f"{c}/{model_name}_{c}_Best_Model_Results.pkl", 'rb') as file: # import trained model
                best_results = pickle.load(file)
      input_params = best_results.loc['Feature names'][0]
      print(input_params)
      

      #Create Feature Order
      col2num = {col: i for i, col in enumerate(input_params)}
      feature_order = list(map(col2num.get, input_params))
      print(feature_order)
      #Check Figure save path
      if os.path.exists(plot_save_path) == False:
          os.makedirs(plot_save_path, 0o666)

      #Create and save plots
      plot_summary(shap_values, c, model_name, feature_order, plot_save_path)
      plot_importance(shap_values, c, model_name, feature_order, plot_save_path)
      #plot_force(479, shap_values_list[cell_type_list.index(c)], c, model_name, feature_order, plot_save_path)



      # # Interaction plots
      # with open(model_folder + f"{c}/{model_name}_{c}_Best_Model_Results.pkl", 'rb') as file: # import trained model
      #           best_results = pickle.load(file)
      # input_param_names = best_results.loc['Feature names'][0]


      # with open(model_folder + f'/{c}/{model_name}_{c}_Best_Training_Data.pkl', "rb") as file:   # Unpickling
      #   train_data = pickle.load(file)

      # X =  train_data[input_param_names]
      # plot_interaction(shap_inter_values_list[cell_type_list.index(c)], X, c, model_name, plot_save_path)
      


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