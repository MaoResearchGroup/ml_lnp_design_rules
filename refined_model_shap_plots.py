# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

def plot_dependence(feature_index, interaction, shap_values, X, features_names,  save_path):
  shap.dependence_plot(ind=feature_index, 
                        shap_values=shap_values.values, 
                        features=X, 
                        feature_names=features_names,
                        interaction_index = interaction, 
                        show=False)
  #Save plot
  plt.show()
  # plt.savefig(save_path + f'{model}_{cell}_{formulation}_dependence.png', bbox_inches = 'tight') 
def SHAP_cluster(projections, color_values, cmap, size, title, figure_save_path, fmin, fmax):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 12


    # Dimensionality Reduction: SHAP values on Transfection Efficiency
    f = plt.figure(figsize=(size,size))
    plt.scatter(projections[:,0],
            projections[:,1],
            c = color_values,
            marker = 'o',
            s = size*5,
            linewidth=0.1, 
            alpha=0.8, 
            edgecolor='black', 
            cmap=cmap)
    # plt.title(f"{title}")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    cb = plt.colorbar(label=None, aspect=40, orientation="horizontal")
    cb.set_alpha(1)
    cb.draw_all()
    cb.outline.set_linewidth(0.2)
    cb.ax.tick_params('x', length=2)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.get_title()
    cb.ax.set_title(label=f"{title}", fontsize=10, color="black", weight="bold")
    cb.set_ticks([0.05, 0.95])
    cb.set_ticklabels([fmin, fmax])
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(), fontsize=10, color="black", weight="bold")

    plt.gca().axis("off") # no axis

    # y-axis limits and interval
    plt.gca().set(ylim=(-30, 30), yticks=np.arange(-40, 40, 10))
    plt.gca().set(xlim=(-40, 60), xticks=np.arange(-50, 70, 10))

    plt.savefig(figure_save_path, dpi = 600, transparent = True, bbox_inches = 'tight')
    plt.close()







def main(cell_model_list, model_folder, shap_value_path, plot_save_path):
  for cell_model in cell_model_list:
      c = cell_model[0]
      model_name = cell_model[1]
      #SHAP Values
      with open(shap_value_path + f"{model_name}_{c}_SHAP_values.pkl", "rb") as file:   # Unpickling
        shap_values = pickle.load(file)
      X = shap_values.data

    # #SHAP interaction Values
    # with open(shap_value_path + f"{model_name}_{c}_SHAP_inter_value_list.pkl", "rb") as file:   # Unpickling
    #   shap_inter_values_list = pickle.load(file)

      #Extract input parameters for each cell type
      with open(model_folder + f"{c}/{model_name}_{c}_Best_Model_Results.pkl", 'rb') as file: # import trained model
                best_results = pickle.load(file)
      input_params = best_results.loc['Feature names'][0]

      #Extract actual transfection data
      with open(model_folder + f"{c}/{model_name}_{c}_output.pkl", 'rb') as file: # import trained model
        true_Y = pickle.load(file)

      #Create Feature Order
      col2num = {col: i for i, col in enumerate(input_params)}
      feature_order = list(map(col2num.get, input_params))

      #Check Figure save path
      if os.path.exists(plot_save_path) == False:
          os.makedirs(plot_save_path, 0o666)

      # #Create and save plots
      # plot_summary(shap_values, c, model_name, feature_order, plot_save_path)
      # plot_importance(shap_values, c, model_name, feature_order, plot_save_path)
      
      #plot_force(479, shap_values_list[cell_type_list.index(c)], c, model_name, feature_order, plot_save_path)
      plot_dependence(0, 
                      7, 
                      shap_values, 
                      X, 
                      input_params, 
                      plot_save_path + f'{model_name}_{c}_SHAP_Depdendence.svg')
      # #EMBEDDED PLOT (USE FROM SHAP PACKAGE?)
      # shap_pca50 = PCA(n_components=len(input_params)).fit_transform(shap_values.values)
      # shap_embedded = TSNE(n_components=2, perplexity=50, random_state = 0).fit_transform(shap_values.values)
      # SHAP_cluster(projections=shap_embedded,
      #               color_values=true_Y.values,
      #               cmap = "Reds",
      #               size = 3,
      #               title = "Normalized Transfection Efficiency",
      #               figure_save_path= plot_save_path + f'{model_name}_{c}_SHAP_tfxn.svg',
      #               fmin = 0,
      #               fmax = 1)
      # for i in range(len(input_params)):
      #   print(f"Feature: {input_params[i]}")
      #   norm_feature = (X[:,i]-X[:,i].min())/(X[:,i].max()-X[:,i].min())
      #   SHAP_cluster(projections=shap_embedded,
      #             color_values=norm_feature,
      #             cmap = "viridis",
      #             size = 2,
      #             title = input_params[i],
      #             figure_save_path= plot_save_path + f'{model_name}_{c}_SHAP_cluster_{input_params[i]}.svg',
      #             fmin = X[:,i].min(),
      #             fmax = X[:,i].max())

      # plot_interaction(shap_inter_values_list[cell_type_list.index(c)], X, c, model_name, plot_save_path)
      


    # with open(shap_value_path + f"{model_name}_SHAP_inter_value_list.pkl", "rb") as file:   # Unpickling
    #   shap_inter_values_list = pickle.load(file)
    # # #dependance plot
    # # shap.dependence_plot(
    # # ('Dlin-MC3_Helper lipid_ratio', 'cLogP'),
    # # shap_inter_values_list[0], train_data,
    # # display_features=train_data)
if __name__ == "__main__":
    main()