# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
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
  model_folder = "Trained_Models/230204_Models/" #Leo
  datafile_path = 'Raw_Data/7_Master_Formulas.csv' 
  plot_save_path = "Figures/SHAP/"
  ################ INPUT PARAMETERS ############################################

  wt_percent = False
  if wt_percent == True:
    formulation_param_names = ['wt_Helper', 'wt_Dlin','wt_Chol', 'wt_DMG', 'wt_pDNA']
  else:
    formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                        'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio']
  helper_lipid_names = ['18PG', 'DOPE','DOTAP','DSPC', '14PA', 'DDAB']

  
  lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA',
                       'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
  input_param_names =  lipid_param_names + formulation_param_names


##################### Run Predictions ###############################
  #Training Data
  cell_type = ['HEK293', 'HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']
  #cell_type = ['HEK293', 'HepG2', 'N2a']
  model_list = ['RF']
  #model_list = ['RF', 'MLR', 'lasso', 'PLS', 'SVR', 'kNN', 'LGBM', 'XGB', 'DT']
  #Extracting SHAP Values
  for model_name in model_list:
    with open(f"SHAP_Values/{model_name}_SHAP_value_list.pkl", "rb") as file:   # Unpickling
      shap_values_list = pickle.load(file)
    
    
    #Plotting
    fig = plt.figure()               
    col2num = {col: i for i, col in enumerate(input_param_names)}
    feature_order = list(map(col2num.get, input_param_names))
    #print(feature_order)
    
    #Plots
    ax1 = fig.add_subplot(231)
    shap.plots.beeswarm(shap_values_list[0], max_display=12,show=False, color_bar=False, order=feature_order)
    ax1.set_title(cell_type[0])
    
    ax2 = fig.add_subplot(232)
    shap.plots.beeswarm(shap_values_list[1], max_display=12,show=False, color_bar=False, order=feature_order)
    ax2.set_title(cell_type[1])

    ax3 = fig.add_subplot(233)
    shap.plots.beeswarm(shap_values_list[2], max_display=12,show=False, color_bar=False, order=feature_order)
    ax3.set_title(cell_type[2])

    ax4 =fig.add_subplot(234)
    shap.plots.beeswarm(shap_values_list[3], max_display=12,show=False, color_bar=False, order=feature_order)
    ax4.set_title(cell_type[3])

    ax5 =fig.add_subplot(235)
    shap.plots.beeswarm(shap_values_list[4], max_display=12,show=False, color_bar=False, order=feature_order)
    ax5.set_title(cell_type[4])

    ax6 =fig.add_subplot(236)
    shap.plots.beeswarm(shap_values_list[5], max_display=12,show=False, color_bar=False, order=feature_order)
    ax6.set_title(cell_type[5])

    #Remove duplicate X axis labels
    ax1.set(xlabel=None)  
    ax2.set(xlabel=None)
    ax3.set(xlabel=None)
    ax4.set(xlabel=None)
    ax6.set(xlabel=None)

    #Set X axis limis
    ax1.set_xlim( xmin = -0.2, xmax = 0.2)
    ax2.set_xlim( xmin = -0.2, xmax = 0.2)
    ax3.set_xlim( xmin = -0.2, xmax = 0.2)
    ax4.set_xlim( xmin = -0.2, xmax = 0.2)
    ax5.set_xlim( xmin = -0.2, xmax = 0.2)
    ax6.set_xlim( xmin = -0.2, xmax = 0.2)
    #Remove duplicate y axis labels
    ax2.get_yaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)

    #Format Y axis
    ax1.tick_params(axis='y', labelsize=10)
    ax4.tick_params(axis='y', labelsize=10)


    #Colorbar
    cbar = plt.colorbar(label = "Feature Value", ax = [ax1, ax2, ax3, ax4, ax5, ax6], ticks = [])
    cbar.ax.text(0.5, -0.01, 'Low', transform=cbar.ax.transAxes, 
      va='top', ha='center')
    cbar.ax.text(0.5, 1.0, 'High', transform=cbar.ax.transAxes, 
      va='bottom', ha='center')
    # plt.tight_layout() 
    # plt.show()
 

    #Overall Plot Formats
    fig.suptitle(f'{model_name} SHAP Summary Plots' , horizontalalignment='right', verticalalignment='top', fontsize = 20)
    plt.gcf().set_size_inches(15,8)

    #Save plot
    plt.savefig(plot_save_path + f'{model_name}_Summary.png', bbox_inches = 'tight')


if __name__ == "__main__":
    main()

'''
Implement Gridsearch for hyperparameter tuning; this could be improved by focusing on a single hyperparameter and demonstrating the impact on accuracy based on chnaging this
'''

'''
See if areas of high transfection efficiency can be identified based on the attributes; another way of phrasing this problem can be of one based on unsupervised clustering; can try Bayesian optimization as well
'''