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


def main():
  ################ Retreive Data ##############################################
  
    plot_save_path = "Figures/Validation/"

    results = pd.read_csv('Predictions/HEK_in_vitro_validation/230127_HEK_combined_results.csv')
    results["Experimental_Transfection"].replace(to_replace = 0, value = 1, inplace=True) #Remove 0 values
    print(results)

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
    input_param_names = lipid_param_names + formulation_param_names 


    ####################################################
    cell_type = 'HEK293'
    model_list = ['RF', 'XGB', 'LGBM']
    experimental = results["Experimental_Transfection"]
    prediction_list = []
    for model_name in model_list:
        print(f'###############{model_name}#############')
        prediction_list.append(results[f"{model_name}_RLU_Predicted_Values"])

    #Plotting
    fig, axs = plt.subplots(1, len(model_list), sharex=True, sharey=True, figsize=(10,4))
    for ax_ind, ax in enumerate(axs):
        print(ax_ind)
        predicted_values = prediction_list[ax_ind]
        ax.scatter(predicted_values, experimental)
        ax.set_title(f'{model_list[ax_ind]}')
        if ax_ind == 0:
            ax.set_ylabel('Experimental_RLU')
        elif ax_ind == 2:
            ax.set_xlabel('Predicted_RLU')

        ax.legend(frameon=False, handlelength=0)
    plt.savefig(plot_save_path + f'Validation.png', bbox_inches = 'tight')

if __name__ == "__main__":
    main()
