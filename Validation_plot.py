import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy import stats
import seaborn as sns

def plot_ind_scatter(df, model, pearsons, spearmans):
    cell_list = df["Cell_Type"].unique()
    fig, axes = plt.subplots(1 , len(cell_list), sharex = True, sharey= True, figsize = (15,5))
    # acc, spearman, pearson = evaluate_pred_quality(df["Experimental_RLU"], df["Predicted_RLU"])
    # print(acc, spearman, pearson)
    
    #plt.title(f'{cell_type} {model} Validation')
    plt.xlim(0,13)
    plt.ylim(0,13)
    fig.suptitle(f'{model}_Validation')
    for i, axs in enumerate(axes):
        sns.regplot(x = "Predicted_RLU", y = "Experimental_RLU", ax = axs, data = df.loc[df['Cell_Type'] == cell_list[i]]).set_title(f'{cell_list[i]}')
        axs.plot([0, 13], [0, 13], linestyle = 'dotted', color = 'r') #Ideal line
        axs.text(1, 12, f"Pearson's r= {round(pearsons[i], 3)}")
        axs.text(1, 11, f"Spearman's R= {round(spearmans[i], 3)}")
        if i > 0:
            axs.set(ylabel = None)
    
    plt.savefig(plot_save_path + f'{model}_ind_scatter.png', bbox_inches = 'tight')



def plot_combined_scatter(df, model):
    plt.figure(figsize = (5,5))
    plt.xlim(0,13)
    plt.ylim(0,13)
    
    sns.lmplot(x = "Predicted_RLU", y = "Experimental_RLU", hue = "Cell_Type", data = df).set(title = f'{model}_Validation')
    plt.plot([0, 13], [0, 13], linestyle = 'dotted', color = 'r') #Ideal line
    
    plt.savefig(plot_save_path + f'{model}_combined_scatter.png', bbox_inches = 'tight')


def evaluate_pred_quality(exp, pred):
    #evaluate accuracy
    acc = mean_absolute_error(exp, pred)
    spearmans_rank = stats.spearmanr(exp, pred)
    pearsons_r = stats.pearsonr(exp, pred)

    return acc, spearmans_rank[0], pearsons_r[0]


plot_save_path = "Figures/Experimental_Validation/20230228/"
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



def main():
  ################ Retreive Data ##############################################
    cell_type_list = ['HEK293', 'B16', 'HepG2']
    model_list = ['XGB', 'RF']
    all_data = pd.DataFrame()
    pearson_list = []
    spearman_list = []
    MAE_list = []
    for model in model_list:
        for cell in cell_type_list:
            df = pd.read_csv(f'Validation_Data/20230228/20230228_{model}_{cell}.csv' )
            df["Cell_Type"] = cell
            acc, spearman, pearson = evaluate_pred_quality(df["Experimental_RLU"], df["Predicted_RLU"])
            MAE_list.append(acc)
            pearson_list.append(pearson)
            spearman_list.append(spearman)
            all_data = all_data.append(df, ignore_index = True)
            
        plot_ind_scatter(all_data, model, pearson_list, spearman_list) #Individual scatter
        plot_combined_scatter(all_data,  model) #Combined Scatter



if __name__ == "__main__":
    main()
