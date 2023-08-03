import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy import stats
import seaborn as sns

def plot_ind_scatter(df, model, pearsons, spearmans):
    cell_list = df["Cell_Type"].unique()
    fig, axes = plt.subplots(1 , len(cell_list), sharex = True, sharey= True, figsize = (15,5))

    experimental = df.loc[df['Type'] == "Experimental"]
    control = df.loc[df['Type'] == "Control"]

    # acc, spearman, pearson = evaluate_pred_quality(df["Experimental_RLU"], df["Predicted_RLU"])
    # print(acc, spearman, pearson)
    
    #plt.title(f'{cell_type} {model} Validation')
    plt.xlim(0,13)
    plt.ylim(0,13)
    fig.suptitle(f'{model}_Validation')
    for i, axs in enumerate(axes):
        sns.regplot(x = "Predicted_RLU", y = "Experimental_RLU", ax = axs, scatter = False, data = experimental.loc[experimental['Cell_Type'] == cell_list[i]]).set_title(f'{cell_list[i]}')
        sns.scatterplot(x = "Predicted_RLU", y = "Experimental_RLU", ax = axs, hue = "Helper_lipid", data = experimental.loc[experimental['Cell_Type'] == cell_list[i]])
        sns.scatterplot(x = "Predicted_RLU", y = "Experimental_RLU", ax = axs, data = control.loc[control['Cell_Type'] == cell_list[i]], color = 'r')
        axs.plot([0, 13], [0, 13], linestyle = 'dotted', color = 'r') #Ideal line
        axs.text(1, 12, f"Pearson's r= {round(pearsons[i], 3)}")
        axs.text(1, 11, f"Spearman's R= {round(spearmans[i], 3)}")
        
        #format y label
        if i > 0:
            axs.set(ylabel = None)

        #format legend
        if i < len(cell_list)-1:
            axs.legend([],[], frameon=False)
        else:
            axs.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    plt.savefig(plot_save_path + f'{model}_ind_scatter.png', bbox_inches = 'tight')


def plot_ind_scatter_feature(df, model, pearsons, spearmans, feature_name):
    cell_list = df["Cell_Type"].unique()
    fig, axes = plt.subplots(1 , len(cell_list), sharex = True, sharey= True, figsize = (15,5))

    experimental = df.loc[df['Type'] == "Experimental"]
    control = df.loc[df['Type'] == "Control"]

    plt.xlim(0,13)
    plt.ylim(0,13)

    fig.suptitle(f'{model}_Validation')
    for i, axs in enumerate(axes):
        sns.regplot(x = "Predicted_RLU", y = "Experimental_RLU", ax = axs, scatter = False, data = experimental.loc[experimental['Cell_Type'] == cell_list[i]]).set_title(f'{cell_list[i]}')
        sns.scatterplot(x = "Predicted_RLU", y = "Experimental_RLU", ax = axs, hue = feature_name, data = experimental.loc[experimental['Cell_Type'] == cell_list[i]])
        sns.scatterplot(x = "Predicted_RLU", y = "Experimental_RLU", ax = axs, data = control.loc[control['Cell_Type'] == cell_list[i]], color = 'r')
        axs.plot([0, 13], [0, 13], linestyle = 'dotted', color = 'r') #Ideal line
        axs.text(1, 12, f"Pearson's r= {round(pearsons[i], 3)}")
        axs.text(1, 11, f"Spearman's R= {round(spearmans[i], 3)}")
        
        #format y label
        if i > 0:
            axs.set(ylabel = None)

        #format legend
        if i < len(cell_list)-1:
            axs.legend([],[], frameon=False)
        else:
            axs.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    plt.savefig(plot_save_path + f'{model}_{feature_name}_ind_scatter.png', bbox_inches = 'tight')


def plot_ind_scatter_rank(df, model, pearsons, spearmans):
    cell_list = df["Cell_Type"].unique()
    fig, axes = plt.subplots(1 , len(cell_list), sharex = True, sharey= True, figsize = (15,5))

    experimental = df.loc[df['Type'] == "Experimental"]
    control = df.loc[df['Type'] == "Control"]

    fig.suptitle(f'{model}_Validation')


    for i, axs in enumerate(axes):
        sns.regplot(x = "Pred_Rank", y = "Exp_Rank", ax = axs, scatter = False, data = experimental.loc[experimental['Cell_Type'] == cell_list[i]]).set_title(f'{cell_list[i]}')
        sns.scatterplot(x = "Pred_Rank", y = "Exp_Rank", ax = axs, hue = "Helper_lipid", style = 'Type', markers = ["o", "X"], data = df.loc[df['Cell_Type'] == cell_list[i]])
        #sns.scatterplot(x = "Pred_Rank", y = "Exp_Rank", ax = axs, size = "Predicted_RLU", sizes =(1.5, 12), data = control.loc[control['Cell_Type'] == cell_list[i]], color = 'r')
        axs.plot([0, 100], [0, 100], linestyle = 'dotted', color = 'r') #Ideal line
        axs.invert_xaxis()
        axs.invert_yaxis()
        
        axs.text(100, 1, f"Pearson's r= {round(pearsons[i], 3)}", ha = 'left', va= 'top')
        axs.text(100, 5, f"Spearman's R= {round(spearmans[i], 3)}", ha = 'left', va= 'top')
        #format y label
        if i > 0:
            axs.set(ylabel = None)

        #format legend
        if i < len(cell_list)-1:
            axs.legend([],[], frameon=False)
        else:
            axs.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # plt.tight_layout()
    # plt.show()
    plt.savefig(plot_save_path + f'{model}_rank_ind_scatter.png', bbox_inches = 'tight')


def plot_ind_exp_rank(df, model):
    cell_list = df["Cell_Type"].unique()
    fig, axes = plt.subplots(1 , len(cell_list), sharex = True, sharey= True, figsize = (15,5))

    experimental = df.loc[df['Type'] == "Experimental"]
    control = df.loc[df['Type'] == "Control"]

    fig.suptitle(f'{model}_Validation')
    for i, axs in enumerate(axes):
        #sns.regplot(x = "Experimental_RLU", y = "Exp_Rank", ax = axs, scatter = False, data = experimental.loc[experimental['Cell_Type'] == cell_list[i]]).set_title(f'{cell_list[i]}')
        sns.scatterplot(x = "Experimental_RLU", y = "Exp_Rank", ax = axs, hue = "Helper_lipid", style = 'Type', markers = ["o", "X"], data = df.loc[df['Cell_Type'] == cell_list[i]])
        #sns.scatterplot(x = "Pred_Rank", y = "Exp_Rank", ax = axs, size = "Predicted_RLU", sizes =(1.5, 12), data = control.loc[control['Cell_Type'] == cell_list[i]], color = 'r')
        axs.plot([2, 10], [0, 100], linestyle = 'dotted', color = 'r') #Ideal line
        #axs.invert_xaxis()
        axs.invert_yaxis()

        #format y label
        if i > 0:
            axs.set(ylabel = None)

        #format legend
        if i < len(cell_list)-1:
            axs.legend([],[], frameon=False)
        else:
            axs.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # plt.tight_layout()
    # plt.show()
    plt.savefig(plot_save_path + f'{model}_exprank_vs_RLU.png', bbox_inches = 'tight')

def plot_ind_pred_rank(df, model):
    cell_list = df["Cell_Type"].unique()
    fig, axes = plt.subplots(1 , len(cell_list), sharex = True, sharey= True, figsize = (15,5))

    experimental = df.loc[df['Type'] == "Experimental"]
    control = df.loc[df['Type'] == "Control"]

    fig.suptitle(f'{model}_Validation')
    for i, axs in enumerate(axes):
        #sns.regplot(x = "Experimental_RLU", y = "Exp_Rank", ax = axs, scatter = False, data = experimental.loc[experimental['Cell_Type'] == cell_list[i]]).set_title(f'{cell_list[i]}')
        sns.scatterplot(x = "Pred_Rank", y = "Experimental_RLU", ax = axs, hue = "Helper_lipid", style = 'Type', markers = ["o", "X"], data = df.loc[df['Cell_Type'] == cell_list[i]])
        #sns.scatterplot(x = "Pred_Rank", y = "Exp_Rank", ax = axs, size = "Predicted_RLU", sizes =(1.5, 12), data = control.loc[control['Cell_Type'] == cell_list[i]], color = 'r')
        axs.plot([0, 100],[2, 12],  linestyle = 'dotted', color = 'r') #Ideal line
        axs.invert_xaxis()
        #axs.invert_yaxis()

        #format y label
        if i > 0:
            axs.set(ylabel = None)

        #format legend
        if i < len(cell_list)-1:
            axs.legend([],[], frameon=False)
        else:
            axs.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # plt.tight_layout()
    # plt.show()
    plt.savefig(plot_save_path + f'{model}_predrank_vs_RLU.png', bbox_inches = 'tight')


def plot_combined_scatter(df, model):
    plt.figure(figsize = (5,5))
    plt.xlim(0,13)
    plt.ylim(0,13)
    
    sns.lmplot(x = "Predicted_RLU", y = "Experimental_RLU", hue = "Cell_Type", data = df).set(title = f'{model}_Validation')
    sns.scatterplot(x = "Predicted_RLU", y = "Experimental_RLU", data = df.loc[df["Type"] == "Control"], color = 'r')
    plt.plot([0, 13], [0, 13], linestyle = 'dotted', color = 'r') #Ideal line
    
    plt.savefig(plot_save_path + f'{model}_combined_scatter.png', bbox_inches = 'tight')


def evaluate_pred_quality(exp, pred):
    #evaluate accuracy
    acc = mean_absolute_error(exp, pred)
    spearmans_rank = stats.spearmanr(exp, pred)
    pearsons_r = stats.pearsonr(exp, pred)

    return acc, spearmans_rank[0], pearsons_r[0]


################ INPUT PARAMETERS ############################################
plot_save_path = "Figures/Experimental_Validation/20230319/"
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
    cell_type_list = ['HepG2', 'PC3', 'B16']
    model_list = ['LGBM']
    control_list = [473, 470, 471, 210, 663, 422]

    for model in model_list:
        all_data = pd.DataFrame()
        RLU_pearson_list = []
        RLU_spearman_list = []
        RLU_MAE_list = []
        rank_pearson_list = []
        rank_spearman_list = []
        rank_MAE_list = []
        controls = []
        for cell in cell_type_list:
            df = pd.read_csv(f'Validation_Data/20230319/20230319_{model}_{cell}.csv' )
            df["Cell_Type"] = cell
            df["Type"] = "Experimental"
            df.loc[df["Formula_label"].isin(control_list), "Type"] = "Control" #Label control groups

            df["Pred_Rank"] = df["Predicted_RLU"].rank(ascending = False) #Add ranking
            df["Exp_Rank"] = df["Experimental_RLU"].rank(ascending = False) #Add ranking

            RLU_acc, RLU_spearman, RLU_pearson = evaluate_pred_quality(df.loc[df['Type'] == "Experimental", "Experimental_RLU"], df.loc[df['Type'] == "Experimental", "Predicted_RLU"])
            RLU_MAE_list.append(RLU_acc)
            RLU_pearson_list.append(RLU_pearson)
            RLU_spearman_list.append(RLU_spearman)

            rank_acc, rank_spearman, rank_pearson = evaluate_pred_quality(df.loc[df['Type'] == "Experimental", "Exp_Rank"], df.loc[df['Type'] == "Experimental", "Pred_Rank"])
            rank_MAE_list.append(rank_acc)
            rank_pearson_list.append(rank_pearson)
            rank_spearman_list.append(rank_spearman)

            all_data = pd.concat([all_data, df], ignore_index = True)
            
        plot_ind_scatter(all_data,model, RLU_pearson_list, RLU_spearman_list) #Individual scatter
        plot_ind_scatter_rank(all_data, model, rank_pearson_list, rank_spearman_list) #individual ranked scatter
        plot_ind_exp_rank(all_data, model) #Individual Experimental RLU vs Experimental Rank
        plot_ind_pred_rank(all_data, model) #Individual Experimental RLU vs Predicted Rank
        
        for feature in formulation_param_names:
            plot_ind_scatter_feature(all_data,model, RLU_pearson_list, RLU_spearman_list, feature)

        plot_combined_scatter(all_data,  model) #Combined Scatter



if __name__ == "__main__":
    main()
