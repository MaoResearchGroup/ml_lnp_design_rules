import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns


################ Global Variables ##############################################
datafile_path = "Raw_Data/8_Master_Formulas.csv"
wt_percent = False
size_zeta = False
if wt_percent == True:
  formulation_param_names = ['wt_Helper', 'wt_Dlin','wt_Chol', 'wt_DMG', 'wt_pDNA']
else:
  formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                      'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio']
helper_lipid_names = ['18PG', 'DOPE','DOTAP','DSPC', '14PA', 'DDAB']


lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA',
                      'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
if size_zeta == True:
    input_param_names = lipid_param_names +  formulation_param_names + ['Size', 'Zeta']
else:
    input_param_names = lipid_param_names +  formulation_param_names 

"""**MAIN**"""
def main():
#Extract Training Data
    df = pd.read_csv(datafile_path)
          
    #Formatting Training Data
    X = df[input_param_names]
    X = X.dropna() #Remove any NaN rows
    if size_zeta == True:
        X = X[X.Size != 0] #Remove any rows where size = 0
        X  = X[X.Zeta != 0] #Remove any rows where zeta = 0

    X.reset_index(drop = True, inplace=True)
    print(X)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    corr = spearmanr(X).correlation # generate a correlation matrix is symmetric
    corr = (corr + corr.T) / 2 # ensure the correlation matrix is symmetric
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr) # convert the correlation matrix to a distance matrix 
    dist_linkage = hierarchy.ward(squareform(distance_matrix)) # generate Ward's linkage values for hierarchical clustering

    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X.columns.tolist(), ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax1.set_title("Hierarchical Clustering (Ward-linkage)", fontsize=14, color="black", weight="bold")
    ax1.set_xlabel('FEATURE NAMES', fontsize=14, color="black")
    ax1.set_ylabel('HEIGHT', fontsize=14, color="black")
    ax1.tick_params(axis='y', which='both', labelsize=12)
    ax1.tick_params(axis='x', which='both', labelsize=12)


    im = ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]], 
                    alpha = 1.0)
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    ax2.figure.colorbar(im, format='% .2f')
    ax2.tick_params(axis='y', which='both', labelsize=12)
    ax2.tick_params(axis='x', which='both', labelsize=12)
    ax2.set_title("Spearman's Rank Correlation", fontsize=14, color="black", weight="bold")
    fig.tight_layout()

    #plt.savefig('drive/My Drive/RF_14feature_corr&cluster', dpi=600, format = 'png', transparent=True, bbox_inches='tight')

    plt.show()


    # corr_data = pd.read_csv(datafile_path)
    corr_X = X
    correlations = corr_X.corr()
    sns.heatmap(round(np.abs(correlations),2), annot=True, 
                annot_kws={"size": 7}, vmin=0, vmax=1);



    plt.figure(figsize=(12,5))
    dissimilarity = 1 - abs(correlations)
    Z = linkage(squareform(dissimilarity), 'complete')

    dendrogram(Z, labels=corr_X.columns, orientation='top', 
              leaf_rotation=90);

    # Clusterize the data
    threshold = 0.8
    labels = fcluster(Z, threshold, criterion='distance')

    # Show the cluster
    labels
    # Keep the indices to sort labels
    labels_order = np.argsort(labels)

    # Build a new dataframe with the sorted columns
    for idx, i in enumerate(corr_X.columns[labels_order]):
        if idx == 0:
            clustered = pd.DataFrame(corr_X[i])
        else:
            df_to_append = pd.DataFrame(corr_X[i])
            clustered = pd.concat([clustered, df_to_append], axis=1)

    plt.figure(figsize=(10,10))
    correlations = clustered.corr()
    plot = sns.heatmap(round(correlations,2), cmap='mako', annot=True, 
                annot_kws={"size": 7}, vmin=-1, vmax=1);



    plt.figure(figsize=(15,10))

    for idx, t in enumerate(np.arange(0.2,1.1,0.1)):
        
        # Subplot idx + 1
        plt.subplot(3, 3, idx+1)
        
        # Calculate the cluster
        labels = fcluster(Z, t, criterion='distance')

        # Keep the indices to sort labels
        labels_order = np.argsort(labels)

        # Build a new dataframe with the sorted columns
        for idx, i in enumerate(corr_X.columns[labels_order]):
            if idx == 0:
                clustered = pd.DataFrame(corr_X[i])
            else:
                df_to_append = pd.DataFrame(corr_X[i])
                clustered = pd.concat([clustered, df_to_append], axis=1)
                
        # Plot the correlation heatmap
        correlations = clustered.corr()
        sns.heatmap(round(correlations,2), cmap='RdBu', vmin=-1, vmax=1, 
                    xticklabels=False, yticklabels=False)
        plt.title("Threshold = {}".format(round(t,2)))


    my_list = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 
            'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 
            'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 
            'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 
            'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 
            'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 
            'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 
            'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 
            'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 
            'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 
            'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 
            'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 
            'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r']

    # Colors - cmap="mako", cmap="viridis", cmap="Blues", cmap='RdBu', rocket, flare, "seagreen", Reds, Magma
    for color in my_list:
        kws = dict(cbar_kws=dict(ticks=[0, 0.50, 1], orientation='horizontal'), figsize=(6, 6))

    g = sns.clustermap(round(np.abs(correlations),2), method="complete", cmap=my_list[2], annot=True, 
                  annot_kws={"size": 8}, vmin=0, vmax=1, figsize=(10,10));

    x0, _y0, _w, _h = g.cbar_pos

    g.ax_cbar.set_position([x0, 1.0, g.ax_row_dendrogram.get_position().width, 0.15])
    g.ax_cbar.set_title("Spearman's Rank Correlation")
    g.ax_cbar.tick_params(axis='x', length=10)
    for spine in g.ax_cbar.spines:
        g.ax_cbar.spines[spine].set_color('crimson')
        g.ax_cbar.spines[spine].set_linewidth(2)

    plt.tick_params(axis='y', which='both', labelsize=12)
    plt.tick_params(axis='x', which='both', labelsize=12)

    plt.tight_layout()
    plt.savefig('Figures/Clustering/Hierarchical/FN_Cluster.png', dpi=600, format = 'png', transparent=True, bbox_inches='tight')

    plt.show()
if __name__ == "__main__":
    main()