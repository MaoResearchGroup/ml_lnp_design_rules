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
datafile_path = "Raw_Data/7_Master_Formulas.csv"
cell_type_list = ["HEK293", "B16", "HepG2", "PC3"] #Only cell types with all 1080 datapoints
"""**MAIN**"""
def main():
#Extract Training Data
    df = pd.read_csv(datafile_path)
    print(df)
    X = df.loc[:,df.columns.isin("RLU_" + x for x  in cell_type_list)] #Take input parameters and associated values
    X.head()
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

 



if __name__ == "__main__":
    main()