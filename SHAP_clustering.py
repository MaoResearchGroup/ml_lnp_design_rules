# Set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image
import utilities
def SHAP_cluster(projections, color_values, title, figure_save_path, fmin, fmax):
    # Dimensionality Reduction: SHAP values on Transfection Efficiency
    f = plt.figure(figsize=(4,4))
    plt.scatter(projections[:,0],
            projections[:,1],
            #c=shap_values.values.sum(1).astype(np.float64),
            c = color_values,
            linewidth=0.1, alpha=0.8, edgecolor='black', cmap='viridis_r')
    plt.title(f"{title}")
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

def main(cell_model_list, shap_value_path, model_save_path,  figure_save_path):

##################### Run Predictions ###############################d

  # try with all cell types and model types
    for cell_model in cell_model_list:
        c = cell_model[0]
        model_name = cell_model[1]

        #Check Save path
        if os.path.exists(figure_save_path) == False:
            os.makedirs(figure_save_path, 0o666)

        #Extract actual transfection data
        with open(model_save_path + f"{c}/{model_name}_{c}_output.pkl", 'rb') as file: # import trained model
                true_Y = pickle.load(file)
        #SHAP Values
        with open(shap_value_path + f"{model_name}_{c}_SHAP_values.pkl", "rb") as file:   # Unpickling
            shap_values = pickle.load(file)
        X = shap_values.data

        #Extract input parameters for each cell type
        with open(model_save_path + f"{c}/{model_name}_{c}_Best_Model_Results.pkl", 'rb') as file: # import trained model
                best_results = pickle.load(file)
        input_params = best_results.loc['Feature names'][0]
        print(input_params)

        shap_pca50 = PCA(n_components=len(input_params)).fit_transform(shap_values.values)
        shap_embedded = TSNE(n_components=2, perplexity=50, random_state = 0).fit_transform(shap_values.values)

        SHAP_cluster(projections=shap_embedded,
                     color_values=true_Y.values,
                     title = "Normalized Experimental Transfection Efficiency",
                     figure_save_path= figure_save_path + f'{model_name}_{c}_SHAP_tfxn.svg',
                     fmin = 0,
                     fmax = 1)
        
        for i in range(len(input_params)):
            print(f"Feature: {input_params[i]}")
            norm_feature = (X[:,i]-X[:,i].min())/(X[:,i].max()-X[:,i].min())
            SHAP_cluster(projections=shap_embedded,
                     color_values=norm_feature,
                     title = input_params[i],
                     figure_save_path= figure_save_path + f'{model_name}_{c}_SHAP_{input_params[i]}.svg',
                     fmin = X[:,i].min(),
                     fmax = X[:,i].max())

            # f = plt.figure(figsize=(5,5))
            # plt.scatter(shap_embedded[:,0],
            #         shap_embedded[:,1],
            #         c=norm_feature,
            #         linewidth=0.1, alpha=1.0, edgecolor='black', cmap='viridis')

            # cb = plt.colorbar(label=None, aspect=40, orientation="horizontal")
            # cb.set_alpha(1)
            # cb.draw_all()
            # cb.outline.set_linewidth(0.2)
            # cb.ax.tick_params('x', length=2)
            # cb.ax.xaxis.set_label_position('top')
            # cb.ax.get_title()
            # cb.ax.set_title(label=input_params[i], fontsize=10, color="black", weight="bold")
            # cb.set_ticks([0.1, 0.9])
            # cb.set_ticklabels(['Low', 'High'])
            # cb.ax.set_xticklabels(cb.ax.get_xticklabels(), fontsize=10, color="black", weight="bold")

            # plt.gca().axis("off") # no axis
            # #plt.gca().axis("on") # with axis

            # # Title
            # plt.gca().axes.set_title(str(input_params[i]),
            #                         fontsize=12, color="black", weight="bold")

            # # y-axis limits and interval
            # plt.gca().set(ylim=(-40, 40), yticks=np.arange(-50, 50, 10))
            # plt.gca().set(xlim=(-40, 60), xticks=np.arange(-50, 70, 10))
            # plt.savefig(figure_save_path + f'{model_name}_{c}_SHAP_{input_params[i]}.svg', dpi = 600, transparent = True, bbox_inches = 'tight')
            # plt.close()

if __name__ == "__main__":
    main()