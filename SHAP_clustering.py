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


def main(shap_value_path, model_save_path, cell_type, model_list, figure_save_path):




##################### Run Predictions ###############################d

  # try with all cell types and model types

  for c in cell_type:
    for model_name in model_list:

        #Check Save path
        if os.path.exists(figure_save_path) == False:
            os.makedirs(figure_save_path, 0o666)


        #SHAP Values
        with open(shap_value_path + f"{model_name}_{c}_SHAP_values.pkl", "rb") as file:   # Unpickling
            shap_values = pickle.load(file)
        X = shap_values.data
        print(X.shape)
        print(shap_values.values.shape)
        #Extract input parameters for each cell type
        with open(model_save_path + f"{c}/{model_name}_{c}_Best_Model_Results.pkl", 'rb') as file: # import trained model
                best_results = pickle.load(file)
        input_params = best_results.loc['Feature names'][0]
        print(input_params)

        shap_pca50 = PCA(n_components=len(input_params)).fit_transform(shap_values.values)
        shap_embedded = TSNE(n_components=2, perplexity=50, random_state = 0).fit_transform(shap_values.values)

        # Dimensionality Reduction: SHAP values on Transfection Efficiency
        f = plt.figure(figsize=(5,5))
        plt.scatter(shap_embedded[:,0],
                shap_embedded[:,1],
                c=shap_values.values.sum(1).astype(np.float64),
                linewidth=0.1, alpha=1.0, edgecolor='black', cmap='YlOrBr')
        plt.title('Dimensionality Reduction: SHAP values on Transfection Efficiency in ' + c)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        plt.savefig(figure_save_path + f'{model_name}_{c}_SHAP_tfxn.svg', dpi = 600, transparent = True, bbox_inches = 'tight')

        
        
        for i in range(len(input_params)-1):
            print(f"Feature: {input_params[i]}")
            norm_feature = (X[:,i]-X[:,i].min())/(X[:,i].max()-X[:,i].min())
            

            f = plt.figure(figsize=(5,5))
            plt.scatter(shap_embedded[:,0],
                    shap_embedded[:,1],
                    c=norm_feature,
                    linewidth=0.1, alpha=1.0, edgecolor='black', cmap='YlOrBr')

            cb = plt.colorbar(label=None, aspect=40, orientation="horizontal")
            cb.set_alpha(1)
            cb.draw_all()
            cb.outline.set_linewidth(0.2)
            cb.ax.tick_params('x', length=2)
            cb.ax.xaxis.set_label_position('top')
            cb.ax.get_title()
            cb.ax.set_title(label=input_params[i], fontsize=10, color="black", weight="bold")
            cb.set_ticks([0.1, 0.9])
            cb.set_ticklabels(['Low', 'High'])
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), fontsize=10, color="black", weight="bold")

            plt.gca().axis("off") # with axis
            #plt.gca().axis("on") # no axis

            # Title
            plt.gca().axes.set_title("PCA: SHAP values for "+str(input_params[i]),
                                    fontsize=12, color="black", weight="bold")
            # y-axis limits and interval
            plt.gca().set(ylim=(-70, 70), yticks=np.arange(-60, 65, 20))
            plt.gca().set(xlim=(-70, 70), xticks=np.arange(-60, 65, 20))
            plt.savefig(figure_save_path + f'{model_name}_{c}_SHAP_{input_params[i]}.svg', dpi = 600, transparent = True, bbox_inches = 'tight')


if __name__ == "__main__":
    main()