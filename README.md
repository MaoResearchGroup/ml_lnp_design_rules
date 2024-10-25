# ml_lnp_design_rules
**Machine Learning Pipeline to predict LNP transfection efficiency and Analyze LNP design rules**

[Manuscript](https://pubs.acs.org/doi/full/10.1021/acsnano.4c07615#)

![ML LNP TOC Graphic_Cells (2)](https://github.com/user-attachments/assets/e1208206-9ebe-4319-aeca-0d5b7fd6cce6)


**Structure of Repository and Code:**

* Raw_Data directory contains relevant datasets used for model training and validation
  
* Runs directory contains different pipeline runs (using different datasets)
    * "Final_HL_Features_PDI1_RLU1.5_SIZE10000" run provides the trained models (only B16F10 cell type) and figures shown in main text of the manuscript. 

    * "example_HL_Features_PDI1_RLU1.5_SIZE10000" run provides examples of pipeline outputs for model selection, feature reduction, model diagnostics, and SHAP values. Also see note below:

    * NOTE: For model selection, random states of outer cross-validation loops have been set for accurate model comparisions, however, random states of inner cross-validation loops have not been set leading to the optimization of different model architectures. These random states can be set to improve reproducibility. 
      * Thus, new runs of the pipeline with the provided dataset will lead to slightly differing downstream results (such as feature refinement and SHAP values) than presented on the manuscript. 
      * Importantly, SHAP values for compositional features ('NP_ratio','PEG_(Chol+PEG)','(IL+HL)','HL_(IL+HL)') remain generally consistent no matter the random_state. 
      * On the other hand, helper lipid chemical feature refinement and analysis produce more variable results due to small sample size (only 6 helper lipids tested), thus less weight should be placed on chemical feature results until expanded lipid chemical libraries are tested. 
  
* Within each run there are subdirectories for related results and figures for models trained on each individual cell type datasets and cell-wise comparison figures folder.
  
  
* Each cell type directory contains directories related to different aspects of the ML pipeline (see below):

  * Trained_models: Model hyperparameter tuning and trained models

  * Model_diagnostics: contains HL-1 (leave-one-lipid-out or helper lipid minus one analysis of optimized models), learning_curve, and straw models results
  
  * Feature_Reduction : training data and results for feature reduction/refinement procedures
  
  * SHAP_Values: Calculated SHAP values for feature importance and design rules
        
  * Figures: Contains most figures used in manuscript and group by relevance
      
      

**To run code:**

  * ML_LNP.yml provided to set up conda enviroment. This code has been tested on Windows10.
  
  * run_pipeline.py runs the machine learning pipeline on a provided dataset in the Raw_Data directory. Change run parameters as needed at the top of the main function. Pipelines will be saved in the Runs folder.
  
  * plot_Pipeline.py generates plots for the pipeline. Change plotting parameters as needed at the top of the main function. Figures will be saved in the respective Runs folder.
  
  * validate_predictions.py provides ML transfection efficiency predictions of given novel LNPs (provided in the Raw_Data folder) and generate model performance metrics when comparing against experimental values.


Formatting/basic preprocessing of training data:

  * Provide training data in a csv workbook where columns represent features and rows represent unique datapoints (see example) and stored in the "Raw_Data" directory

  * The first row of each column is used as the feature name. If altered, feature names must be updated within the scripts, namely select_input_params() within utilities.py.

  * Output parameter should be labeled by the target_prefix (e.g. "RLU_" in example code) concatenated with provided the cell name (e.g. "B16" in example code).
    * If luciferase readings are used as the target output parameter, raw luciferase readings should be preprocessed using a log transformation (e.g. natural log was used in manuscript).


Model validation in silico and in vitro
  * Performance of top models were validated in silico on a stratified hold-out dataset, which consisted of 15% of all training data that was never used for model optimization or tuning. The hold out data set was stratified to include representative populations of training data from each helper lipid class

  * Model performance was further validated in vitro by creating a new library of formulations (n = 72) using the same 6 helper lipids within the training data, but altering compositional parameters. The compositional parameters were varied such that each individual composition parameter value was absent from the training data. 