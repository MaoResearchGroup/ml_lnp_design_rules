# ml_lnp_design_rules
Machine Learning Pipeline to predict LNP transfection efficiency and Analyze LNP design rules

![alt text](https://github.com/MaoResearchGroup/ml_lnp_design_rules/main/ML_LNP_TOC_Graphic_Cells.png?raw=true)

Structure of Repository and Code:

* Raw_Data directory contains relevant datasets used for model training and validation
  
* Runs directory contains different pipeline runs (using different datasets)
  
* Within each run there are subdirectories for related results for models trained on each individual cell type datasets
  
* Each cell type directory contains directories related to different aspects of the ML pipeline (see below):

  * Trained_models: Model hyperparameter tuning and trained models
  
  * HL_1 : leave-one-lipid-out or helper lipid minus one analysis of optimized models
  
  * Feature_Reduction : training data and results for feature reduction/refinement procedures
  
  * SHAP_Values: Calculated SHAP values for feature importance and design rules

  * Straw_Models: Straw model analysis of selected models.
        
  * Figures: Contains most figures used in manuscript and group by relevance
      
      

To run code:

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