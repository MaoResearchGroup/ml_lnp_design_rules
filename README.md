# ml_lnp_design_rules
Machine Learning Pipeline to predict LNP transfection efficiency and Analyze LNP design rules

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
