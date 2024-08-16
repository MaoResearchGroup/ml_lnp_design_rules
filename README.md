# lnp-ml
Machine Learning Pipeline to predict LNP transfection efficiency and Analyze LNP design rules


To run code:

ML_LNP.yml provided to set up conda enviroment. This code has been tested on Windows.

run_pipeline.py runs the machine learning pipeline on provided dataset in the Raw_Data directory. Change run parameters as needed at the top of the main function. Pipelines will be saved in the Runs folder.

plot_Pipeline.py generates plots for the pipeline. Change plotting parameters as needed at the top of the main function. Figures will be saved in the respective Runs folder.

validate_predictions.py provides ML transfection efficiency predictions of given novel LNPs (provided in the Raw_Data folder) and generate model performance metrics when comparing against experimental values.