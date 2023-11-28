import pickle
from scipy import stats
from copy import deepcopy
import pandas as pd
import plotting_functions as plotter
import numpy as np
from sklearn.metrics import mean_absolute_error
from itertools import chain

def main():
    
    #Load Pipeline of interest
    cell = 'B16'
    RUN_NAME  = f"Runs/Final_PDI1_RLU2/"
    with open(RUN_NAME + f'{cell}/Pipeline_dict.pkl', 'rb') as file:
                                pipeline = pickle.load(file)

    #Load Model
    prefix = pipeline['Data_preprocessing']['prefix']
    trained_model = deepcopy(pipeline['Model_Selection']['Best_Model']['Model'])
    input_params = pipeline['Feature_Reduction']['Refined_Params'].copy()
    scaler = deepcopy(pipeline['Data_preprocessing']['Scaler'])

    #Train model with refined features
    X = pipeline['Feature_Reduction']['Refined_X']
    y = pipeline['Data_preprocessing']['y']

    X = X[input_params].copy()
    trained_model.fit(X,np.ravel(y))


    #Load Dataset to predict
    validation_path = RUN_NAME + "In_Vitro_Validation/"
    validation_df = pd.read_csv(validation_path +"In_Vitro_Validation_List.csv")

    X_test = validation_df[input_params]
    
    y_test = validation_df[prefix + cell].to_numpy()

    y_test = scaler.transform(y_test.reshape(-1,1))

    y_test = list(chain(*y_test))
    #Get Predictions
    y_pred = trained_model.predict(X_test)

    df_pred =  pd.DataFrame(y_pred, columns = ['Predicted'])
    df_test = pd.DataFrame(y_test, columns = ['Experimental'])

    #Calculate MAE
    MAE = mean_absolute_error(y_pred,y_test)
    print(MAE)

    #Plot
    plotter.plot_predictions(pipeline=pipeline,
                             save =validation_path,
                             pred = y_pred,
                             exp = y_test)
    
if __name__ == "__main__":
    main()