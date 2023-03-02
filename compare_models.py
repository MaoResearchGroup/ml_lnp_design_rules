# Set up
import pickle

# import the necessary libraries to execute this code
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from scipy import stats
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV as RSCV
from statistics import mean 

# import model frameworks
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def apply_minmaxscaling(X, y):
  scaler = MinMaxScaler()
  scaler.fit(y.reshape(-1,1))
  temp = scaler.transform(y.reshape(-1,1))
  y = temp.flatten()

  return X, y, scaler
def init_data(filepath, cell_type_names):
     """ Takes a full file path to a spreadsheet and an array of cell type names. 
     Returns a dataframe with 0s replaced with 1s."""
     df = pd.read_csv(filepath)
     for cell_type in cell_type_names:
        df["RLU_" + cell_type].replace(0, 1, inplace= True) #Replace 0 transfection with 1
     return df

def init_training_data(df, cell):
  training_data = df.loc[:,df.columns.isin(['Formula label', 'Helper_lipid'] + input_param_names + ['RLU_'+ cell])]
  formatted_training_data = training_data.dropna()
  formatted_training_data.reset_index(drop = True, inplace=True)
  training_data = formatted_training_data

  X = training_data.loc[:, training_data.columns.isin(input_param_names)]                         
  y = training_data['RLU_' + cell].to_numpy()
  scaled_X, scaled_y, scaler = apply_minmaxscaling(X, y)

  return scaled_X, scaled_y, scaler

def write_df_to_sheets(data_df, save_path):
    
  with open(save_path, 'w', encoding = 'utf-8-sig') as f:
    data_df.to_csv(f)
################ Global Variables ##############################################
datafile_path = "Raw_Data/7_Master_Formulas.csv"
model_path = 'Trained_Models/Final_Models/'
save_path = 'Compare_Models/'
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
    cell_type_model = ['HEK293', 'HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']
    cell_type_test = ['HEK293', 'HepG2', 'N2a', 'ARPE19', 'B16', 'PC3']
    model_list = ['RF', 'XGB', 'LGBM']
    df = init_data(datafile_path, cell_type_test)


    #Load and retrain model for each cell type and run 5-fold Cross validation evaluation.
    #Run for each model
    #Storage Lists
    model_used = []
    model_cell_used = []
    model_params = []
    cell_tested = []
    MAE = []
    spearman = []
    pearson = []
    for model_name in model_list:
        #Retrain and evaluate for each cell_type model
        for model_cell in cell_type_model:
            #Get Optimized and Trained ML Model for that cell_type
            with open(model_path +f'{model_name}/{model_cell}/{model_name}_{model_cell}_Trained.pkl', 'rb') as file: # import trained model
                model = pickle.load(file)
            
            #reevaluate model predictions for each cell_type
            for test_cell in cell_type_test:
                print(f'############ Evaluating: {model_cell}_{model_name} for {test_cell} ###############')

                cell_MAE = []
                cell_spearman = []
                cell_pearson = []
                #Get Training Data
                X, y, scaler = init_training_data(df, test_cell)


                #Cross-Validation Evaluation
                cv = KFold(n_splits=5, random_state= 0, shuffle=True)
                for i, (train_index, test_index) in enumerate(cv.split(X)):
                    # #X = input parameters, y = transfection, F = formulation number
                    X_train = X.iloc[train_index]
                    X_test = X.iloc[test_index]
                    y_train = y[train_index]
                    y_test = y[test_index]

                    #Retrain Model and predict
                    model.fit(X_train, y_train)
                    yhat = model.predict(X_test)

                    #evaluate accuracy
                    acc = mean_absolute_error(y_test, yhat)
                    spearmans_rank = stats.spearmanr(y_test, yhat)
                    pearsons_r = stats.pearsonr(y_test, yhat)

                    #Store individual Results
                    cell_MAE.append(acc)
                    cell_spearman.append(spearmans_rank[0])
                    cell_pearson.append(pearsons_r[0])

                #Store Average Results
                model_used.append(model_name)
                model_cell_used.append(model_cell)
                model_params.append(model.get_params())
                cell_tested.append(test_cell)
                MAE.append(mean(cell_MAE))
                spearman.append(mean(cell_spearman))
                pearson.append(mean(cell_pearson))

    #Save Data into CSV
    #create dataframe with results of nested CV
    list_of_tuples = list(zip(model_used, model_cell_used, model_params, cell_tested, MAE, spearman, pearson))
    CV_results = pd.DataFrame(list_of_tuples, columns = ['Model',
                                                         'Optimized_Cell_Type',
                                                         'Model Params',
                                                         'Cell_Type_Tested',
                                                         'MAE',
                                                         'Spearmans Rank',
                                                         'Pearsons Correlation'])
    write_df_to_sheets(CV_results, save_path + "Model_HP_Comparison.csv")

            

    
    # for model in model_list:
    #     for cell in cell_type:
    #         result_file_path = result_folder + f'{model}/{cell}/{model}_HP_Tuning_Results.pkl'
    #         with open(result_file_path, 'rb') as file:
    #             results = pickle.load(file)
    #             results.drop(columns = ['Iter','Formulation_Index'], inplace = True)
    #             results = results.iloc[[0]] #keep only Best model, return dataframe type
    #             results.insert(0, 'Model', model) #Add model
    #             results.insert(1, 'Cell_Type', cell) #Add cell type
    #             all_results = pd.concat([results, all_results.loc[:]], ignore_index = True).reset_index(drop = True)

    # #Save results
    # with open(result_folder + "Model_Selection_Results.csv", 'w', encoding = 'utf-8-sig') as f:
    #     all_results.to_csv(f)
    # print('Saved Results')
        
    # ########## Extract Results ##################
    # MAE_results = pd.DataFrame(index = model_list, columns = cell_type)
    # spearman_results = pd.DataFrame(index = model_list, columns = cell_type)
    # pearson_results = pd.DataFrame(index = model_list, columns = cell_type)
    # pred_transfection = pd.DataFrame(index = model_list, columns = cell_type)
    # exp_transfection = pd.DataFrame(index = model_list, columns = cell_type)
    # for model in model_list:
    #     for cell in cell_type:
    #         m1 = all_results["Model"] == model
    #         m2 = all_results["Cell_Type"] == cell
    #         MAE_results.at[model, cell] = all_results[m1&m2]['Test Score'].values[0]
    #         spearman_results.at[model, cell] = all_results[m1&m2]['Spearmans Rank'].values[0][0]
    #         pearson_results.at[model, cell] = all_results[m1&m2]['Pearsons Correlation'].values[0][0]
    #         pred_transfection.at[model, cell] = all_results[m1&m2]['Predicted_Transfection'].values[0]
    #         exp_transfection.at[model, cell] = all_results[m1&m2]['Experimental_Transfection'].values[0].transpose()[0] #Format as list
    
    # ########## Tabulate Results ##################
    # with open(result_folder + "Model_Selection_MAE.csv", 'w', encoding = 'utf-8-sig') as f:
    #     MAE_results.to_csv(f)
    # with open(result_folder + "Model_Selection_spearman.csv", 'w', encoding = 'utf-8-sig') as f:
    #     spearman_results.to_csv(f)
    # with open(result_folder + "Model_Selection_pearson.csv", 'w', encoding = 'utf-8-sig') as f:
    #     pearson_results.to_csv(f)   
    


if __name__ == "__main__":
    main()