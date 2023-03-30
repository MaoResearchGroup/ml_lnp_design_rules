from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import KFold

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
#from ngboost import NGBRegressor

def init_data(filepath, cell_type_names):
     """ Takes a full file path to a spreadsheet and an array of cell type names. 
     Returns a dataframe with 0s replaced with 1s."""
     df = pd.read_csv(filepath)
     for cell_type in cell_type_names:
        df["RLU_" + cell_type].replace(0, 1, inplace= True) #Replace 0 transfection with 1
     return df
def apply_minmaxscaling(X, y):
  scaler = MinMaxScaler()
  scaler.fit(y.reshape(-1,1))
  temp = scaler.transform(y.reshape(-1,1))
  y = temp.flatten()
  return X, y, scaler

def init_training_data(df, cell):
  training_data = df.loc[:,df.columns.isin(['Formula label', 'Helper_lipid'] + input_param_names + ['RLU_'+ cell])]
  formatted_training_data = training_data.dropna()
  formatted_training_data.reset_index(drop = True, inplace=True)
  training_data = formatted_training_data

  X = training_data.loc[:, training_data.columns.isin(input_param_names)]                         
  y = training_data['RLU_' + cell].to_numpy()
  scaled_X, scaled_y, scaler = apply_minmaxscaling(X, y)

  return scaled_X, scaled_y, scaler
def get_best_estimator(model_type, best_params):
    if model_type == 'MLR':
        model = LinearRegression()

    elif model_type == 'lasso':
        model = linear_model.Lasso()

    elif model_type == 'kNN':
        model = KNeighborsRegressor()
    elif model_type == 'PLS':
        model = PLSRegression()

    elif model_type == 'SVR':
        model = SVR()
    
    elif model_type == 'DT':
        model = DecisionTreeRegressor(random_state=4)
    
    elif model_type == 'RF':
        model = RandomForestRegressor(random_state=4)
        

    elif model_type == 'LGBM':
        model = LGBMRegressor(random_state=4)
    
    elif model_type == 'XGB':
        model = XGBRegressor(objective ='reg:squarederror')              

    # elif model_type == 'NGB':
    #   b1 = DecisionTreeRegressor(criterion='squared_error', max_depth=2)
    #   b2 = DecisionTreeRegressor(criterion='squared_error', max_depth=4)
    #   b3 = DecisionTreeRegressor(criterion='squared_error', max_depth=8) 
    #   b4 = DecisionTreeRegressor(criterion='squared_error', max_depth=12)
    #   b5 = DecisionTreeRegressor(criterion='squared_error', max_depth=16)
    #   b6 = DecisionTreeRegressor(criterion='squared_error', max_depth=32) 
    #   self.user_defined_model = NGBRegressor()
    #   self.p_grid ={'n_estimators':[100,200,300,400,500,600,800],
    #                 'learning_rate': [0.1, 0.01, 0.001],y
    #                 'minibatch_frac': [1.0, 0.8, 0.5],
    #                 'col_sample': [1, 0.8, 0.5],
    #                 'Base': [b1, b2, b3, b4, b5, b6]}
    
    else:
        print("#######################\nSELECTION UNAVAILABLE!\n#######################\n\nPlease chose one of the following options:\n\n 'MLR'for multiple linear regression\n\n 'lasso' for multiple linear regression with east absolute shrinkage and selection operator (lasso)\n\n 'kNN'for k-Nearest Neighbors\n\n 'PLS' for partial least squares\n\n 'SVR' for support vertor regressor\n\n 'DT' for decision tree\n\n 'RF' for random forest\n\n 'LGBM' for LightGBM\n\n 'XGB' for XGBoost\n\n 'NGB' for NGBoost")

    best_estimator = model.set_params(**best_params)
    print(best_estimator)

    return best_estimator
### Global Variables 
################ Retreive/Store Data ##############################################
datafile_path = "Raw_Data/7_Master_Formulas.csv"
model_path = 'Trained_Models/Final_Models/'
save_path = "Figures/Training_size/"
################ INPUT PARAMETERS ############################################

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


### SET PARAMS OF MODEL TO BEST FROM OPTIMIATION SET

## best_model = self.user_defined_model.set_params(**best_model_params)


"""**MAIN**"""
def main():
##################### Run ###############################
    cell_type_list = ['HEK293', 'HepG2', 'N2a', 'B16', 'PC3', 'ARPE19']
    model_list = ['RF', 'XGB']
    df = init_data(datafile_path, cell_type_list)


#################### MODEL TRAINING AND EVALUATION ##############
#   for model_name in model_list:
#     for cell_type in cell_type_list:
#         num_splits = 5
#         X, y, scaler = init_training_data(df, cell_type)
#         train_size= np.linspace(0.01, 1, 100)*len(X)*(num_splits-1)/num_splits
#         train_size = np.floor(train_size).astype(int)
#         train_scores_mean = pd.DataFrame(index=train_size)
#         validation_scores_mean = pd.DataFrame(index=train_size)
#         print(f"\n############ Creating plot for : {model_name}_{cell_type} ###############")
        
#         with open(model_path +f'{model_name}/{cell_type}/{model_name}_HP_Tuning_Results.pkl', 'rb') as file: # 
#             training_results = pickle.load(file)
#         best_params = training_results.iloc[0,5]

#         model = get_best_estimator(model_name, best_params)
        
#         for i in range(10):
#             cross_val = KFold(n_splits= num_splits, random_state= i, shuffle=True)
#             train_sizes, train_scores, validation_scores = learning_curve(estimator = model, X = X, y = y, cv = cross_val, train_sizes= train_size,
#                                                                         scoring = 'neg_mean_absolute_error', n_jobs= -1)
#             # print('Training scores:\n\n', train_scores)
#             # print('\n', '-' * 70) # separator to make the output easy to read
#             # print('\nValidation scores:\n\n', validation_scores)

#             train_scores_mean[i] = -train_scores.mean(axis = 1)
#             validation_scores_mean[i] = -validation_scores.mean(axis = 1)
#             # print('Mean training scores\n\n', train_scores_mean)
#             # print('\n', '-' * 20) # separator
#             # print('\nMean validation scores\n\n',validation_scores_mean)
    

#         #Save Data
#         with open(save_path + f'{model_name}_{cell_type}_training_results.csv', 'w', encoding = 'utf-8-sig') as f:
#             train_scores_mean.to_csv(f)

#         with open(save_path + f'{model_name}_{cell_type}_val_results.csv', 'w', encoding = 'utf-8-sig') as f:
#             validation_scores_mean.to_csv(f)


    ############## LOAD DATA AND PLOTTING #####################
    NUM_ITER = 10
    #Initialize Data
    all_data = pd.DataFrame()
    for model_name in model_list:
        for cell_type in cell_type_list:
            mean_train_scores = pd.read_csv(save_path + f'{model_name}_{cell_type}_training_results.csv' )
            mean_train_scores.rename(columns={mean_train_scores.columns[0]: "Train_size" }, inplace = True)
            mean_train_scores["Score_Type"] = "Train"
            mean_train_scores["Mean_MAE"] = mean_train_scores.iloc[:, range(1,NUM_ITER+1)].mean(axis=1)
            mean_train_scores["sd"] = mean_train_scores.iloc[:, range(1,NUM_ITER+1)].std(axis=1)
        
            mean_validation_scores = pd.read_csv(save_path + f'{model_name}_{cell_type}_val_results.csv' )
            mean_validation_scores.rename(columns={mean_validation_scores.columns[0]: "Train_size" }, inplace = True)
            mean_validation_scores["Score_Type"] = "Validation"
            mean_validation_scores["Mean_MAE"] = mean_validation_scores.iloc[:, range(1,NUM_ITER+1)].mean(axis=1)
            mean_validation_scores["sd"] = mean_validation_scores.iloc[:, range(1,NUM_ITER+1)].std(axis=1)
      
            train_valid_scores = pd.concat([mean_validation_scores, mean_train_scores], ignore_index= True)
            train_valid_scores["Cell_Type"] = cell_type
            train_valid_scores["Model_Type"] = model_name
            all_data = pd.concat([all_data, train_valid_scores], ignore_index=True)

    ###### Plotting
    for model_name in model_list:
        fig, axes = plt.subplots(2,3, sharex = True, sharey = True, figsize = (18, 12))
        plt.ylim(0, 0.25)
        plt.xlim(0, 900)
        fig.suptitle(f'{model_name}_Train_size', fontsize = 18) 
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'HEK293')],
                      x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[0,0]).set(title ='HEK293')
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'B16')],
                      x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[0,1]).set(title ='B16')
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'HepG2')],
                      x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[0,2]).set(title ='HepG2')
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'N2a')],
                      x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[1,0]).set(title ='N2a')        
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'PC3')],
                      x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[1,1]).set(title ='PC3') 
        sns.lineplot(data = all_data.loc[(all_data['Model_Type'] == model_name) & (all_data['Cell_Type'] == 'ARPE19')],
                      x = "Train_size", y = "Mean_MAE", hue = "Score_Type", errorbar = "sd", ax = axes[1,2]).set(title ='ARPE19')
        
        plt.savefig(save_path + f'{model_name}_Train_size.png', bbox_inches = 'tight')


if __name__ == "__main__":
    main()