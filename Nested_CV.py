"""**Hyperparameter Optimization**"""

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
import utilities
import copy

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
from sklearn.neural_network import MLPRegressor
#from ngboost import NGBRegressor




class NESTED_CV:
  
    """
    NESTED_CV Class:
    - based on a dataset for long acting injectible (LAI) drug delivey systems
    - contains 12 different model architectures and non-exaustive hyperparamater spaces for those models
    - actiavted by abbriviations for these model - incorrect keywords triggers a message with available key words
    - once model type is selected, NEST_CV will be conducted, data is spli as follows:
          - outer_loop (test) done by GroupShuffleSplit where 20% of the drug-polymer groups in the dataset are held back at random
          - inner_loop (HP screening) done by GroupKFold based 10 splits in the dataset - based on drug-polymer groups
    - default is 10-folds for the NESTED_CV, but this can be entered manually
    - prints progress and reults at the end of each loop
    - configures a pandas dataframe with the reults of the NESTED_CV
    - fits and trains the best model based on the reults of the NESTED_CV
    """

    #Functions here
    def __init__(self, model_type = None):
        
        if model_type == 'MLR':
          self.user_defined_model = LinearRegression()
          self.p_grid = {'fit_intercept':[True, False],
                         'positive':[True, False]}
    
        elif model_type == 'lasso':
          self.user_defined_model = linear_model.Lasso()
          self.p_grid = {'alpha':[0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
                        'positive':[True, False]}

        elif model_type == 'kNN':
          self.user_defined_model = KNeighborsRegressor()
          self.p_grid ={'n_neighbors':[2, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50],
                        'weights': ["uniform", 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'leaf_size': [10, 30, 50, 75, 100],
                        'p':[1, 2],
                        'metric': ['minkowski']}

        elif model_type == 'PLS':
          self.user_defined_model = PLSRegression()
          self.p_grid ={'n_components':[2, 4, 6],
                        'max_iter': [250, 500, 750, 1000]}

        elif model_type == 'SVR':
          self.user_defined_model = SVR()
          self.p_grid ={'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree':[2, 3, 4, 5, 6],
                        'gamma':['scale', 'auto'],
                        'C':[0.1, 0.5, 1, 2],
                        'epsilon':[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
                        'shrinking': [True, False]}
        
        elif model_type == 'DT':
          self.user_defined_model = DecisionTreeRegressor(random_state=4)
          self.p_grid ={'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        'splitter':['best', 'random'],
                        'max_depth':[None],
                        'min_samples_split':[2,4,6],
                        'min_samples_leaf':[1,2,4],
                        'max_features': [None, 1.0, 'sqrt','log2'],
                        'ccp_alpha': [0, 0.05, 0.1, 0.15]}  
        
        elif model_type == 'RF':
          self.user_defined_model = RandomForestRegressor(random_state=4)
          self.p_grid ={'n_estimators':[100,200,300,400],
                        'criterion':['squared_error', 'absolute_error'],
                        'max_depth': [2, 6, 10, 14, 18, 22, 26, 30],
                        'min_samples_split':[2,4,6,8],
                        'min_samples_leaf':[1,2,4],
                        'min_weight_fraction_leaf':[0.0],
                        'max_features': [None, 'sqrt'],
                        'max_leaf_nodes':[None],
                        'min_impurity_decrease': [0.0],
                        'bootstrap':[True],
                        'oob_score':[True],
                        'ccp_alpha': [0, 0.005, 0.01]}
          
        elif model_type == 'LGBM':
          self.user_defined_model = LGBMRegressor(random_state=10)
          self.p_grid ={"n_estimators":[100,150,200,250,300,400,500,600],
                        'boosting_type': ['gbdt', 'dart', 'goss'],
                        'num_leaves':[16,32,64,128,256],
                        'max_bin': [10, 20, 30],
                        'max_depth': [2, 6, 10, 14, 18, 22, 26, 30],
                        'learning_rate':[0.1,0.01,0.001,0.0001],
                        'min_child_weight': [0.001,0.01,0.1,1.0,10.0],
                        'subsample': [0.4,0.6,0.8,1.0],
                        'min_child_samples':[2,10,20,40,100],
                        'reg_alpha': [0, 0.005, 0.01, 0.015],
                        'reg_lambda': [0, 0.005, 0.01, 0.015]}
        
        elif model_type == 'XGB':
          self.user_defined_model = XGBRegressor(objective ='reg:squarederror')
          self.p_grid ={'booster': ['gbtree', 'gblinear', 'dart'],
                        "n_estimators":[100, 150, 300, 400],
                        'max_depth':[3, 4, 5, 6, 7, 8, 9, 10],
                        'gamma':[0, 2, 4, 6, 8, 10],
                        'learning_rate':[0.3, 0.2, 0.1, 0.05, 0.01],
                        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'min_child_weight': [1.0, 2.0, 4.0, 5.0],
                        'max_delta_step':[1, 2, 4, 6, 8, 10],
                        'reg_alpha':[0.001, 0.01, 0.1],
                        'reg_lambda': [0.001, 0.01, 0.1]} 
        elif model_type == 'MLP':
          self.user_defined_model = MLPRegressor()
          self.p_grid ={
                        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                        'activation': ['tanh', 'relu', 'logistic', 'identity'],
                        'solver': ['sgd', 'adam'],
                        'alpha': [0.0001,0.01, 0.05],
                        'batch_size':['auto', 64, 128],
                        'max_iter': [100, 200, 300],
                        'learning_rate': ['constant','adaptive'],
                        'tol': [1e-4, 1e-5]
                        }                
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

    def input_target(self, X, y, data):
        self.X, self.y, self.cell_data = X, y, data
        self.F = self.cell_data['Formula label'] #used to track LNPs
        self.HL = self.cell_data['Helper_lipid'] #used for stratify split
    
    def cross_validation(self, input_value):
        if input_value == None:
            NUM_TRIALS = 10
        else: 
            NUM_TRIALS = input_value

        self.itr_number = [] # create new empty list for itr number 
        self.outer_MAE = []
        self.outer_spearman = []
        self.outer_pearson = []
        self.inner_results = []
        self.model_params = []
        self.y_test_list = []
        self.F_test_list = []
        self.pred_list = []


        # Split CV loop model development set and final test set (Fully insulated from any training data)
        indices = self.X.index
        # self.X_cv_loop, self.X_final_test, self.y_cv_loop, self.y_final_test = train_test_split(self.X, self.Y, test_size= 0.15, random_state= 4, stratify= self.HL, shuffle= True)
        cv_idx, test_idx, _,_ = train_test_split(indices, self.y, test_size= 0.15, random_state= 4, stratify= self.HL, shuffle= True)
        
        self.X_cv_loop, self.X_final_test, = self.X.iloc[cv_idx], self.X.iloc[test_idx]
        self.y_cv_loop, self.y_final_test  = self.y.iloc[cv_idx], self.y.iloc[test_idx]
        self.F_cv_loop, self.F_final_test  = self.F.iloc[cv_idx], self.F.iloc[test_idx]
        # Nested CV
        cv_outer = KFold(n_splits = NUM_TRIALS , random_state= 100, shuffle=True) # Kfold split of the developemnt dataset


        for i, (train_index, test_index) in enumerate(cv_outer.split(self.X_cv_loop)):
  
          #X = input parameters, y = transfection, F = formulation number
          #X_train, X_test, y_train, y_test, F_train, F_test = train_test_split(self.X, self.Y, self.F, test_size=0.2, random_state= i) #Iterate through different random_state to randomize test_train split
          X_train = self.X_cv_loop.iloc[train_index].copy()
          X_test = self.X_cv_loop.iloc[test_index].copy()
          y_train = self.y_cv_loop.iloc[train_index].copy()
          y_test = self.y_cv_loop.iloc[test_index].copy()
          F_train = self.F_cv_loop.iloc[train_index].copy()
          F_test = self.F_cv_loop.iloc[test_index].copy()

          #store test set information
          F_test = np.array(F_test) #prevents index from being brought from dataframe
          self.F_test_list.append(F_test)
          y_test = np.array(y_test) #prevents index from being brought from dataframe
          self.y_test_list.append(y_test)
                
          # configure the cross-validation procedure - inner loop (validation set/HP optimization)
          cv_inner = KFold(n_splits = 5, shuffle = True)

          # define search space
          search = RSCV(self.user_defined_model, self.p_grid, n_iter=100, verbose=0, scoring='neg_mean_absolute_error', cv=cv_inner,  n_jobs= -1, refit=True)
                  
          # execute search
          y_train = np.ravel(y_train)
          result = search.fit(X_train, y_train)
              
          # get the best performing model fit on the whole training set
          best_model = result.best_estimator_

          # get the score for the best performing model and store
          best_score = abs(result.best_score_)
          self.inner_results.append(best_score)
                  
          #### evaluate model on the hold out dataset
          yhat = best_model.predict(X_test)

          #Cell-type transfection predictions
          self.pred_list.append(yhat)

          # evaluate the model accuracy using the hold out dataset Mean Absolute Error
          acc = mean_absolute_error(y_test, yhat)
          spearmans_rank = stats.spearmanr(y_test, yhat)
          
          y_test = np.ravel(y_test) #reformat to 1D array
          pearsons_r = stats.pearsonr(y_test, yhat)

          # store the result
          self.itr_number.append(i+1)
          self.outer_MAE.append(acc)
          self.outer_spearman.append(spearmans_rank)
          self.outer_pearson.append(pearsons_r)
          self.model_params.append(result.best_params_)

          # report progress at end of each inner loop
          #Note: Test score = outerloop - hold out - dataset score
          # Best_Valid_Score = innerloop iteration score
          print('\n################################################################\n\nSTATUS REPORT:')
          print('Iteration '+str(i+1)+' of '+str(NUM_TRIALS)+' runs completed') 
          print('Best_Valid_Score: %.3f, Hold_Out_MAE: %.3f,  Hold_Out_Spearman_Rank: %.3f, Hold_Out_Pearsons_R: %.3f, \n\nBest_Model_Params: \n%s' % (best_score, acc, spearmans_rank[0], pearsons_r[0], result.best_params_))
          print("\n################################################################\n ")
          
    def results(self):   
        #create dataframe with results of nested CV
        list_of_tuples = list(zip(self.itr_number, self.inner_results, self.outer_MAE, self.outer_spearman,self.outer_pearson, self.model_params, self.F_test_list, self.y_test_list, self.pred_list))
        CV_dataset = pd.DataFrame(list_of_tuples, columns = ['Iter', 
                                                             'Valid Score', 
                                                             'Test Score', 
                                                             'Spearmans Rank',
                                                             'Pearsons Correlation',
                                                             'Model Parms', 
                                                             'Formulation_Index', 
                                                             'Experimental_Transfection',
                                                             'Predicted_Transfection'])
        CV_dataset['Score_difference'] = abs(CV_dataset['Valid Score'] - CV_dataset['Test Score']) #Groupby dataframe model iterations that best fit the data (i.e., minimize different between validitaion and test)
        CV_dataset.sort_values(by=['Score_difference', 'Test Score'], ascending=True, inplace=True) 
        CV_dataset = CV_dataset.reset_index(drop=True) # Reset index of dataframe
        print('Cross Validation Results', CV_dataset)
        # save the results as a class object
        self.CV_dataset = CV_dataset
 ###### Retrain model using best parameters to evaluate the test set MAE
    def FINAL_TEST_MAE(self):
        # assign the best hyperparameters from NESTED CV
        self.best_model_params = self.CV_dataset.iloc[0,5]
        print('\nFinal_Best_Model_Params: \n%s' % self.best_model_params)

        # set params from the best model to a class object
        best_model = self.user_defined_model.set_params(**self.best_model_params)

        #Fit on all cv loop training data
        self.best_model = best_model.fit(self.X_cv_loop, self.y_cv_loop)

        #Predict test set
        yhat = self.best_model.predict(self.X_final_test)

        #Make Predictions dataframe for downstream plotting
        pred_df = pd.DataFrame(self.F_final_test, columns = ['Formula label']).reset_index(drop=True)
        yhat_df = pd.DataFrame(yhat, columns = ['Predicted_Transfection'])
        y_test_df = copy.copy(self.y_final_test)
        y_test_df.reset_index(inplace = True, drop = True)
        pred_df = pd.concat([pred_df, yhat_df, y_test_df], axis = 1, ignore_index=True)
        pred_df.columns = ['Formula label','Predicted_Transfection', 'Experimental_Transfection']

        #calclate performance
        AE = abs(pred_df['Predicted_Transfection'] - pred_df['Experimental_Transfection'])
        acc = mean_absolute_error(y_test_df, yhat)
        spearmans_rank = stats.spearmanr(y_test_df, yhat)
        
        y_test = np.ravel(y_test_df) #reformat to 1D array
        pearsons_r = stats.pearsonr(y_test, yhat)

        

        print('\n################################################################\n\n BEST MODEL FINAL HOLD_OUT PERFORMANCE:')
        print('FINAL_Hold_Out_MAE: %.3f,  FINAL_Hold_Out_Spearman_Rank: %.3f, FINAL_Hold_Out_Pearsons_R: %.3f' % (acc, spearmans_rank[0], pearsons_r[0]))
        print("\n################################################################\n ")


        return AE, acc, spearmans_rank, pearsons_r, pred_df

    def best_model_refit(self):
        # assign the best model hyperparameters
        self.best_model_params = self.CV_dataset.iloc[0,5]
        print('\nFinal_Best_Model_Params: \n%s' % self.best_model_params)
        # set params from the best model to a class object
        best_model = self.user_defined_model.set_params(**self.best_model_params)
        y_train = self.y.copy()
        y_train = np.ravel(y_train) #reformat
        self.best_model = best_model.fit(self.X, y_train) #Fit hyperparameter optimized model using all data as training set.
