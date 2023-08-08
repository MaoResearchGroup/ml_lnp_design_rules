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




class NESTED_CV_reformat:
  
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
    def __init__(self, datafile_path, model_type = None):
        self.df = pd.read_csv(datafile_path)
          
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
          self.p_grid ={'n_estimators':[100,300,400],
                        'criterion':['squared_error', 'absolute_error'],
                        'max_depth':[None],
                        'min_samples_split':[2,4,6,8],
                        'min_samples_leaf':[1,2,4],
                        'min_weight_fraction_leaf':[0.0],
                        'max_features': [None, 'sqrt'],
                        'max_leaf_nodes':[None],
                        'min_impurity_decrease': [0.0],
                        'bootstrap':[True],
                        'oob_score':[True],
                        'ccp_alpha': [0, 0.005, 0.01]}
          # # Number of trees in random forest
          # n_estimators = [int(x) for x in np.linspace(start = 1, stop = 200, num = 10)]
          # # Number of features to consider at every split
          # max_features = [None, 'sqrt']
          # # Maximum number of levels in tree
          # #max_depth = [int(x) for x in np.linspace(2, 30, num = 11)]
          # max_depth = [int(x) for x in np.linspace(5, 30, num = 11)]
          # max_depth.append(None)
          # # Minimum number of samples required to split a node
          # min_samples_split = [2, 5, 10]
          # # Minimum number of samples required at each leaf node
          # min_samples_leaf = [1, 2, 4]
          # # Method of selecting samples for training each tree
          # bootstrap = [True, False]
          # # Create the random grid
          # self.p_grid = {'n_estimators': n_estimators,
          #               'max_features': max_features,
          #               'max_depth': max_depth,
          #               'min_samples_split': min_samples_split,
          #               'min_samples_leaf': min_samples_leaf,
          #               'bootstrap': bootstrap}
          

        elif model_type == 'LGBM':
          self.user_defined_model = LGBMRegressor(random_state=4)
          self.p_grid ={"n_estimators":[100,150,200,250,300,400,500,600],
                        'boosting_type': ['gbdt', 'dart', 'goss'],
                        'num_leaves':[16,32,64,128,256],
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

    def input_target(self, cell_type, wt_percent, size_zeta):
        prefix = "RLU_" #WARNING: HARDCODED
        if wt_percent == True:
          formulation_param_names = ['wt_Helper', 'wt_Dlin','wt_Chol', 'wt_DMG', 'wt_pDNA']
        else:
          formulation_param_names = ['NP_ratio', 'Dlin-MC3_Helper lipid_ratio',
                        'Dlin-MC3+Helper lipid percentage', 'Chol_DMG-PEG_ratio'] 
          
        lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP', 'cTPSA', 'Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds']
        #lipid_param_names = ['P_charged_centers', 'N_charged_centers', 'cLogP','Hbond_D', 'Hbond_A', 'Total_Carbon_Tails', 'Double_bonds', 'Helper_MW']
        if size_zeta == True:
          input_param_names = lipid_param_names +  formulation_param_names + ['Size', 'Zeta', 'PDI']
        else:
          input_param_names = lipid_param_names +  formulation_param_names 
        

        #Formatting Training Data
        cell_data = self.df[['Formula label', 'Helper_lipid'] + input_param_names + [prefix + cell_type]]
        cell_data = cell_data.dropna() #Remove any NaN rows
        if size_zeta == True:
          cell_data = cell_data[cell_data.Size != 0] #Remove any rows where size = 0
          cell_data = cell_data[cell_data.Zeta != 0] #Remove any rows where zeta = 0
          cell_data = cell_data[cell_data.PDI <= 0.45] #Remove any rows where PDI > 0.45

        cell_data.loc[cell_data[prefix + cell_type] < 3, prefix + cell_type] = 3 #replace all RLU values below 3 to 3

        print(cell_data)

        self.cell_data = cell_data

      
        
        print("Input Parameters used:", input_param_names)
        print("Number of Datapoints used:", len(self.cell_data.index))

        self.X = self.cell_data[input_param_names]                         
        Y = self.cell_data[prefix + cell_type].to_numpy()
        scaler = MinMaxScaler().fit(Y.reshape(-1,1))
        temp_Y = scaler.transform(Y.reshape(-1,1))
        self.Y = pd.DataFrame(temp_Y, columns = [prefix + cell_type])
        self.F = cell_data['Formula label']
        
    
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
        # self.H_test_list = []

        #for i in range(NUM_TRIALS): #configure the cross-validation procedure - outer loop (test set) 
        cv_outer = KFold(n_splits=NUM_TRIALS, random_state= 4, shuffle=True)
        for i, (train_index, test_index) in enumerate(cv_outer.split(self.X)):
  
          # #X = input parameters, y = transfection, F = formulation number
          #X_train, X_test, y_train, y_test, F_train, F_test = train_test_split(self.X, self.Y, self.F, test_size=0.2, random_state= i) #Iterate through different random_state to randomize test_train split
          X_train = self.X.iloc[train_index]
          X_test = self.X.iloc[test_index]
          y_train = self.Y.iloc[train_index]
          y_test = self.Y.iloc[test_index]
          F_train = self.F.iloc[train_index]
          F_test = self.F.iloc[test_index]

          #store test set information
          F_test = np.array(F_test) #prevents index from being brought from dataframe
          self.F_test_list.append(F_test)
          y_test = np.array(y_test) #prevents index from being brought from dataframe
          self.y_test_list.append(y_test)
                
          # configure the cross-validation procedure - inner loop (validation set/HP optimization)
          cv_inner = KFold(n_splits = 4, shuffle = True) #4 splits to make 80% train set into 60%-20% train-validation

          # define search space
          search = RSCV(self.user_defined_model, self.p_grid, n_iter=100, verbose=0, scoring='neg_mean_absolute_error', cv=cv_inner,  n_jobs= 6, refit=True)
                  
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
        CV_dataset['Score_difference'] = abs(CV_dataset['Valid Score'] - CV_dataset['Test Score']) #Groupby dataframe model iterations that best fit the data (i.e., validitaion <= test)
        CV_dataset.sort_values(by=['Score_difference', 'Test Score'], ascending=True, inplace=True) 
        CV_dataset = CV_dataset.reset_index(drop=True) # Reset index of dataframe
        print('Cross Validation Results', CV_dataset)
        # save the results as a class object
        self.CV_dataset = CV_dataset

    def best_model(self):
        # assign the best model hyperparameters
        best_model_params = self.CV_dataset.iloc[0,5]
        print('\nFinal_Best_Model_Params: \n%s' % best_model_params)
        # set params from the best model to a class object
        best_model = self.user_defined_model.set_params(**best_model_params)
        self.Y = np.ravel(self.Y) #reformat
        self.best_model = best_model.fit(self.X, self.Y) #Fit hyperparameter optimized model using all data as training set.