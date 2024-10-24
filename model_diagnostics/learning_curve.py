from sklearn.model_selection import learning_curve
import time
from copy import deepcopy
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import os


def get_learning_curve(pipeline, refined = False, NUM_ITER =5, num_splits =5, num_sizes= 50):

    start_time = time.time()
    #Initialize
    pipeline['Learning_Curve'] = {'NUM_ITER': NUM_ITER,
                        'num_splits': num_splits,
                        'num_sizes': num_sizes,
                        'Train_Error': None,
                        'Valid_Error': None
                        }
    #Config
    save_path = pipeline['Saving']['Diagnostics'] + 'learning_curve/'
    if os.path.exists(save_path) == False:
        os.makedirs(save_path, 0o666)
    

    trained_model = pipeline['Model_Selection']['Best_Model']['Model']

    #Whether to used feature refined dataset or not
    if refined:
        X = pipeline['Feature_Reduction']['Refined_X']
        pipeline['Learning_Curve']['Dataset_Type']  = 'Refined'
    else:
        X = pipeline['Data_preprocessing']['X']
        pipeline['Learning_Curve']['Dataset_Type']  = 'All'


    model_name = pipeline['Model_Selection']['Best_Model']['Model_Name']
    c = pipeline['Cell']
    Y = pipeline['Data_preprocessing']['y']


    #Copy Trained Model for learning curve
    model = deepcopy(trained_model)


    #initialize training sizes
    train_size= np.linspace(0.005, 1, num_sizes)*len(X)*(num_splits-1)/num_splits
    train_size = np.floor(train_size).astype(int)
    train_scores_mean = pd.DataFrame(index=train_size)
    validation_scores_mean = pd.DataFrame(index=train_size)
    
    print(f"\n############ Calculating Learning Curve: {model_name}_{c}############ ")
    #Train model and record performance
    for i in range(NUM_ITER):
        cross_val = KFold(n_splits= num_splits, random_state= i+10, shuffle=True)
        train_sizes, train_scores, validation_scores = learning_curve(estimator = model, 
                                                                        X = X, 
                                                                        y = np.ravel(Y), 
                                                                        cv = cross_val, 
                                                                        train_sizes= train_size,
                                                                        scoring = 'neg_mean_absolute_error', shuffle= True,
                                                                        random_state = 42,
                                                                        n_jobs= -1)

        train_scores_mean[i] = -train_scores.mean(axis = 1)
        validation_scores_mean[i] = -validation_scores.mean(axis = 1)
        
    #Calculate overall results
    train_scores_mean['Train_size'] = train_scores_mean.index
    train_scores_mean["Score_Type"] = "Train"
    train_scores_mean["Mean_MAE"] = train_scores_mean.iloc[:,:NUM_ITER].mean(axis=1)
    train_scores_mean["sd"] = train_scores_mean.iloc[:,:NUM_ITER].std(axis=1)
    train_scores_mean['Percent_Error'] = train_scores_mean['Mean_MAE'] * 100
    train_scores_mean['Percent_SD'] = train_scores_mean['sd'] * 100   

    validation_scores_mean['Train_size'] = validation_scores_mean.index
    validation_scores_mean["Score_Type"] = "Validation"
    validation_scores_mean["Mean_MAE"] = validation_scores_mean.iloc[:,:NUM_ITER].mean(axis=1)
    validation_scores_mean["sd"] = validation_scores_mean.iloc[:,:NUM_ITER].std(axis=1)
    validation_scores_mean['Percent_Error'] = validation_scores_mean['Mean_MAE'] * 100
    validation_scores_mean['Percent_SD'] = validation_scores_mean['sd'] * 100  

    #Save Data
    with open(save_path + f'{model_name}_{c}_training_results.csv', 'w', encoding = 'utf-8-sig') as f:
        train_scores_mean.to_csv(f)

    with open(save_path + f'{model_name}_{c}_val_results.csv', 'w', encoding = 'utf-8-sig') as f:
        validation_scores_mean.to_csv(f)

    #Update Pipeline
    pipeline['Learning_Curve']['Model_used']  = trained_model
    pipeline['Learning_Curve']['Train_Error'] = train_scores_mean
    pipeline['Learning_Curve']['Valid_Error'] = validation_scores_mean
    pipeline['STEPS_COMPLETED']['Learning_Curve'] = True

    print('\n######## Learning_Curve Results Saved')
    print("\n\n--- %s minutes for Learning Curve---" % ((time.time() - start_time)/60))  
    return pipeline