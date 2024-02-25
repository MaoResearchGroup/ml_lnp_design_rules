import pickle
from scipy import stats
from copy import deepcopy, copy
import pandas as pd
import plotting_functions as plotter
import numpy as np
from sklearn.metrics import mean_absolute_error
from itertools import chain, product
from operator import truediv
import os
import utilities as util
import matplotlib.pyplot as plt

def init_validation_dict(pipeline, pipeline_save):
    print('\n\n########## INITIALIZING NEW VALIDATION DICT ##############\n\n')
    RUN_NAME = pipeline['Saving']['RUN_NAME']
    cell = copy(pipeline['Cell'])
    model = deepcopy(pipeline['Model_Selection']['Best_Model']['Model'])
    prefix = pipeline['Data_preprocessing']['prefix']
    input_params = copy(pipeline['Feature_Reduction']['Refined_Params'])
    scaler = deepcopy(pipeline['Data_preprocessing']['Scaler'])
    all_data = copy(pipeline['Data_preprocessing']['all_proc_data'])

    #Saving
    validation_save_path = f"{RUN_NAME}{cell}/IV_Validation/"
    figure_save_path = pipeline['Saving']['Figures'] + 'IV_Validation/'

    #Parameter Selection
    general_lipid_parameters = ['P_charged_centers', 
                        'N_charged_centers', 
                        'cLogP', 
                        'Hbond_D', 
                        'Hbond_A', 
                        'Total_Carbon_Tails', 
                        'Double_bonds']
    
    lipid_parameters = [element for element in general_lipid_parameters if element in input_params]
    
    general_formula_parameters = ['NP_ratio',
                            'Chol_DMG-PEG_ratio',
                            'Dlin-MC3+Helper lipid percentage',
                            'Dlin-MC3_Helper lipid_ratio']
    
    formula_parameters = [element for element in general_formula_parameters if element in input_params]

    print(f'INPUT PARAMETERS USED:{lipid_parameters+formula_parameters}')
    

    #Train model with refined features
    X = pipeline['Feature_Reduction']['Refined_X']
    y = pipeline['Data_preprocessing']['y']


    X = X[input_params].copy()
    model.fit(X,np.ravel(y))

    validation_dict = { 'Cell' : cell,
                        'STEPS_COMPLETED':{
                            'Load_pipeline': False,
                            'Validation_array': False,
                            'Validation_set': False,
                            'Plot_validation_set': False,
                            'Experimental': False,
                            'Analyze_validation': False
                            },
                        'Saving':{
                            'RUN_NAME': RUN_NAME,
                            'Validation_set': validation_save_path,
                            'Figures': figure_save_path
                            },
                        'Pipeline_load':{
                            'Model': model,
                            'prefix': prefix,
                            'scaler' : scaler,
                            'Raw_training_data':all_data},
                        'Validation_set':{
                            'Input_parameters': input_params,
                            'Lipid_parameters': lipid_parameters,
                            'Formula_parameters': formula_parameters,
                            'New_formula_params_values': None,
                            'Validation_array':None,
                            'array_y_pred': None,
                            'Validation_array_predicted':None,
                            'Search_conditions': None,
                            'final_selection': None,
                            'final_y_pred': None
                            },
                        'Experimental': {
                            'data_path': None,
                            'raw_data': None,
                            'MAE': None,
                            'Spearman': None,
                            'Pearson': None,

                            }
                        }
    

        ####check save paths ########
    for save_path in validation_dict['Saving'].values():
        if not os.path.exists(save_path):
            # Create the directory if it doesn't exist
            os.makedirs(save_path)
            print(f"Directory '{save_path}' created.")
    else:
        print(f"Directory '{save_path}' already exists.")

    with open(pipeline_save , 'wb') as file:
        pickle.dump(validation_dict, file)
        print(f"\n\n--- SAVED New {cell} Validation CONFIG  ---")


    return validation_dict


def create_formulation_array(pipeline, formula_validation_array, remove_training = True):
  
    #Extract necessary data
    df = copy(pipeline['Pipeline_load']['Raw_training_data'])
    lipid_params = pipeline['Validation_set']['Lipid_parameters']
    formula_params = pipeline['Validation_set']['Formula_parameters']
    helper_lipid_names = df['Helper_lipid'].drop_duplicates().to_numpy()

    #Extract lipid parameterizations from training data
    lipid_parameterization = df.loc[:, df.columns.isin(['Helper_lipid'] + lipid_params)].drop_duplicates(keep='first').reset_index(drop= True).set_index('Helper_lipid')

    #Update novel formulation parameter values with different helper lipids
    formulation_params_dic = {'Helper_lipid' : helper_lipid_names, **formula_validation_array}
                            
    #Create array of all unique formulations
    formulation_param_array = pd.DataFrame([row for row in product(*formulation_params_dic.values())], 
                        columns=formulation_params_dic.keys())

    # Remove formulations that are present in Training dataset
    if remove_training:
        training_data = df[['Helper_lipid'] + formula_params]
        merged_df = pd.merge(formulation_param_array, training_data, how='left', indicator=True)
        # Filter out rows that are found in both dataframes
        formulation_param_array = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    #Add lipid parameterization for each helper lipid to formulation parameter array
    for parameter in lipid_params:
        for helper_lipid in helper_lipid_names:
            formulation_param_array.loc[formulation_param_array.Helper_lipid == helper_lipid, parameter] = lipid_parameterization.loc[helper_lipid, parameter]

    #reformat array
    formulation_param_array = formulation_param_array[['Helper_lipid'] + lipid_params + formula_params] #reorder columns
    formulation_param_array.reset_index(inplace = True, names = "Validation Formula label") #Bring down index as formulation labels

    #Update Pipeline object
    pipeline['Validation_set']['New_formula_params_values'] = formulation_params_dic
    pipeline['Validation_set']['Validation_array'] = formulation_param_array

    print('Number of formulations in validation array: ', len(formulation_param_array.index))

    return formulation_param_array

def get_multiplex_list(validation, multiplex_list):
    RUN_NAME = validation['Saving']['RUN_NAME']
    list_of_validations = []

    for c in multiplex_list:
        list_path =  f'{RUN_NAME}{c}/IV_Validation/Validation_dict.pkl'
        with open(list_path, 'rb') as file:
            temp_validation = pickle.load(file)
        #Add source to each df
        selection = copy(temp_validation['Validation_set']['final_selection'])
        selection['Source_cell'] = c
        list_of_validations.append(selection)

    #concat together
    df_multiplex_validation = pd.concat(list_of_validations, axis=0, ignore_index=True)


    #get predicted values for multiplex set
    predicted_multiplex,df_multiplex_validation['Normalized_y'],df_multiplex_validation['RLU_y'] = get_predictions(df_multiplex_validation[validation['Validation_set']['Input_parameters']], 
                                                                    validation['Pipeline_load']['Model'],
                                                                    validation['Pipeline_load']['scaler'])            
    #sort by predicted values
    df_multiplex_validation = df_multiplex_validation.sort_values(by='Normalized_y', ascending=True).reset_index(drop = True)

    validation['Validation_set']['Multiplex_set'] = df_multiplex_validation

    #save
    save  = validation['Saving']['Validation_set']
    df_multiplex_validation.to_excel(f'{save}Multiplex_Validation_set.xlsx', index = False)


    #plot
    plotter.plot_validation_predictions(cell = validation['Cell'],
                                        validation_set=df_multiplex_validation,
                                        palette = 'husl',
                                        save = f'{save}Multiplex_Validation_bar.svg'
                                        )
    


def create_validation_list(pipeline, method:str = 'top_bot'):
    
    #Extract data
    df_for_selection = pipeline['Validation_set']['Validation_array_predicted']
    search_conds = pipeline['Validation_set']['Search_conditions']

    print(f'######## SELECTION METHOD: {method} ##############')
    if method == 'top_bot':
        selection,_,_ =  get_top_bot_validation_list(df_for_selection, selection_args= search_conds)
    elif method == 'stratified':
        selection,_,_ = get_stratified_validation_list()
    else:
        raise('ERROR: UNKNOWN SELECTION METHOD')
    
    #update pipeline
    pipeline['Validation_set']['final_selection'] = selection
    
    #save df to folder
    save  = pipeline['Saving']['Validation_set']
    selection.to_excel(f'{save}Validation_set.xlsx', index = False)

    return selection

def get_predictions(X, model, scaler):
    #predict
    normalized_predictions = model.predict(X)

    #convert back to RLU
    converted_predictions = scaler.inverse_transform(normalized_predictions.reshape(-1, 1))

    #combined into a single dataframe
    predictions = pd.DataFrame({'Normalized_y': normalized_predictions,
                                'RLU_y': converted_predictions.flatten()})

    return predictions, normalized_predictions, converted_predictions

def get_stratified_validation_list(predictions, helper_lipid_names, cell, name, save_path):
  #Stratify Predictions
  N_bins = 10
  bins = np.linspace(0,1, endpoint = False, num = N_bins)
  for i in range(N_bins):
    predictions.loc[predictions[f'{cell}_Prediction'] >= bins[i], "Bin"] = i
  
  #Extract Random Formulations to Test
  num_formulations = 90
  formulation_list = predictions.groupby('Bin', group_keys=False).apply(lambda x: x.sample(int(num_formulations/N_bins)))

  #Redo randomized search until diversity conditions are met
  min_lipid = 9
  min_Dlin_helper_ratio = 4
  min_Dlin_helper_percent = 4
  min_Chol_PEG = 2
  min_NP_ratio = 2
  iter = 0
  max_iter = 100000
  while not(all(len(formulation_list[formulation_list['Helper_lipid'] == lipid]) >= min_lipid for lipid in helper_lipid_names) and 
            all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'Dlin-MC3_Helper lipid_ratio'].unique().size >= min_Dlin_helper_ratio for lipid in helper_lipid_names) and
            all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'Chol_DMG-PEG_ratio'].unique().size >= min_Chol_PEG for lipid in helper_lipid_names) and
            all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'Dlin-MC3+Helper lipid percentage'].unique().size >= min_Dlin_helper_percent for lipid in helper_lipid_names) and
            all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'NP_ratio'].unique().size >= min_NP_ratio for lipid in helper_lipid_names)): 

    formulation_list = predictions.groupby('Bin', group_keys=False).apply(lambda x: x.sample(int(num_formulations/N_bins)))   
    iter += 1
    if iter > max_iter:
      print(f'Could not find any samples that met condition after {max_iter} combinations')
      formulation_list = pd.DataFrame() #Return empty dataframe
      return formulation_list
      
  formulation_list.sort_values(by = ['Helper_lipid'], inplace = True)
  print('Final Iter #: ', iter)  
  print('Total Unique Helper Lipids:', formulation_list['Helper_lipid'].unique().size, '/', predictions['Helper_lipid'].unique().size)
  print('Total Unique Dlin:helper :', formulation_list['Dlin-MC3_Helper lipid_ratio'].unique().size, '/', predictions['Dlin-MC3_Helper lipid_ratio'].unique().size)
  print('Total Unique Dlin + helper :', formulation_list['Dlin-MC3+Helper lipid percentage'].unique().size, '/', predictions['Dlin-MC3+Helper lipid percentage'].unique().size)
  print('Total Unique Chol:DMG :', formulation_list['Chol_DMG-PEG_ratio'].unique().size, '/', predictions['Chol_DMG-PEG_ratio'].unique().size)
  print('Total Unique NP :', formulation_list['NP_ratio'].unique().size, '/', predictions['NP_ratio'].unique().size)

  return formulation_list

def get_top_bot_validation_list(df, selection_args):
    

    #Extract High/Mid/Low Transfection Predictions
    low_bar = df['Normalized_y'].quantile(selection_args['low_bound'])
    high_bar = df['Normalized_y'].quantile(selection_args['high_bound'])
    print("High Trasfection Min:", high_bar)
    print("Low Trasfection Max:", low_bar)


    #Stratify Predictions
    high_formulations = df.loc[df['Normalized_y'] >= high_bar]
    #mod_formulations = predictions.loc[(predictions[f'{cell}_Prediction'] < high_bar) & (predictions[f'{cell}_Prediction'] > low_bar)]
    low_formulations = df.loc[df['Normalized_y'] <= low_bar]
    print('High Helper Lipids', high_formulations['Helper_lipid'].unique())
    print('Total Number of high Formulations:', len(high_formulations.index))

    print('Helper Lipids', low_formulations['Helper_lipid'].unique())
    print('Total Number of low Formulations:', len(low_formulations.index))

    #Randomly selection Formulations to Test from each bin
    n_formulations = selection_args['num_formulations']
    formulation_list = pd.concat([high_formulations.sample(n = n_formulations), 
                                  low_formulations.sample(n = n_formulations)])


    #Redo randomized search until conditions are met
    min_lipid = selection_args['min_lipid']
    min_Dlin_helper_ratio = selection_args['min_Dlin_helper_ratio']
    min_Dlin_helper_percent = selection_args['min_Dlin_helper_percent']
    min_Chol_PEG = selection_args['min_Chol_PEG']
    min_NP_ratio = selection_args['min_NP_ratio']

    iter = 0
    max_iter = 100000

    if selection_args['lipid_specific']:
        helper_lipid_names = df['Helper_lipid'].unique()
        while not(all(len(formulation_list[formulation_list['Helper_lipid'] == lipid]) >= min_lipid for lipid in helper_lipid_names) and 
                    all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'Dlin-MC3_Helper lipid_ratio'].unique().size >= min_Dlin_helper_ratio for lipid in helper_lipid_names) and
                    all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'Chol_DMG-PEG_ratio'].unique().size >= min_Chol_PEG for lipid in helper_lipid_names) and
                    all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'Dlin-MC3+Helper lipid percentage'].unique().size >= min_Dlin_helper_percent for lipid in helper_lipid_names) and
                    all(formulation_list.loc[formulation_list['Helper_lipid'] == lipid, 'NP_ratio'].unique().size >= min_NP_ratio for lipid in helper_lipid_names)): 
        
            formulation_list = pd.concat([high_formulations.sample(n = n_formulations), low_formulations.sample(n = n_formulations)]) #Take new samples from high and low lists
            iter += 1
            if iter > max_iter:
                print(f'###### Could not find any samples that met condition after {max_iter} combinations ######')
                formulation_list = pd.DataFrame() #Return empty dataframe
                return formulation_list, high_formulations, low_formulations
    else:
        while not(len(formulation_list) >= min_lipid and 
                    formulation_list['Dlin-MC3_Helper lipid_ratio'].unique().size >= min_Dlin_helper_ratio and
                    formulation_list['Chol_DMG-PEG_ratio'].unique().size >= min_Chol_PEG and
                    formulation_list['Dlin-MC3+Helper lipid percentage'].unique().size >= min_Dlin_helper_percent and
                    formulation_list['NP_ratio'].unique().size >= min_NP_ratio): 
        
            formulation_list = pd.concat([high_formulations.sample(n = n_formulations), low_formulations.sample(n = n_formulations)]) #Take new samples from high and low lists
            iter += 1
            if iter > max_iter:
                print(f'###### Could not find any samples that met condition after {max_iter} combinations ######')
                formulation_list = pd.DataFrame() #Return empty dataframe
                return formulation_list, high_formulations, low_formulations

    #Formatting List         
    formulation_list.sort_values(by = ['Helper_lipid'], inplace = True)

    #Print Results
    print('Final Iter #: ', iter)  
    print('Total Unique Helper Lipids:', formulation_list['Helper_lipid'].unique().size, '/', df['Helper_lipid'].unique().size)
    print('Total Unique Dlin:helper :', formulation_list['Dlin-MC3_Helper lipid_ratio'].unique().size, '/', df['Dlin-MC3_Helper lipid_ratio'].unique().size)
    print('Total Unique Dlin + helper :', formulation_list['Dlin-MC3+Helper lipid percentage'].unique().size, '/', df['Dlin-MC3+Helper lipid percentage'].unique().size)
    print('Total Unique Chol:DMG :', formulation_list['Chol_DMG-PEG_ratio'].unique().size, '/', df['Chol_DMG-PEG_ratio'].unique().size)
    print('Total Unique NP :', formulation_list['NP_ratio'].unique().size, '/', df['NP_ratio'].unique().size)


    return formulation_list, high_formulations, low_formulations

############################################### MAIN ##############################################################
def import_and_plot_validation_list(pipeline, path):
    df = pd.read_excel(path)

    #get predicted values for multiplex set
    _,df['Normalized_y'],df['RLU_y'] = get_predictions(df[pipeline['Validation_set']['Input_parameters']], 
                                                                    pipeline['Pipeline_load']['Model'],
                                                                    pipeline['Pipeline_load']['scaler'])            
    #sort by predicted values
    df_sorted = df.sort_values(by='Normalized_y', ascending=True).reset_index(drop = True)


    #plot
    save  = pipeline['Saving']['Validation_set']
    plotter.plot_validation_predictions(cell = pipeline['Cell'],
                                        validation_set=df_sorted,
                                        palette = 'husl',
                                        save = f'{save}Imported_Validation_bar.svg')

    return df
def main():
    
    ################ What parts of the pipeline to run ###############
    RUN_NAME  = f"Runs/Final_PDI1_RLU2/"
    new_validation = False

    #Parts to Run/Update
    run_validation_array            = False
    run_validation_set_selection    = False
    run_multiplex_validation        = False #run to also include validate sets from other cells lines into the dataset (other cell lines must have been run before)
    
    import_validation               = True
    run_comparison                  = True

    

    cell_type_list = ['HepG2','HEK293', 'N2a', 'ARPE19','B16', 'PC3']
    multiplex_cell_list = ['HepG2','HEK293', 'N2a', 'ARPE19','B16', 'PC3']


    for c in cell_type_list:
        ########## Parameters for Validation set search #################
        new_formula_parameters ={'NP_ratio' : [5,6,7,9,10,11],
                        'Dlin-MC3_Helper lipid_ratio' : [2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 40, 60, 70, 80, 90, 120, 140, 160, 180],
                        'Dlin-MC3+Helper lipid percentage': [25, 30, 35, 45, 50, 55, 65, 70, 75], 
                        'Chol_DMG-PEG_ratio': [20, 30, 40, 50, 60,70,80, 90, 150, 200, 250,350, 400, 450]}
        
        
        
        search_conditions = {'num_formulations': 6,
                            'lipid_specific': False,
                            'low_bound': 0.10,
                            'high_bound': 0.98,
                            'min_lipid' : 0,
                            'min_Dlin_helper_ratio' : 2,
                            'min_Dlin_helper_percent' : 2,
                            'min_Chol_PEG' : 2,
                            'min_NP_ratio' : 2}
        
        
        #Load previous validation dictionary if exists
        pipeline_path  = f'{RUN_NAME}{c}/IV_Validation/Validation_dict.pkl'


        if os.path.exists(pipeline_path) and new_validation == False:
            print(f'\n\n########## LOADING PREVIOUS PIPELINE FOR {c} ##############\n\n')
            with open(pipeline_path, 'rb') as file:
                validation = pickle.load(file)
        
        #Initialize new pipeline if wanted
        else: 
            #Load Pipeline of interest
            with open(RUN_NAME + f'{c}/Pipeline_dict.pkl', 'rb') as file:
                                        ML_pipeline = pickle.load(file)


            #Initialize validation dict
            validation = init_validation_dict(pipeline= ML_pipeline, 
                                            pipeline_save = pipeline_path)
            
            
            #Run everything except experimental comparison
            run_validation_array            = True
            run_validation_set_selection    = True
            run_multiplex_validation        = False
            run_comparison                  = False

            util.save_pipeline(pipeline=validation, path = pipeline_path, 
                    step = 'DICTIONARY INITIALIZED')


        if run_validation_array:
            #Create array for validation set selection
            validation_array = create_formulation_array(pipeline = validation, 
                                                        formula_validation_array= new_formula_parameters,
                                                        remove_training=True)
            
            #Predict Values for validation array
            validation_input_params = validation['Validation_set']['Input_parameters']
            validation['Validation_set']['array_y_pred'],_,_ = get_predictions(validation_array[validation_input_params], 
                                                                            validation['Pipeline_load']['Model'],
                                                                            validation['Pipeline_load']['scaler'])

            validation['Validation_set']['Validation_array_predicted'] = pd.concat([validation['Validation_set']['Validation_array'], 
                                                                                    validation['Validation_set']['array_y_pred']], axis = 1)
            
            util.save_pipeline(pipeline=validation, path = pipeline_path, 
                step = 'Validation array generation')
            
        if run_validation_set_selection:
            #Select formulations for experimental validation
            validation['Validation_set']['Search_conditions'] = search_conditions
            validation_list = create_validation_list(validation, method = 'top_bot')

            #Plot
            util.save_pipeline(pipeline=validation, path = pipeline_path, 
                step = 'Validation set selected')
    
        if run_multiplex_validation:
            get_multiplex_list(validation=validation, 
                               multiplex_list=multiplex_cell_list)
                
        if import_validation:
            pred_df = import_and_plot_validation_list(validation, path  = f"{RUN_NAME}{c}/IV_Validation/Import_validation.xlsx")
            print(pred_df)

        ######## Compare Validation Results########## 
        if run_comparison:
            validation_path = f"{RUN_NAME}{c}/IV_Validation/"
            exp_df = pd.read_excel(validation_path +"Import_validation.xlsx")

            y_pred = pred_df['RLU_y']

            y_test = exp_df['Exp_RLU']


            print(y_pred)
            print(y_test)


            #Plot
            plotter.plot_predictions(pipeline=validation,
                                    save =validation_path,
                                    pred = y_pred,
                                    exp = y_test,
                                    normalized=False,
                                    correlations=False)
            
            #Calculate MAE
            MAE = mean_absolute_error(y_pred,y_test)
            print(f'Prediction MAE = {MAE}')
    
if __name__ == "__main__":
    main()