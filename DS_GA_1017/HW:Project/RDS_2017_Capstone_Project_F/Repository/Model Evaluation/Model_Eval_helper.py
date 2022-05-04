from dataclasses import dataclass
import sys
import json
import pandas as pd
import gc
# Main Function that will define model behavior

class DataLoader(object):
    """
    Object whose goal is to load a fully pre-processed train and test data, that we can then pass into our Model Class
    """
    def __init__(self, dimmension_check = None):
        """
        initialization parameters:
            directionry_path_dict: a directory dictionary referencing the preprocessed and feature engineering dataset folder paths
            self.train = the fully cleaned training dataset
            self.test = the fully cleaned testing dataset
            self.dimmension_check = make sure we have the same number of columns as we would expect, 783
        """
        self.directory_path_dict = {"Preprocess":"Data/Preprocessing/",
                               "Feat_Eng":"Data/Feature_Engineering/"} # reference the locations of csvs for data generated via preprocessing
        self.data = None
        self.test = None
        self.dimmension_check = dimmension_check
        
        
    def import_train_test(self):
        data, test = self.preprocess_raw_data()
        data, test = self.merge_feature_engineering(data, test)
        data, test = self.apply_final_preprocessing(data,test)
<<<<<<< HEAD
        if self.dimmension_check:
            DataLoader.dimmension_check()
=======
        
>>>>>>> 7bb4276 (uploading WIP files to github)
        return(data, test)
    def preprocess_raw_data(self):
        """
        function that loads and merges csvs of already preprocessed data, 
        outputs the entire dataset and testing dataset after preprocessing
        takes about 1.5 minutes on my local machine
        """

        #create path list
        directory = self.directory_path_dict['Preprocess']
<<<<<<< HEAD
        file_list = ['data_app', 'test_app', 'avg_buro', 'ccbl_mon', 'pos_recent', 'inst_avg'] #create list of file_names w/o .csv extension
=======
        file_list = ['data_app', 'test_app', 'avg_buro', 'ccbl_mon', 'pos_recent', 'avg_prev', 'inst_avg'] #create list of file_names w/o .csv extension
>>>>>>> 7bb4276 (uploading WIP files to github)
        path_list = [directory + x + '.csv' for x in file_list] #create path list by adding necessary folder and extensions to each file
        #load data
        print('Reading Raw Data')
        data = pd.read_csv(path_list[0], compression='zip')  #load preprocessed train data
        test = pd.read_csv(path_list[1], compression='zip') #load preprocessed test data 
        avg_buro = pd.read_csv(path_list[2], compression='zip') #load preprocessed buro data
        ccbl_mon = pd.read_csv(path_list[3], compression='zip') #load preprocessed cc balance data
        pos_recent = pd.read_csv(path_list[4], compression='zip') #load preprocessed cash data
<<<<<<< HEAD
        inst_avg = pd.read_csv(path_list[5], compression='zip') #load preprocessed installment data
        
        #speciallyl oad avg_prev, it was a csv w > 100 MB so i had to save it as a compressed Parquet to push to github
        avg_prev = pd.read_parquet(directory+'avg_prev.parquet.gzip')
=======
        avg_prev = pd.read_csv(path_list[5], compression='zip') #load preprocessed previous application data
        inst_avg = pd.read_csv(path_list[6], compression='zip') #load preprocessed installment data
>>>>>>> 7bb4276 (uploading WIP files to github)

        print('Joining datasets to training data ~ 1 min out of 9 mins to import data')
        #merge data
        data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
        data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
        data = data.merge(right=ccbl_mon.reset_index(), how='left', on='SK_ID_CURR')
        data = data.merge(right=pos_recent.reset_index(), how='left', on='SK_ID_CURR')
        data = data.merge(right=inst_avg.reset_index(), how='left', on='SK_ID_CURR')

        test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
        test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
        test = test.merge(right=ccbl_mon.reset_index(), how='left', on='SK_ID_CURR')
        test = test.merge(right=pos_recent.reset_index(), how='left', on='SK_ID_CURR')
        test = test.merge(right=inst_avg.reset_index(), how='left', on='SK_ID_CURR')

        #delete variables
        del avg_prev, avg_buro, ccbl_mon, pos_recent, inst_avg
        gc.collect()

        #return outputs data and test
        return(data,test)
    
    def merge_feature_engineering(self,data,test):
        """
        function that merges the csv output files from our feature engineering scripts
        to the preprocessed application data, takes about 7 minutes
        Logic path:
                Create a dictionary that holds all file path names
                Read files in the order as the original script
                Merge files in the order as the original script
                Delete unnecessary files
                Return data,test
                
        Takes about 7 minutes
        """
          #create path list
        print('Starting to merge feature engineering datasets ~ 1 minute 40 secs / 9 mins')
        directory = self.directory_path_dict['Feat_Eng']
        base_dict= {'Doc':'doc_score',
                    'House':'house_score', 
                    'CC': 'cc_score', 
                    'Bureau_Bal':'bubl_score', 
                    'Inst':'inst_score', 
                    'POS':'pos_score'} #create list of file_names that end in either _train.csv or _test.csv

        #create a dictionary of csv path names that are used for both datasets 
        joint_dict = {'House':'house_ex.csv',
                      'Buro':'agg_buro_score.csv', 
                      'Month_Score':'agg_month_score.csv', 
                      'Prev_Score':'agg_prev_score.csv'}
                     
        self.train_dict = {k:directory + base_dict[k]+ '_train.csv' for k in base_dict.keys()} #create train path dict by adding necessary folder and extensions to each file
        self.test_dict= {k:directory + base_dict[k] + '_test.csv' for k in base_dict.keys()}  #create test path dict 
        self.joint_dict = {k:directory + joint_dict[k] for k in joint_dict.keys()} #dict comprehension to prepend the folder path to the path name for joint files
        

        #load data, starting with joint_data agg month _Cols
        print('Reading Joint Data ~ 2 mins / 9 mins')
        buro_cols = ['SK_ID_CURR','month_score_max','month_score_std','month_score_mean','month_score_sum']
        agg_month_score = pd.read_csv(self.joint_dict['Month_Score'],usecols=buro_cols,compression='zip')

        #join agg_month score to train and test
        data = data.merge(right=agg_month_score, how='left', on='SK_ID_CURR') 
        test = test.merge(right=agg_month_score, how='left', on='SK_ID_CURR')

        #delete data
        del agg_month_score, buro_cols
        gc.collect()

        #load data, prev_score and buro_score
        agg_prev_score = pd.read_csv(self.joint_dict['Prev_Score'], compression='zip')
        agg_buro_score = pd.read_csv(self.joint_dict['Buro'], compression = 'zip')
                
        #delete target data

        del agg_prev_score['TARGET']
        del agg_buro_score['TARGET']
        gc.collect()

        #merge both datasets
        data = data.merge(right=agg_prev_score, how='left', on='SK_ID_CURR')
        test = test.merge(right=agg_prev_score, how='left', on='SK_ID_CURR')
        data = data.merge(right=agg_buro_score, how='left', on='SK_ID_CURR')
        test = test.merge(right=agg_buro_score, how='left', on='SK_ID_CURR')

        #delete 
        del agg_prev_score, agg_buro_score
        gc.collect()

   
        print('Reading Housing Data, 4 minutes out of ~ 9 minutes completed')
        #load train/test house scores and house score ext
        train_house_score = pd.read_csv(self.train_dict['House'])
        test_house_score = pd.read_csv(self.test_dict['House'])
        house_score_ext  =  pd.read_csv(self.joint_dict['House'])

        #merge data
        data = data.merge(right=train_house_score, how='left', on='SK_ID_CURR')
        data = data.merge(right=house_score_ext, how='left', on='SK_ID_CURR')

        test = test.merge(right=test_house_score, how='left', on='SK_ID_CURR')
        test = test.merge(right=house_score_ext, how='left', on='SK_ID_CURR')

        #load train and test cc data
        train_cc_score = pd.read_csv(self.train_dict['CC'])
        test_cc_score = pd.read_csv(self.test_dict['CC'])

        #merge
        data = data.merge(right=train_cc_score, how='left', on='SK_ID_CURR')
        test = test.merge(right=test_cc_score, how='left', on='SK_ID_CURR') 

        #load train and test bureau balance data
        train_bureau_bal = pd.read_csv(self.train_dict['Bureau_Bal'])
        test_bureau_bal = pd.read_csv(self.test_dict['Bureau_Bal'])
        
        print('Merging Bureau Data, 6.5 minutes ~ 9 minutes completed ')
        #merge
        data = data.merge(right=train_bureau_bal, how='left', on='SK_ID_CURR')
        test = test.merge(right=test_bureau_bal, how='left', on='SK_ID_CURR')

        ##load train and test point of sale data
        train_pos_score = pd.read_csv(self.train_dict['POS'])
        test_pos_score  = pd.read_csv(self.test_dict['POS'])

        #merge
        data = data.merge(right=train_pos_score, how='left', on='SK_ID_CURR')
        test = test.merge(right=test_pos_score, how='left', on='SK_ID_CURR')
        print('Reading Installemnt data, ~8/9 minutes completed')
        ##load train and test installment data
        train_inst_score = pd.read_csv(self.train_dict['Inst'])
        test_inst_score  = pd.read_csv(self.test_dict['Inst'])

        ## merge
        data = data.merge(right=train_inst_score, how='left', on='SK_ID_CURR')
        test = test.merge(right=test_inst_score, how='left', on='SK_ID_CURR')

        #delete unnecessary data
        del train_inst_score, test_inst_score, train_pos_score, test_pos_score, train_bureau_bal, test_bureau_bal, train_cc_score, test_cc_score, train_house_score, test_house_score
        del house_score_ext
        gc.collect()

        return(data, test)
    
    
    
    
    def apply_final_preprocessing(self,data,test):
        """
        taken directly from the author himself, applies the last feature engineering
        takes 17 seconds
        """
        print('applying final preprocessing, train/test will be loaded in ~ 20 seconds')
        data['Total_AMT_ANNUITY'] = data[['AMT_ANNUITY','bureau_active_sum_AMT_ANNUITY','prev_active_sum_AMT_ANNUITY']].sum(axis=1)
        data['Total_ANNUITY_INCOME_RATIO'] = data['Total_AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
        data['Total_CREDIT'] = data[['AMT_CREDIT','prev_active_sum_AMT_LEFT']].sum(axis=1) #exclude AMT already paid
        data['Total_CREDIT_INCOME_RATIO'] = data['Total_CREDIT'] / data['AMT_INCOME_TOTAL']
        data['Total_acc'] = data[['prev_count','bureau_count']].sum(axis=1)
        data['Total_active_acc'] = data[['prev_active_count','bureau_active_count']].sum(axis=1)
        data['Total_AMT_LEFT'] = data['AMT_CREDIT'] + data['prev_active_sum_AMT_LEFT'] + data['bureau_active_sum_AMT_CREDIT_LEFT']
        data['Total_AMT_LEFT_INCOME_RATIO'] = data['Total_AMT_LEFT']/data['AMT_INCOME_TOTAL']

        test['Total_AMT_ANNUITY'] = test[['AMT_ANNUITY','bureau_active_sum_AMT_ANNUITY','prev_active_sum_AMT_ANNUITY']].sum(axis=1)
        test['Total_ANNUITY_INCOME_RATIO'] = test['Total_AMT_ANNUITY'] / test['AMT_INCOME_TOTAL']
        test['Total_CREDIT'] = test[['AMT_CREDIT','prev_active_sum_AMT_LEFT']].sum(axis=1)
        test['Total_CREDIT_INCOME_RATIO'] = test['Total_CREDIT'] / test['AMT_INCOME_TOTAL']
        test['Total_acc'] = test[['prev_count','bureau_count']].sum(axis=1)
        test['Total_active_acc'] = test[['prev_active_count','bureau_active_count']].sum(axis=1)
        test['Total_AMT_LEFT'] = test['AMT_CREDIT'] + test['prev_active_sum_AMT_LEFT'] + test['bureau_active_sum_AMT_CREDIT_LEFT']
        test['Total_AMT_LEFT_INCOME_RATIO'] = test['Total_AMT_LEFT']/test['AMT_INCOME_TOTAL']

        #current application compare to previous application
        shared_feats = ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_PAY_YEAR', 
                        'AMT_DIFF_CREDIT_GOODS', 'AMT_CREDIT_GOODS_PERC']
        for f_ in shared_feats:
            data[f_+'_to_prev_approved'] = (data[f_] - data['prev_approved_'+f_+'_MEAN'])/data['prev_approved_'+f_+'_MEAN']
            data[f_+'_to_prev_refused'] = (data[f_] - data['prev_refused_'+f_+'_MEAN'])/data['prev_refused_'+f_+'_MEAN']
            test[f_+'_to_prev_approved'] = (test[f_] - test['prev_approved_'+f_+'_MEAN'])/test['prev_approved_'+f_+'_MEAN']
            test[f_+'_to_prev_refused'] = (test[f_] - test['prev_refused_'+f_+'_MEAN'])/test['prev_refused_'+f_+'_MEAN']
<<<<<<< HEAD
       
        #drop unnecessary index columns
        # data= data.drop(['index','index_x','index_y'],axis = 1)
        # test = test.drop(['index','index_x','index_y'],axis = 1)
        #assign as attributes
        self.data = data
        self.test = test
=======
        #drop unnecessary index columns
        data, test = data.drop(['index','index_x','index_y'],axis = 1), test.drop(['index','index_x','index_y'],axis = 1)
>>>>>>> 7bb4276 (uploading WIP files to github)
        return(data, test)
    
    
    def dimmensension_check(self):
<<<<<<< HEAD
        print('validating dimmesnsions match original input')
        true_train_shape = (307511, 783)
        train_shape = self.data.shape

        true_test_shape = (48744, 783)
        test_shape = self.test.shape
        check = 0
        if train_shape != true_train_shape:
            print(f'Training data has {train_shape}, which does not equal {true_train_shape}')
        else:
            print(f'Success! Training data has {train_shape}, which equals {true_train_shape}')
            self.data.drop(columns = ['index','index_x','index_y'], inplace = True)
            if self.train.shape != true_train_shape:
                check +=1
            else:
                pass 
        
        if test_shape != true_test_shape:
            print(f'Training data has {test_shape}, which does not equal {true_train_shape}')
        else:
            print(f'Success! Training data has {test_shape}, which equals {true_test_shape}')
            self.test.drop(columns = ['index','index_x','index_y'], inplace = True)
            if self.test.shape != true_train_shape:
                check +=1
            else:
                pass 
        if check == 2:
            print('Dimmension Check passed')
        else:
            print('Dimmension Check failed')
            print(f'train shape: {self.train.shape}, test shape: {self.test.shape}')
        
=======
        if self.dimension_check:
            print('validating dimmesnsions match original input')
            true_shape = (307511, 783)
            our_shape = self.data.shape
            if our_shape != true_col_num:
                print(f'Training data has {our_shape}, which does not equal {true_shape}')
            else:
                print(f'Success! Training data has {str(our_shape)}, which equals {true_Shape}')
        pass
>>>>>>> 7bb4276 (uploading WIP files to github)
