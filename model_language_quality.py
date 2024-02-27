#!/usr/bin/env python3

# %%
import numpy as np
import pandas as pd
import json
import random
import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import joblib
import pickle
import os.path
import sys
from model_util import ModelAlgos, ModelingStage
from model_util import get_file_loc, crc32_hash, model_labels

# ------------------------------------------------------------------------------
#  MAIN_CLASS
# ------------------------------------------------------------------------------


class ModelLanguageQuality:
    """
    Model language proficiency consists of 2 stages:
    - Training over language assessments given a score from 0 to 5, with 5 
    being the best. Assessments are made over several criteria.
    - Analyze samples not trained on to give similar assessments.
    """

    def __init__(self, **kwargs):
        self.modeling_stage = kwargs.get('modeling_stage')
        self.training_data = kwargs.get('training_data')
        self.test_data = kwargs.get('test_data', None)
        self.asset_id = kwargs.get('asset', None)
        self.input_data = kwargs.get('input_data')
        # self.compare_data = kwargs.get('compare_data', None)
        self.category = kwargs.get('category', None)
        self.target = kwargs.get('target')
        self.model_choice = kwargs.get('model_choice', None)
        self.model_files = kwargs.get('model_files', {})
        self.sim_max_assets = kwargs.get('sim_max_assets', (2000))
        self.start_time = time.asctime()
        self.model_string = model_labels.get(self.model_choice)

    # -------------------------------------------------------------------------

    def parse_data_input(self, data):
        """ Any filtering or cleaning of training/transcript data is done here """
        logging.info(f"Parsing training or data file {data}")
        df = pd.read_csv(data, sep=',', index_col=0, header=0,
            nrows=int(self.sim_max_assets)
        )
        logging.debug('_________________________________________________________')
        logging.debug(f"Length B4 filter: {len(df)}")
        transcript_columns = ['question_1_transcript', 'question_2_transcript',
                   'question_3_transcript','question_4_transcript',
                   'question_5_transcript']
        assessment_columns = ['pronunciation_avg', 'vocab_avg', 'fluency_avg', 
                            'cohesion_avg', 'grammar_avg', 'cefr_avg']
        if 'cohesion_avg' in df.columns and 'vocab_avg' in df.columns:
            df_filter1 = df[(df['cohesion_avg'] <= 6.0) | (df['vocab_avg'] <= 6.0)]
        else:
            assessment_columns = []
            df_filter1 = df

        # test if there are actual assessment columns like 'vocab_avg'
        if assessment_columns:
            transcript_columns.append(self.target)
        self.df_transcript = df_filter1[transcript_columns]
        self.df_assessment = df_filter1[assessment_columns]
        logging.debug('_________________________________________________________')
        logging.debug(f"Length After filter: {len(self.df_transcript)}")
        # logging.debug(self.df_transcript.info())
        # logging.debug(self.df_assessment.head(5))
        # logging.debug(self.df_assessment.iloc[1:3])
        logging.debug(self.df_transcript.iloc[0:1])
        if len(self.df_assessment.columns.values) > 0:
            logging.debug(f"parse_data_input(): Assessment Columns => [{','.join(self.df_assessment.columns.values)}]")
        else:
            logging.debug(f"parse_data_input(): No Assessment Columns found in dataset file {data}")

    # -------------------------------------------------------------------------

    def train_model(self):
        # parse training input
        self.parse_data_input(self.training_data)
        logging.info(f"Running training {self.model_choice} for Target '{self.target}'")
        logging.debug(f"Evaluating transcripts of type '{self.category}'")
        if self.model_choice != ModelAlgos.CATBOOST_BL.name:
            logging.debug(f"Training with model {self.model_choice} by using transcripts for {self.category}'")
            if self.model_choice == ModelAlgos.CATBOOST.name:
                self.df_transcript = self.df_transcript.map(crc32_hash)
            df_train, df_test = train_test_split(self.df_transcript, test_size=0.2, random_state=1)
            bl_string = ""
        else:
            logging.debug(f"Training for baseline values for {self.category}'")
            df_train, df_test = train_test_split(self.df_assessment, test_size=0.2, random_state=1)
            bl_string = "_BL"
        logging.debug(f"size of training set vs size of test set {len(df_train)}/{len(df_test)}")

        # *** solve for 'target' ex. 'vocab_avg'
        y_train = df_train[self.target].values
        self.y_test = df_test[self.target].values
        del df_train[self.target]
        del df_test[self.target]
        X_train = df_train
        self.X_test = df_test   
        logging.debug(f"** Shape of X_train:\n{X_train.shape}")

        if self.model_string == 'CatBoost':
            # training_model = CatBoostRegressor(n_estimators=1800, max_depth = 12, random_state=42)
            self.training_model = CatBoostRegressor(max_depth = 12, random_state=42)
        else:
            raise Exception(f"Unknown model {self.model_string}")

        self.training_model.fit(X_train, y_train)
        fname = (
                    f'data/model-{self.model_string}_'
                    f'for-{self.target}_type-{self.category}{bl_string}.joblib'
                )
        # Save the training_model to a file
        joblib.dump(self.training_model, fname)
        logging.debug(f"Start Time: {self.start_time}")
        logging.debug(f"End Time: {time.asctime()}")

    # -------------------------------------------------------------------------

    def run_model(self):
        # run model and if relevant compare to baseline if comparing
        logging.info(f"Running model for Target '{self.target}' of type '{self.category}'")
        if self.test_data is not None:
            test_set = get_file_loc(f'data/{self.test_data}')
        else:
            # retrieve training data, default is from 'model_files' in initial dict., modelD, in model_util.py
            test_set = self.training_data

        default_bl_string = '_BL' # baseline suffix in filename to rate results against
        bl_string = ""
        self.y_test = list()
        # if training_model object does not exist, than we need to setup our input
        if not hasattr(self, 'training_model'):
            # get default dataset with actual assessment values
            self.parse_data_input(test_set)
            if self.model_choice != ModelAlgos.CATBOOST_BL.name:
                logging.debug(f"Evaluating with model {self.model_choice} by using {self.category} transcripts")
                if self.model_choice != ModelAlgos.CATBOOST.name:
                    df_test = self.df_transcript
                else:
                    df_test = self.df_transcript.map(crc32_hash)
                if self.target in self.df_assessment:
                    self.y_test = self.df_assessment[self.target].values
            else:
                logging.debug(f"Evaluating for baseline values for '{self.category}'")
                df_test = self.df_assessment
                bl_string = default_bl_string
                self.y_test = df_test[self.target].values
                del df_test[self.target]
            self.X_test = df_test  # time periods missing evaluation_actual to solve for
            assert self.model_files.get(self.model_choice) is not None, "Model file not found"
            model_file = self.model_files[self.model_choice][self.target]
            logging.debug(f"{self.model_string} is loading model from file {model_file}")
            model = joblib.load(model_file)
        else:
            model = self.training_model

        y_pred = model.predict(self.X_test)
        y_pred = [round(num, 2) for num in y_pred]
        self.df_results = pd.DataFrame(y_pred, index=self.X_test.index,
                                  columns=['evaluation_predicted']
                                  )
        # if hasattr(self, 'df_test'):
        #     self.df_results.join(self.df_test)
        # if self.model_choice != ModelAlgos.CATBOOST_BL.name and hasattr(self.df_assessment):
        #     self.df_results.join(self.df_assessment)

        # *AA TODO Update filename to accomadate multiple datasets 
        results_fname = get_file_loc('data/run_results_' + self.target + bl_string + '.csv')
        self.df_results.to_csv(results_fname)
        logging.debug(f"Run results sent to {results_fname}")
        # {'id':f'{t_id}', 'category':f'{category}', 'vocab_avg':4.5,'fluency_avg':4,'grammar_avg':3.2, 'cefr_avg':4.2}
        if self.asset_id:
            # return json result for specific assessment
            asset_result = { 'id': int(self.asset_id), 'category': self.category, 
                            self.target: self.df_results.loc[int(self.asset_id), 'evaluation_predicted']
                            # self.target: self.df_results.iloc[0,0]
                            }

            return(asset_result)

        baseline_pred = None 
        # evaluate results when true values for target exist
        if len(self.y_test) > 0 and self.asset_id is None:
            logging.debug("Checking for baseline predictions")
            # if exists retrieve/run baseline predictions for current dataset
            # TODO add ability to recognize multiple datasets, run baseline model if new
            if self.model_choice != ModelAlgos.CATBOOST_BL.name:
                logging.debug("Retrieving baseline predictions")
                baseline_data = get_file_loc('data/run_results_' + self.target + default_bl_string + '.csv')
                # Filter df_baseline based on matching index rows in df_test
                df_baseline = pd.read_csv(baseline_data, sep=',', index_col=0, header=0,
                    nrows=int(self.sim_max_assets)
                )
                # align rows in baseline with same assessment id's
                filtered_df = df_baseline.loc[df_baseline.index.isin(self.X_test.index), :]
                baseline_pred = filtered_df['evaluation_predicted'].values
                logging.debug(f"len y_pred vs baseline_pred => {len(y_pred)}/{len(baseline_pred)}")
                if len(baseline_pred) == 0:
                    logging.debug(f"No baseline_predict values found")
                    baseline_pred = None
                
            self.evaluate_results(y_pred, baseline_pred)

    # -------------------------------------------------------------------------

    def evaluate_results(self, y_pred, baseline_pred):
        logging.info(f"Evaluating model on '{self.category}'")

        analyze_resultsH = {'Model Prediction vs Actual': y_pred, 
            'Baseline Prediction vs Actual' : baseline_pred
        }
        for cmp_type, results in analyze_resultsH.items():
            if results is None:
                continue
            # evaluate model results 
            mae = mean_absolute_error(results, self.y_test)
            mse = mean_squared_error(results, self.y_test)
            rmse = np.sqrt(mse)
            nrmse = rmse/(max(self.y_test)-min(self.y_test))
            logging.info('_________________________________________________________')
            logging.info(f">  Loading {cmp_type} using model {self.model_string} when solving for {self.target} at {time.asctime()}")
            logging.info(f'Mean Absolute Error (MAE): {mae:.3f}')
            logging.info(f'Mean Squared Error (MSE): {mse:.3f}')
            logging.info(f'Root Mean Squared Error (RMSE): {rmse:.3f}')
            logging.info(f'Normalized Root Mean Squared Error (NRMSE): {nrmse:.3f}')

            logging.info(f"min(results) = {round(min(results),2)}")
            logging.info(f"max(results) = {round(max(results),2)}")
            logging.info(f"min(self.y_test) = {round(min(self.y_test),2)}")
            logging.info(f"max(self.y_test) = {round(max(self.y_test),2)}")

        error_threshold = 10
        counter=0
        instance=[]
        logging.debug("\n* Results out of range: actual vs predicted")
        ttl_predicted = len(y_pred)
        for i in range(ttl_predicted):
            if abs((y_pred[i]-self.y_test[i])/y_pred[i]*100) >= error_threshold:
                counter += 1
                instance.append((self.y_test[i],y_pred[i]))
                logging.debug(f"{self.y_test[i]} vs {y_pred[i]}")
        percent_over_threshold = round(counter/ttl_predicted *100, 1)
        logging.info(f"----\nTotal predicted out of range ({error_threshold})% "
                     f"or more of actual: {counter} out of {ttl_predicted} = "
                     f"{percent_over_threshold}%")

        logging.info("\n* Showing actual vs predicted results")
        amnt_to_show = 8 
        amnt_on_each_line = 8
        logging.info(f"   - First {amnt_to_show}")
        log_strings = list() 
        for i in range(amnt_to_show):
            log_strings.append(f"{self.y_test[i]}/{round(y_pred[i],2)}")
            if (i +1) % amnt_on_each_line == 0:
                logging.info(", ".join(log_strings))
                log_strings = list() 
            
        logging.info(f"   - Last {amnt_to_show}")
        log_strings = list() 
        for i, (t, p) in enumerate(zip(self.y_test[-(amnt_to_show):], y_pred[-(amnt_to_show):])):
            log_strings.append(f"{t}/{round(p,2)}")
            if (i +1) % amnt_on_each_line == 0:
                logging.info(", ".join(log_strings))
                log_strings = list() 

    def upload_results(self):
        logging.info("Uploading results to database")
