#!/usr/bin/env python3

# %%
import numpy as np
import pandas as pd
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
from model_util import ModelAlgos, ModelingStage, get_file_loc, model_labels

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
        self.input_data = kwargs.get('input_data')
        self.compare_data = kwargs.get('compare_data', None)
        self.category = kwargs.get('category', None)
        self.target = kwargs.get('target')
        self.model_name = kwargs.get('model_name', [])
        self.model_files = kwargs.get('model_files', {})
        self.lin_regr_cols = kwargs.get('lin_regr_cols', {})
        self.sim_max_assets = kwargs.get('sim_max_assets', (200))
        self.start_time = time.asctime()

    def parse_training_input(self):
        """ Any filtering or cleaning of training data should be done here """
        logging.info(f"Parsing training file {self.training_data}")
        df = pd.read_csv(self.training_data,
                         sep=',', index_col=0, header=0,
            nrows=int(self.sim_max_assets)
        )
        logging.debug('_________________________________________________________')
        logging.debug(f"Length B4 filter: {len(df)}")
        df_filter1 = df[(df['cohesion_avg'] < 6.0) | (df['vocab_avg'] < 6.0)]
        assessment_columns = ['pronunciation_avg', 'vocab_avg', 'fluency_avg', 
                              'cohesion_avg', 'grammar_avg', 'cefr_avg']
        transcript_columns = ['question_1_transcript', 'question_2_transcript',
                   'question_3_transcript','question_4_transcript',
                   'question_5_transcript']
        self.df_transcript = df_filter1[transcript_columns]
        self.df_assessment = df_filter1[assessment_columns]
        logging.debug('_________________________________________________________')
        logging.debug(f"Length After filter: {len(self.df)}")
        logging.debug(self.df_transcript.info())
        logging.debug(self.df_assessment.head(5))
        # logging.debug(self.df_assessment.iloc[1:3])
        logging.debug(self.df_transcript.iloc[0:1])
        logging.debug(f"Assessment Columns => [{','.join(self.df_assessment.columns.values)}]")

    def run_training(self):
        self.parse_training_input()
        logging.info(f"Running training for Target '{self.target}'")
        logging.debug(f"Evaluating assessments of type '{self.category}'")
        # For sanity check of training, use assessment columns 
        # df_train, df_test = train_test_split(self.df_transcript, test_size=0.2, random_state=1)
        df_train, df_test = train_test_split(self.df_assessment, test_size=0.2, random_state=1)
        logging.debug("size of training set vs size of test set", len(df_train), "/", len(df_test))
        # *** solve for 'target' ex. 'vocab_avg'
        y_train = df_train[self.target].values
        self.y_test = df_test[self.target].values
        del df_train[self.target]
        del df_test[self.target]
        X_train = df_train
        X_test = df_test   
        X_train.shape

        # cb = CatBoostRegressor(n_estimators=1800, max_depth = 12, random_state=42)
        cb = CatBoostRegressor(max_depth = 12, random_state=42)
        cb.fit(X_train, y_train)
        # Save the model to a file
        joblib.dump(cb, f'data/cb_modelA_for_{self.target}.joblib')
        print('Start Time: ', self.start_time)
        print('End Time:', time.asctime())

    def evaluate_training(self):
        logging.info(f"Evaulating training for Target '{self.target}'")
        logging.debug(f"Evaluating assessments of type '{self.category}'")

    def run_assessment_regression(self):
        logging.info("Running assessment regression")
        use_pkl_model = True
        run_regression = True
        fname = self.model_files.get(ModelAlgos.LINREGRESSION)
        model_string = model_labels.get(ModelAlgos.LINREGRESSION)

        # Create a linear regression model and fit it to the data
        if use_pkl_model and os.path.exists(fname):
            run_regression = False
        if run_regression:
            # solve for missing values in y in 2nd column
            # Split data into X and y
            X = self.df_assessment[self.lin_regr_cols['predictor']].values.reshape(-1, 1)
            y = self.df_assessment[self.lin_regr_cols['response']].values
            logging.info(f"Creating model {model_string} using assessment predict")
            model = LinearRegression().fit(X[~pd.isna(y)], y[~pd.isna(y)])

        if use_pkl_model: 
            if run_regression:
                logging.info(f"Saving lin. regr. model to disk as {fname}")
                # Save the model to disk
                with open(fname, 'wb') as f:
                    pickle.dump(model, f)
            else:
                # Load the model from disk
                logging.info(f"Loading {model_string} model from disk")
                with open(fname, 'rb') as f:
                    model = pickle.load(f)

        if self.input_data is not None:
            logging.info(f"Predicting missing values in {self.input_data}")
            df = pd.read_excel(self.input_data, skiprows=1,
                              index_col=0, header=0,
                nrows=int(self.sim_max_assets)
            )
            # TODO vs usecols="A:C"
            # keep only the first three columns
            dfw = df.iloc[:, [0, 1, 2]]
            logging.debug("Length B4 assessment filter:", len(df))
            dfw.columns = ['assessment_predicted', 'relative_humidity', 'pressure']
            assessment_columns = ['assessment_predicted']
            # assessment predict dataframe
            df_assessment_fc = dfw.loc[:, assessment_columns]
            # add an empty column 'y' to the DataFrame to solve for missing values
            df_assessment_fc['y'] = pd.Series(dtype=float)

            logging.debug(dfw.info())
            logging.debug(dfw.head(4))
            logging.debug(dfw.iloc[30:40])
            X = df_assessment_fc[self.lin_regr_cols['predictor']].values.reshape(-1, 1)
            y = df_assessment_fc['y'].values

            # Use the model to predict the missing values of y
            # get X values for rows that are missing y
            X_missing = X[pd.isna(y)].reshape(-1, 1)
            y_missing = model.predict(X_missing).round(2)

            # Create a new DataFrame with the missing y values replaced by the predicted values
            # TODO still need this to be a class variable or is local enough?
            self.df_missing = df_assessment_fc[pd.isna(df_assessment_fc['y'])].copy()
            self.df_missing['y'] = y_missing
            # TODO remove this line after testing
            self.df_assessment_fc = df_assessment_fc

            # Print the updated data
            logging.debug("len of df_missing:", len(self.df_missing))
            logging.debug(self.df_missing.head(10))

            # format input data for evaluation prediction
            # dfw.insert(loc=0, column='assessment_actual', value=self.df_missing['y'])
            dfw = dfw.join(self.df_missing['y'], how='left')
            logging.debug('_________________________________________________________')
            logging.debug('*** With updated Assessment Evaluation')
            logging.debug("len dfw:", len(dfw))
            logging.debug(dfw.info())
            logging.debug(dfw.head(4))
            logging.debug(dfw.iloc[30:40])
            dfw.to_csv(get_file_loc('data/assessment_predictB.csv'))

            # align columns with evaluation prediction columns
            self.df_test = dfw.drop('assessment_predicted', axis=1).rename(columns={'y': 'assessment_actual'})
            colA = self.df_test.pop('assessment_actual')
            self.df_test.insert(0, colA.name, colA)
            self.df_test['month'] = 5
            self.df_test['hour'] = 23
            logging.debug('_________________________________________________________')
            logging.debug('*** Now prepped for evaluation prediction')
            logging.debug("len self.df_test:", len(self.df_test))
            logging.debug(self.df_test.info())
            logging.debug(self.df_test.head(4))


    def run_prediction(self):
        logging.info("Running prediction")
        y_test = None
        if hasattr(self, 'df_test'):
            X_test = self.df_test
            if self.compare_data is not None:
                dfc = pd.read_excel(self.compare_data, usecols=['evaluation_actual', 'evaluation_predicted'],
                                index_col=0, header=0, 
                    nrows=int(self.sim_max_assets)
                )
                logging.debug("dfc:", dfc.head(4))
                logging.debug("len dfc:", len(dfc))
                #  X = df_assessment_fc[self.lin_regr_cols['predictor']].values.reshape(-1, 1)

        else:
            df_train, self.df_test = train_test_split(
                self.df, test_size=0.2, random_state=1)
            y_test = self.df_test.evaluation_actual.values  # used to validate the model
            del self.df_test['evaluation_actual']
            X_test = self.df_test  # time periods missing evaluation_actual to solve for

        logging.debug("Shape of X_test:", X_test.shape)
        # assert(self.model_files.get(self.model_name) is not None, "Model file not found")
        if not isinstance(self.model_name, list):
            self.model_name = [self.model_name]
        for model in self.model_name:
            fname = self.model_files.get(model)
            model_string = model_labels.get(model)
            df = joblib.load(self.model_files.get(model))
            y_pred = df.predict(X_test)  # returns results of prediction
            self.df_results = pd.DataFrame(y_pred, index=X_test.index,
                                      columns=['evaluation_predicted'])
            self.df_results.join(self.df_test)
            self.df_results.to_csv(get_file_loc('data/pp_resultsB.csv'))
            logging.debug('_________________________________________________________')
            logging.debug('*** evaluation Prediction')
            logging.debug("len self.df_results:", len(self.df_test))
            logging.debug(self.df_results.info())
            logging.debug(self.df_results.head(4))

            if y_test is not None:
                # validate the model
                mae = mean_absolute_error(y_pred, y_test)
                mse = mean_squared_error(y_pred, y_test)
                rmse = np.sqrt(mse)
                nrmse = rmse/(max(y_test)-min(y_test))
                logging.info('_________________________________________________________')
                logging.info(f">  Loading Model Prediction {model_string} with {fname} at", time.asctime())
                logging.info('Mean Absolute Error (MAE): %.3f' % mae)
                logging.info('Mean Squared Error (MSE): %.3f' % mse)
                logging.info('Root Mean Squared Error (RMSE): %.3f' % rmse)
                logging.info('Normalized Root Mean Squared Error (NRMSE): %.3f' % nrmse)

    def upload_results(self):
        logging.info("Uploading results to database")
