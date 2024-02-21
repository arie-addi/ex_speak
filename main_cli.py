#!/usr/bin/env python3

import os.path
import click
import logging
from model_util import ModelAlgos, ModelingStage, get_file_loc, model_labels
from model_language_quality import ModelLanguageQuality

# model Dictionary
modelD = {
    'modeling_stage': ModelingStage.PREDICT,
    'training_data': get_file_loc('data/speak_first_4.csv'),
    'training_audio' : get_file_loc('assets/audio'),
    'training_text' : get_file_loc('assets/text'),
}
modelP = {
    'input_data': get_file_loc('data/language_hints.csv'),
    # 'compare_data': get_file_loc('data/load_results.xlsx'),
    'model_name': [ModelAlgos.CATBOOST, 
                   # ModelAlgos.RANDOMFOREST
                    ],
    'model_files': {
        ModelAlgos.CATBOOST: get_file_loc('data/cb_modelAA.joblib'),
        ModelAlgos.RANDOMFOREST: get_file_loc('data/rf_modelA.joblib'),
        ModelAlgos.LINREGRESSION: get_file_loc('data/lr_temp_modelA.pkl')
    },
    'lin_regr_cols': {
        'predictor': 'rating_predicted',
        'response': 'rating_actual'
    }
}

# -----------------------------

@click.group()
def cli():
    pass

@cli.command()
@click.option('--category', type=int, help='Description of category')
@click.option('--asset', type=str, help='Description of asset')
@click.option('--log_level', type=str, default='INFO', help='Log level (default: INFO)')
def analyze(category, log_level, asset):
    """Analyze the given asset"""
    global modelD, modelP
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.debug(f"category: {category}")

    modelD['category'] = category
    modelD['asset'] = asset
    modelO = ModelLanguageQuality(**modelD)
    # modelO.run_prediction()
    logging.debug("modelD =", modelD)
    logging.info("Done with analyze()")

@cli.command()
@click.option('--use_online', is_flag=True, help='Set to True if online')
@click.option('--training_csv', default=None, type=str, help='Name of csv training file')
@click.option('--log_level', type=str, default='INFO', help='Log level (default: INFO)')
def run_training(use_online, test_file, log_level):
    """Train the language model"""
    global modelD, modelP
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.debug(f"use_online: {use_online}, training_file: {training_csv}, log_level: {log_level}")

    modelD['modeling_stage'] = ModelingStage.TRAINING
    modelO = ModelLanguageQuality(**modelD)
    modelO.run_training()
    logging.debug("modelD =", modelD)
    logging.info("Done with run_training()")

# -------------
# MAIN
# -------------

if __name__ == '__main__':
    g_this_dir = os.path.dirname(os.path.abspath(__file__))
    cli()
