#!/usr/bin/env python3

import os.path
import click
import logging
from model_util import ModelAlgos, ModelingStage, get_file_loc 
from model_util import model_labels, modelD
from model_language_quality import ModelLanguageQuality

@click.group()
def cli():
    pass

@cli.command()
@click.option('--category', type=str, default='text', help='Category to evaluate: audio or text')
@click.option('--model', type=str, default='CATBOOST_BL', help='Model to use')
@click.option('--asset', type=str, default=None, help='Name of specific asset to analyze')
@click.option('--target', default="vocab_avg", type=str, help='Target variable desired to predict')
@click.option('--test_data', default=None, type=str, help='Name of csv file to test')
@click.option('--eval_model', is_flag=True, help='Set to True if wanting to evaluate model as well')
@click.option('--log_level', type=str, default='INFO', help='Log level (default: INFO)')
def analyze(category, model, asset, target, test_data, eval_model, log_level):
    """Analyze the given asset"""
    global modelD
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=log_level.upper(), format='[%(levelname)s] %(message)s')
    logging.debug(f"category: {category}")

    # see default values in global modelD dict. definition
    modelD['model_choice'] = model
    modelD['modeling_stage'] = ModelingStage.PREDICT
    modelD['category'] = category
    modelD['asset'] = asset
    modelD['target'] = target
    if test_data:
        modelD['test_data'] = test_data
    modelO = ModelLanguageQuality(**modelD)
    # modelO.run_prediction()
    modelO.run_model()
    logging.debug("Done with analyze()")

@cli.command()
@click.option('--category', type=str, default='text', help='Category to evaluate: audio or text')
@click.option('--model', type=str, default='CATBOOST_BL', help='Model to use')
@click.option('--training_data', default=None, type=str, help='Name of csv training file')
@click.option('--target', default="vocab_avg", type=str, help='Target variable desired to predict')
@click.option('--log_level', type=str, default='INFO', help='Log level (default: INFO)')
def run_training(category, model, training_data, target, log_level):
    """Train the language model"""
    global modelD 
    logger = logging.getLogger(__name__)
    # logging.basicConfig(level=log_level.upper(), format='%(asctime)s [%(levelname)s] %(message)s')
    logging.basicConfig(level=log_level.upper(), format='[%(levelname)s] %(message)s')
    logging.debug(f"model: {model}, training_file: {training_data}, log_level: {log_level.upper()}")

    # see default values in global modelD dict. definition
    modelD['model_choice'] = model
    modelD['modeling_stage'] = ModelingStage.TRAINING
    modelD['category'] = category
    if training_data:
        modelD['training_data'] = training_data
    modelD['target'] = target
    modelO = ModelLanguageQuality(**modelD)
    modelO.train_model()
    modelO.run_model()
    logging.debug("Done with run_training()")

# -------------
# MAIN
# -------------

if __name__ == '__main__':
    g_this_dir = os.path.dirname(os.path.abspath(__file__))
    cli()
