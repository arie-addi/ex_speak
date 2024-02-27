# -------------
#  GENERIC_CLASSES and DICTIONARIES
# -------------

import os.path
from enum import Enum
import zlib

class ModelingStage(Enum):
    TRAINING = 1
    PREDICT = 2


class ModelAlgos(Enum):
    CATBOOST = 1
    RANDOMFOREST = 2
    LINREGRESSION = 3
    # BaseLine evaluation based on correlation with other measurements 
    CATBOOST_BL = 4
    # LightGBM,  XGBoost

# -------------
#  UTIL_FUNCTIONS
# -------------
g_this_dir = ""

def get_file_loc(location):
    # at root directory if not url or location starts with a slash
    global g_this_dir

    if (
        any(':' in ltr for ltr in location.split())
        or location[0] == '/'
    ):
        return location
    else:
        return os.path.join(g_this_dir, location)

def crc32_hash(text):
    if isinstance(text, str):
        return zlib.crc32(text.encode())
    else:
        return text

# -----------------------------

model_labels = {
    "CATBOOST": 'CatBoost',
    "RANDOMFOREST": 'Random Forest',
    "LINREGRESSION": 'Linear Regression',
    "CATBOOST_BL": 'CatBoost',
}

# model Dictionary
modelD = {
    'training_data': get_file_loc('data/SpeakNow_test_data.csv'),
    # 'training_data': get_file_loc('data/speak_first_4.csv'),
    'training_audio' : get_file_loc('assets/audio'),
    'training_text' : get_file_loc('assets/text'),
    'input_data': get_file_loc('data/language_hints.csv'),
    # 'compare_data': get_file_loc('data/load_results.xlsx'),
    'model_files': {
        'CATBOOST': {
            'vocab_avg' : get_file_loc('data/model-CatBoost_for-vocab_avg_type-text.joblib')
            },
        'RANDOMFOREST': {
            'vocab_avg' : get_file_loc('data/rf_modelA.joblib'),
            },
        'LINREGRESSION': {
            'vocab_avg' : get_file_loc('data/lr_temp_modelA.pkl')
            },
        'CATBOOST_BL': {
            'vocab_avg' : get_file_loc('data/model-CatBoost_for-vocab_avg_type-text_BL.joblib')
            },
    },
}

