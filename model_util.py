# -------------
#  GENERIC_CLASSES and DICTIONARIES
# -------------

import os.path
from enum import Enum

class ModelingStage(Enum):
    TRAINING = 1
    PREDICT = 2


class ModelAlgos(Enum):
    CATBOOST = 1
    RANDOMFOREST = 2
    LINREGRESSION = 3
    # LightGBM,  XGBoost

model_labels = {
    ModelAlgos.CATBOOST: 'CatBoost',
    ModelAlgos.RANDOMFOREST: 'Random Forest',
    ModelAlgos.LINREGRESSION: 'Linear Regression',
}

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

