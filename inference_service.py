import logging, os
import sys
from inference_service_sub import *

py_file_location = "/work/kobertsum/ext/models/"
sys.path.append(os.path.abspath(py_file_location))
sys.path.append( "/work/kobertsum/ext/models/")
py_file_location2 = "/work/kobertsum/src/models/"
sys.path.append(os.path.abspath(py_file_location2))
sys.path.append( "/work/kobertsum/src/models/")
sys.path.append("/data/nas/search/t3q-dl/work/kobertsum/src/model/")


import t3qai_client as tc
from t3qai_client import *


logger = logging.getLogger()
logger.setLevel('INFO')
 
def init_model():
    params = exec_init_model()
    logging.info('[hunmin log] the end line of the function [init_model]')
    return { **params }

def inference_dataframe(df, model_info_dict):
    result = exec_inference_dataframe(df, model_info_dict)
    logging.info('[hunmin log] the end line of the function [inference_dataframe]')
    return result
    
def inference_file(files, model_info_dict):
    result = exec_inference_file(files, model_info_dict)
    logging.info('[hunmin log] the end line of the function [inference_file]')
    return result