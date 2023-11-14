# train_sub.py

import t3qai_client as tc
from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_DATA_PATH, T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH, T3QAI_INIT_MODEL_PATH

# Imports
import os
import numpy as np
import logging

## user algorithm 
# T3Q.ai 공통, 알고리즘 파라미터 불러오기(dictionary 형태)
params = tc.train_load_param()
logging.info('params : {}'.format(params))
        
# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[Search log] Files and directories in {} :'.format(path))
    logging.info('[Search log] dir_list : {}'.format(dir_list))


def exec_train():
    logging.info('[Search log] the start line of the function [exec_train]')
    logging.info('[Search log] T3QAI_TRAIN_DATA_PATH : {}'.format(T3QAI_TRAIN_DATA_PATH))
    
    ## user algorithm 
    # T3Q.ai 공통, 알고리즘 파라미터 불러오기(dictionary 형태)
    params = tc.train_load_param()
    logging.info('[Search log] params : {}'.format(params))
        
    
    logging.info('[Search log] ★ Files and directories in [T3QAI_TRAIN_MODEL_PATH]')
    list_files_directories(T3QAI_TRAIN_MODEL_PATH)
    
    logging.info('[Search log] ★ Files and directories in [T3QAI_MODULE_PATH]')
    list_files_directories(T3QAI_MODULE_PATH)
    
    logging.info('[Search log] ★ Files and directories in [T3QAI_INIT_MODEL_PATH]')
    list_files_directories(T3QAI_INIT_MODEL_PATH)
    
    logging.info('[Search log] ★ Files and directories in [T3QAI_TRAIN_OUTPUT_PATH]')
    list_files_directories(T3QAI_TRAIN_OUTPUT_PATH)
    
    #현재 디렉토리 경로가져오기
    logging.info('[Search log] ★  os.getcwd()')
    list_files_directories(os.getcwd())
    # os.chdir('mods')
    # os.chdir('algo30')
    # os.chdir('1')
    # /data/aip/logs/t3qai/000820703792611374/mods/algo_30/1  자동으로 들어와짐
    
    #######################################################################
    # make_data (json > pt)
    n_cpus = str(params.get('n_cpus', '4')) # params['n_cpus'])
    folder_path = str(params.get('folder_path', '1103_1718')) # params['folder_path'])
    model_step_path = str(params.get('model_step_path', '10000')) # params['folder_path'])
    user_task_input = str(params.get('task','train'))
    
    logging.info("[dtype user_task_input] : {}".format(user_task_input))
    logging.info("[dtype user_task_input] : {}".format(user_task_input.__class__))
    logging.info("[model_step_path]:{}".format(model_step_path))
    
    logging.info("[loggin start]")
    if user_task_input.isdigit():
        logging.info("check... input type")
    else:
        if 'make_data' == user_task_input:
            os.system(f"python3 main.py -task make_data -n_cpus {n_cpus}")
        elif 'train' == user_task_input:
            logging.info("[loggin train]")
            os.system(f"python3 main.py -task train -target_summary_sent ext -visible_gpus 0")
        elif 'valid' == user_task_input:
            os.system(f"python3 main.py -task valid -model_path {folder_path}")
        elif 'test' == user_task_input:
            os.system(f"python3 main.py -mode test -test_from {folder_path}/model_step_{model_step_path}.pt -visible_gpus 0")        
        else:
            logging.info("check task of user input : [make_data, train, valid, test]")
            
    

###########################################################################
## exec_train() 호출 함수 끝
###########################################################################
