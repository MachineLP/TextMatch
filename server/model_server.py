# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  albert train
   Author :       machinelp
   Date :         2020-06-04
-------------------------------------------------

'''

import os
import sys
import time
import platform
import argparse
import cloudpickle
import numpy as np
import mlflow.pyfunc
from pyspark.sql import DataFrame
from sklearn.externals import joblib
from textmatch.utils.logging import logging
from textmatch.config.constant import Constant as const
from textmatch.models.text_embedding.model_factory_sklearn import ModelFactory


cur_abs_dir = os.path.dirname(os.path.abspath(__file__))
code_home = cur_abs_dir
sys.path.insert(0, code_home)
logging.info( "[model_server] python version:{}".format( platform.python_version() ) )
logging.info( "[model_server] code_home:{}".format( code_home ) )

start = time.time()
exec_time = time.time() - int(time.time()) % 900
local_time = time.localtime(exec_time - 30 * 60)
exec_day = time.strftime('%Y-%m-%d', local_time)
exec_hour = time.strftime('%H', local_time)
exec_minute = time.strftime('%M', local_time)


class TextMatchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, experiment_name, version_name):
        self.experiment_name = experiment_name
        self.version_name = version_name
        self.wordstest_dict = {"0":"我去玉龙雪山并且喜欢玉龙雪山玉龙雪山", "1":"我在玉龙雪山并且喜欢玉龙雪山", "2":"我在九寨沟", "3":"你好"}   #["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
        self.mf = ModelFactory( match_models=['bow', 'tfidf', 'ngram_tfidf'] )

    def load_context(self, context):
        # wordstest_dict = context.artifacts["wordstest_dict"]
        self.mf.init(words_dict=self.wordstest_dict, update=True)

    def predict(self, context, model_input):
        res = self.mf.predict(model_input["text"].values[0])
        logging.info( "[TextMatchWrapper] res:{}".format( res ) )
        return res



# 模型预测主流程
def model_server(experiment_name, version_name, args): #, model_path='./data/'):

    artifacts = {
        "train_model": os.path.join(const.BOW_DIC_PATH )
    }
    if args.local_store:
        mlflow.pyfunc.save_model(path=args.model_file,
                                python_model=TextMatchWrapper(experiment_name, version_name),
                                artifacts=artifacts)
    else:
        mlflow.pyfunc.log_model(artifact_path=args.model_file,
                                 python_model=TextMatchWrapper(experiment_name, version_name),
                                 artifacts=artifacts)



def parse_argvs():
    parser = argparse.ArgumentParser(description='textmatch ---- 模型线上部署')
    parser.add_argument("--experiment_name", help="实验名称")
    parser.add_argument("--version_name", help="版本名称")
    parser.add_argument("--model_file", help="模型存储路径",default='model')
    parser.add_argument("--local_store", help="是否本地存储",action='store_true', default=True)
    args = parser.parse_args()
    logging.info( "[model_predictor] args:{}".format( args ) )

    return parser, args

# python model_server.py --experiment_name "textmatch" --version_name "001" --model_file "textmodel"
# mlflow models serve -m /Users/qudian/Desktop/TextMatch/textmodel/ -h 0.0.0.0 -w 3 -p 5000 --no-conda
if __name__ == '__main__':
    parser, args = parse_argvs()
    # 输入参数解析
    experiment_name = args.experiment_name      # 实验名称
    input_version_name = args.version_name      # 输入的版本名称
    model_server(experiment_name=experiment_name, version_name=input_version_name, args=args)

    end = time.time()
    logging.info( "运行时长: {}s".format( int(end - start) ) )

