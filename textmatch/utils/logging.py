# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from logging.handlers import TimedRotatingFileHandler


start_time = time.time()
local_time = time.localtime(start_time)

exec_day = time.strftime('%Y-%m-%d', local_time)  # Execution date
exec_hour = time.strftime('%H', local_time)
exec_minute = time.strftime('%M', local_time)

def log_init(log_path):
    fmt_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig( level=logging.INFO, format=fmt_str)
    log_file_handler = logging.handlers.TimedRotatingFileHandler(log_path, when='D', interval=1, backupCount=3)
    log_file_handler.suffix = "%Y%m%d_%H%M%S.log"
    log_file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt_str)
    log_file_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(log_file_handler)
    logging.info('exec_day :{}, exec_hour :{}, exec_minute :{}'.format(exec_day, exec_hour, exec_minute))
    return logging
