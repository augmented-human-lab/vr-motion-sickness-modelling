import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import csv
import math
import json
import ast
import cv2
import logging
import warnings
# Ignore warnings
warnings.filterwarnings('ignore')

from clean_v3 import * 
from logger_config import setup_logger

# Get logger
logger = setup_logger()

cleaned_dfs = []

# def panel(df, user_id, game_id,order):
#     df.reset_index(drop=False, inplace=True)
#     df.rename(columns={'index': 'time'}, inplace=True)
#     df['user_id'] = user_id
#     df['game_id'] = game_id  
#     df['order'] = order 
#     return df

# with open('/home/dinithi/vr-motion-sickness-modelling/Data_Pre_Processing/Econ/games_used.json', 'r') as f:
#     games = json.load(f)

root_dir='/data/VR_NET/folders'
save_file_at=

logger.info('--------------------------------aggregation starting----------------------------------------------------------------------------------')

# games={'Arena_Clash': ['45_2_Arena_Clash',
#   '43_2_Arena_Clash']}
def pre_process(sess):
        logger.info(f'*******{sess}********')
        sess_path=os.path.join(root_dir, game, sess)
        # print(sess_path)
        try:
            df_clean=clean_v2(sess_path)
            path2=os.path.join(sess_path,"data_file.csv")
            logger.info('successful saving')
        except:
            logger.info('failed saving')

            

# panel_df = pd.concat(cleaned_dfs, ignore_index=True)

# panel_df.to_csv('panel_data_fin.csv',  index=False)