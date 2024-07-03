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


game_abbs={'Earth_Gym': 1,
 'Wild_Quest': 2,
 'Roommate': 3,
 'Waffle_Restaurant': 4,
 'Superhero_Arena': 5,
 'Army_Men': 6,
 'Bobber_Bay_Fishing': 7,
 'Citadel': 8,
 'Mars_Miners': 9,
 'American_Idol': 10,
 'Kawaii_Daycare': 11,
 'Giant_Paddle_Golf': 12,
 'City_Parkour': 13,
 'Wake_the_Robot': 14,
 'The_aquarium': 15,
 'UFO_crash_site_venue': 16,
 'Kowloon': 17,
 'Arena_Clash': 18,
 'Scifi_Sandbox': 19,
 'Super_Rumble': 20,
 'Slash_RPG': 21,
 'Pirate_Life': 22,
 'Zombie': 23,
 'Horizon_Boxing': 24,
 'VR_Classroom': 25,
 'Venues': 26,
 'Metdonalds': 27,
 '3D_Play_House': 28,
 'Out_Of_Control': 29,
 'Geometry_Gunners': 30,
 'Barnyard': 31,
 'Canyon_Runners': 32,
 'Titanic_Simulation': 33,
 'Creature_Feature': 34}

cleaned_dfs = []

def panel(df, user_id, game_id,order):
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'time'}, inplace=True)
    df['user_id'] = user_id
    df['game_id'] = game_id  
    df['order'] = order 
    return df

with open('/home/dinithi/vr-motion-sickness-modelling/Data_Pre_Processing/Econ/games_used.json', 'r') as f:
    games = json.load(f)

root_dir='/data/VR_NET/folders'

logger.info('--------------------------------aggregation starting----------------------------------------------------------------------------------')

# games={'Arena_Clash': ['45_2_Arena_Clash',
#   '43_2_Arena_Clash']}

for game in games:
    logger.info(f'***************{game}********************')
    for sess in games[game]:
        logger.info(f'*******{sess}********')
        sess_path=os.path.join(root_dir, game, sess)
        user_id=int(sess.split('_')[0])
        game_id=game_abbs[game]
        order=int(sess.split('_')[1])
        # print(sess_path)
        try:
            df_clean=clean_v2(sess_path)
            path2=os.path.join(sess_path,"data_file_4.csv")
            df_clean.to_csv(path2, index=False)
            logger.info('successful saving')
        except:
            logger.info('failed saving')

            

# panel_df = pd.concat(cleaned_dfs, ignore_index=True)

# panel_df.to_csv('panel_data_fin.csv',  index=False)