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
import warnings
import seaborn as sns
import random
from logger_config import setup_logger

# Get logger
logger = setup_logger()

np.random.seed(42)
random.seed(42)

# Ignore warnings
warnings.filterwarnings('ignore')


aggre={
'IndTrig_L' : 'sum',
'IndTrig_R' : 'sum',
'HandTrig_L' : 'sum',
'HandTrig_R' : 'sum',
'Thumb_L_x' : 'sum',
'Thumb_R_x' : 'sum',
'Thumb_L_y' : 'sum',
'Thumb_R_y' : 'sum',
'le_roll' : 'median_change',
'le_pitch' : 'median_change',
'le_yaw' : 'median_change',
'LE_speed' : 'median' ,
'h_roll' : 'median_change',
'h_pitch' : 'median_change',
'h_yaw' : 'median_change',
'Head_Speed' : 'median' , 
'Head_Velocity_Change' : 'median' ,
'Head_AngVel_Change' : 'median',
'c_roll' : 'median_change',
'c_pitch' : 'median_change',
'c_yaw' : 'median_change',
'c_speed' : 'median',
'text_presence' : 'sum',
'change in brightness' : 'sum',
'diff_HE_yaw' : 'median',
'diff_HE_roll' : 'median',
'diff_HE_pitch' : 'median',
'c_acceleration1' : 'median',
'c_acceleration2' : 'median',
'c_acceleration3' : 'median',
'c_velocity1' : 'median',
'c_velocity2' : 'median',
'c_velocity3' : 'median'}

# logger = setup_logger()

# Ignore warnings
warnings.filterwarnings('ignore')

def expand_cols(df, column_name):
    df[column_name]= df[column_name].apply(lambda x: ast.literal_eval(x))
    data_expanded = pd.DataFrame(df[column_name].tolist(), index=df.index)
    data_expanded.columns = [column_name+'1', column_name+'2', column_name+'3']
    df = df.drop(columns=[column_name]).join(data_expanded)
    return df

def change_diff(df, column_list):
    for column in column_list:
        df[column+'_change']=df[column]-df[column].shift(1)
    return df
    
def aggregate_df(df,aggre):
    df1=expand_cols(df,'c_acceleration')
    df2=expand_cols(df1,'c_velocity')
    change=[]
    for i in aggre.keys():
        if aggre[i]=='median_change':
            change.append(i)
            aggre.pop(i)
            aggre[i+'_change']='median'
    df=change_diff(df2,change)
    aggregated_df = df.agg(aggre)
    return pd.DataFrame(aggregated_df).transpose()

def create_frames_1(df_m, aggre,j):
    df=df_m.copy()
    index_list= df[df['MS_rating'].notna()].index.tolist()
    frame_values = df.loc[index_list, 'frame'].tolist()
    msr=df['MS_rating'].dropna().tolist()
    df3=df.iloc[0:1]
    count1=0
    for i in range(len(index_list) - 1):
        if count1==0:
            start_idx=0
            end_idx=1
            df2=df.iloc[start_idx:end_idx]
            aggre1=aggre.copy()
            df2=aggregate_df(df2.copy(), aggre1)
            dfs1=df2.copy()
        start_idx = index_list[i]
        end_idx = index_list[i + 1]  
        if count==1:
            start_idx+=1
        df2=df.iloc[start_idx:end_idx]
        aggre1=aggre.copy()
        df2=aggregate_df(df2.copy(), aggre1)
        df_c1=[dfs1,df2]
        dfs1 = pd.concat(df_c1, ignore_index=True)
        count1+=1
    dfs1['msr']=msr
    dfs1['session']=j
    dfs1['frame']=frame_values
    return dfs1
        


with open('/home/sharedFolder/sessions.json', 'r') as file:
    sessions = json.load(file)

dfs=[]
count=0

# sessions={'Earth_Gym': ['5_2_Earth_Gym',
#   '10_2_Earth_Gym']}


logger.info('--------------------------------aggregation starting----------------------------------------------------------------------------------')

for i in sessions.keys():
    logger.info(f'***************{i}********************')
    for j in sessions[i]:
        logger.info(f'*******{j}********')
        path=os.path.join('/data/VR_NET/folders/', i, j, 'data_file_2.csv')
        df1=pd.read_csv(path)
        df=df1.copy()
        try:
            df=create_frames_1(df, aggre, j)
            if count==0:
                dfs=df.copy()
            else:
                df_c=[dfs,df]
                dfs = pd.concat(df_c)
            count+=1
            logger.info('*******successfully saved********')
        except:
            logger.info('*******failed********')

dfs.to_csv('/home/sharedFolder/data/dataset2_4.csv')