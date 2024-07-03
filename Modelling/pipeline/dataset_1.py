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

from logger_config import setup_logger

# Get logger
logger = setup_logger()

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


# with open('/home/sharedFolder/sessions.json', 'r') as file:
#     sessions = json.load(file)

dfs=[]
count=0
session_saving={}


# for i in sessions.keys():
#     logger.info(f'***************{i}********************')
#     for j in sessions[i]:
#         logger.info(f'*******{j}********')
#         path=os.path.join('/data/VR_NET/folders/', i, j, 'data_file_2.csv')
#         df1=pd.read_csv(path)
#         df=df1.copy()
#         try:
#             msr=df1['MS_rating'].dropna().max()
#             aggre1=aggre.copy()
#             df=aggregate_df(df, aggre1) 
#             df['msr']=msr
#             df['session']=j
#             if count==0:
#                 dfs=df.copy()
#             else:
#                 df_c=[dfs,df]
#                 dfs = pd.concat(df_c, ignore_index=True)
#             session_saving[count]=path
#             count+=1
#             logger.info('*******successfully saved********')
#         except:
#             logger.info('*******failed********')

for i in sessions.keys():
    for j in sessions[i]:
        path=os.path.join('/data/VR_NET/folders/', i, j, 'data_file_2.csv')
        df1=pd.read_csv(path)
        df=df1.copy()
        try:
            msr=df1['MS_rating'].dropna().max()
            aggre1=aggre.copy()
            df=aggregate_df(df, aggre1) 
            df['msr']=msr
            df['session']=j
            # cleaned_dfs.append(df)
            if count==0:
                dfs=df.copy()
            else:
                df_c=[dfs,df]
                dfs = pd.concat(df_c, ignore_index=True)
            # session_saving[count]=path
            count+=1
        except:
            print('*******failed_to _aggregate********')
            
        
# dfs.to_csv('/home/sharedFolder/data/dataset1.csv', index=False)