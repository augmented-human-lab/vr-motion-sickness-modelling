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

np.random.seed(42)
random.seed(42)

# Ignore warnings
warnings.filterwarnings('ignore')
# global aggre

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

def change_diff(df, column_list):
    for column in column_list:
        df[column+'_change']=df[column]-df[column].shift(1)
    return df
    
def aggregate_df(df,aggre):
    change=[]
    for i in aggre.keys():
        if aggre[i]=='median_change':
            change.append(i)
            aggre.pop(i)
            aggre[i+'_change']='median'
    df=change_diff(df,change)
    aggregated_df = df.agg(aggre)
    return pd.DataFrame(aggregated_df).transpose()

def get_index_list(df):
    # Starting timestamp
    start_timestamp = df['timestamp'].iloc[0]
    end_timestamp = df['timestamp'].iloc[-1]
    # List to hold the indices of each one-minute interval
    minute_indices = []
    # Loop through each minute interval
    for i in range(0, int((df['timestamp'].iloc[-1] - start_timestamp) // 60000) + 1):
        target_timestamp = start_timestamp + i * 60000
        closest_index = (df['timestamp'] - target_timestamp).abs().idxmin()
        minute_indices.append(closest_index)
    return minute_indices

def create_frames_1(df_m, aggre_point):
    df=df_m.copy()
    dfs1=None
    index_list= get_index_list(df_m)
    count1=0
    for i in range(len(index_list) - aggre_point):
        start_idx = index_list[i]
        end_idx = index_list[i + aggre_point]  
        df2=df.iloc[start_idx:end_idx]
        aggre1=aggre.copy()
        df2=aggregate_df(df2.copy(),aggre1)
        if count1 ==0:
            dfs1=df2.copy()
        else:
            df_c1=[dfs1,df2]
            dfs1 = pd.concat(df_c1, ignore_index=True)
        count1+=1
    dfs1['time'] = dfs1.index + 1
    if dfs1 is None:
        print('Please check your data to be longer than ', aggre_point, ' minutes')
        return None
    else:
        return dfs1, index_list
        
