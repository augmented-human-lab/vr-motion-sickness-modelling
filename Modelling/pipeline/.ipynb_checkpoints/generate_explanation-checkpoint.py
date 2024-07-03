import argparse
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
import time
import logging
import warnings
import seaborn as sns
import pickle
import lightgbm as lgb
import xgboost as xgb
import random
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

random.seed(42)
np.random.seed(42)
# from logger_config import setup_logger
global path_for_datafile

# Ignore warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', '{:.10f}'.format)

time_now=str(int(time.time()))

def setup_logger(fname):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Prevent adding multiple handlers to the logger
    if not logger.hasHandlers():
        fh = logging.FileHandler(fname)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger


def select_cols(dataset):
    cols=['IndTrig_L', 'IndTrig_R', 'HandTrig_L', 'HandTrig_R', 'Thumb_L_x',
       'Thumb_R_x', 'Thumb_L_y', 'Thumb_R_y', 'Head_Speed','LE_speed',
       'Head_Velocity_Change', 'Head_AngVel_Change', 'c_speed',
       'text_presence', 'change in brightness', 'diff_HE_yaw', 'diff_HE_roll',
       'diff_HE_pitch', 'c_acceleration1', 'c_acceleration2',
       'c_acceleration3', 'c_velocity1', 'c_velocity2', 'c_velocity3',
       'le_roll_change', 'le_pitch_change', 'le_yaw_change', 'h_roll_change',
       'h_pitch_change', 'h_yaw_change', 'c_roll_change', 'c_pitch_change',
       'c_yaw_change','msr_1','time']
    if dataset=='dataset1.csv':
        cols.remove('msr_1')
        cols.remove('time')
    if dataset=='dataset3.csv':
        cols.append('msr_2')
    if dataset=='dataset4.csv':
        cols.append('msr_2')
        cols.append('msr_3')
    return cols

def get_model(modelname,dataset, log):
    model_path=os.path.join('/home/sharedFolder/modelling/', dataset[:-4], log, modelname, 'model.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    threshold_path=os.path.join('/home/sharedFolder/modelling/', dataset[:-4], log, modelname, 'threshold.json')
    # Read the threshold value from the JSON file 
    with open(threshold_path, 'r') as json_file: 
        data = json.load(json_file) 
        threshold = data['threshold'] 
    return model, threshold
    
def get_prediction(model,X,threshold): 
    y_prob = model.predict(X)
    y_pred= (y_prob >= threshold).astype(int)
    return y_pred
    
def get_explanation(model, data_filtered, y_pred):
    count=0
    frames=data_filtered['frame']
    X=data_filtered[cols].abs()
    for i in range(len(X)):
        if y_pred[i]==1:
            explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=cols, class_names=['msr'], mode='regression')
            conditions = explainer.explain_instance(X[i], model.predict)
            count+=1
            reason, range_values=find_explanation(conditions)
            df = pd.DataFrame({'Reason': reason, 'Range': range_values})
            df['Start']=frame[i]
            df['End']=frame[i+1]
            if count==0:
                df2=df
            else:
                df_c1=[df2,df]
                df2 = pd.concat(df_c1, ignore_index=True)
    return df2

def find_than_signs(text):
    count=0
    for i in text:
        if i=="<" or i=='>':
            count+=1
    return count

def find_explanation(conditions):
    # print(conditions)
    variables=[]
    range=[]
    for condition, coefficient in conditions:
        if coefficient > 0:
            condition=str(condition)
            if 'msr' not in condition and 'time' not in condition:
                # print(condition)
                list1=condition.split(' ')
                # print(list1)
                count=find_than_signs(condition)
            
                if count==1:
                    variable=list1[0]
                    if '<' in condition:
                        range.append('low')
                    else:
                        range.append('high')
                else:
                    
                    variable=list1[2]
                    if variable.startswith("change"):
                        if float(list1[0])==0.0 and float(list1[6])==0.0:
                            range.append('low')
                        else:
                            range.append('ranging from '+list1[0]+' to '+ list1[6])
                    else:
                        if float(list1[0])==0.0 and float(list1[4])==0.0:
                            range.append('low')
                        else:
                            range.append('ranging from '+list1[0]+' to '+ list1[4])
                # print(list1[0])
                # print(condition)
                if variable.startswith("change"):
                    if count==1:
                        variable=list1[0]+' '+list1[1]+' '+list1[2]
                        variables.append(variable)
                    else:
                        variable=list1[2]+' '+list1[3]+' '+list1[4]
                        variables.append(variable)
                elif 'change' in variable:
                    variables.append(variable[:-7])
                elif 'Change' in variable:
                    variables.append(variable[:-7])
                else:
                    variables.append(variable)
                # print(variables)
    return variables, range

def find_explanation_sum(conditions):
    statements=[]
    range1=[]
    dflong=pd.read_csv('/home/sharedFolder/modelling/Codebook.csv')
    for condition, coefficient in conditions:
        list1=condition.split(' ')
        count=find_than_signs(condition)
        if coefficient > 0:
            condition=str(condition)
            if count==1:
                variable=list1[0]
                if list1[0]=='change':
                    variable=list1[0]+' '+list1[1]+' '+list1[2]
                elif ('change' in variable) or ('Change' in variable):
                    variable=variable[:-7]
                long_name=get_long_name([variable], dflong)
                if '<' in condition:
                    statements.append(long_name[0] + ' is low')
                    range1.append('positive')
                else: 
                    statements.append(long_name[0] + ' is high')
                    range1.append('positive')

        elif coefficient < 0:
            condition=str(condition)
            if count==1:
                variable=list1[0]
                if list1[0]=='change':
                    variable=list1[0]+' '+list1[1]+' '+list1[2]
                elif ('change' in variable) or ('Change' in variable):
                    variable=variable[:-7]
                long_name=get_long_name([variable], dflong)
                if '<' in condition:
                    statements.append(long_name[0] + ' is low')
                    range1.append('negative')
                else:
                    statements.append(long_name[0] + ' is high')
                    range1.append('negative')
    return statements, range1

# Define the lookup function
def get_long_name(column_names, df):
    if column_names is None:
        return None
    else:
        # Use .isin() to filter rows where Column is in the list of column_names
        matching_rows = df[df['Column'].isin(column_names)]
        # Reindex the result to match the order of the input column_names
        matching_rows = matching_rows.set_index('Column').reindex(column_names)
        # Get the Long_Name values as a list, replacing NaNs with None
        long_names = matching_rows['Long_Name'].where(pd.notnull(matching_rows['Long_Name']), None).tolist()
        return long_names

def get_min_explanation(y_pred, X, dataful, model, indexes, frame,cols, set1):
    # print(frame)
    count=0
    df2=None
    explainer = lime.lime_tabular.LimeTabularExplainer(dataful.values, feature_names=cols, class_names=['msr'], mode='regression')
    dflong=pd.read_csv('/home/sharedFolder/modelling/Codebook.csv')
    for i in range(len(X)):
        if y_pred[i]==1:
            
            conditions = explainer.explain_instance(X.loc[indexes[i]], model.predict)

            reason, range_values=find_explanation(conditions.as_list())
            long_reason= get_long_name(reason, dflong)
            df = pd.DataFrame({'Reason': reason, 'LongReason': long_reason, 'Range': range_values})
            if count==0:
                df2=df
                df['Start']=frame[i]
                df['End']=frame[i+set1]
            else:
                df['Start']=frame[i]
                df['End']=frame[i+set1]
                df_c1=[df2,df]
                df2 = pd.concat(df_c1, ignore_index=True)
            count+=1
    return df2

def get_summary(session):
    model, threshold=get_model('LightGBM','dataset1.csv', 'log_1719903504')
    data2=pd.read_csv('/home/sharedFolder/data/dataset1.csv')
    cols1=select_cols('dataset1.csv')
    data2_1=data2[cols1].abs()
    X_1=data2[data2['session']==session]
    explainer = lime.lime_tabular.LimeTabularExplainer(data2_1.values, feature_names=cols1, class_names=['msr'], mode='regression')
    conditions = explainer.explain_instance(X_1[cols1].abs().values[0], model.predict)
    statement, range1=find_explanation_sum(conditions.as_list())
    y_pred1= model.predict(X_1[cols1].abs())
    if y_pred1[0]<0:
        y_pred1[0]=0
    if y_pred1[0]>1:
        y_pred1[0]=1
    # s1='Overall motion sickness score is '+ str(y_pred1[0]) + ' (the score is in the range of 0-1, where 1 is the highest)'
    s1='Overall motion sickness score is'
    statement.insert(0, s1)
    range1.insert(0, str(y_pred1[0]))
    return statement, range1

def gen_explanation(modelname, dataset, session, log):
    with open("/home/sharedFolder/sessions.json", 'r') as file:
        games = json.load(file)
    game_name = '_'.join(session.split('_')[2:])
    if int(dataset[-5])>1:
        set1=int(dataset[-5])-1
    else:
        set1=int(dataset[-5])
    
    if (game_name not in games.keys()) or (session not in games[game_name]):
        print("Incorrect gaming session, check whether your session is included in Modelling/sessions.json")
        return None
    path_for_datafile=os.path.join('/data/VR_NET/folders/', game_name, session, 'data_file_2.csv')
    full_data_file=pd.read_csv(path_for_datafile)
    cols=select_cols(dataset)
    data_path=os.path.join('/home/sharedFolder/data', dataset)
    data=pd.read_csv(data_path)
    data_filtered=data[data['session']==session]
    indexes = data_filtered.index
    filtered_df_for_msr = full_data_file[full_data_file['MS_rating'].notnull()]
    # print(filtered_df_for_msr['MS_rating'])
    frame= filtered_df_for_msr['frame'].tolist()
    X=data_filtered[cols].abs()
    model, threshold= get_model(modelname,dataset, log)
   
    root_path1=os.path.join('/home/sharedFolder', game_name, session)
    save_file_at=os.path.join(root_path1, 'Explanation.csv')
    os.makedirs(os.path.dirname(save_file_at), exist_ok=True)
    df=None
    
    if len(frame)>set1+1:
        y_pred= get_prediction(model,X,threshold)
        df=get_min_explanation(y_pred, X, data[cols].abs(), model, indexes, frame,cols, set1)

        if df is None:
            # output_file = os.path.join(root_path1,'Explanation.csv')
            with open(save_file_at, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['No high motion sickness areas detected'])
                print('No high motion sickness areas detected')
        else:
            for i in range(1, len(df)):
                for j in range(i+1, len(df)):
                    if df.loc[i, "Reason"] == df.loc[j, "Reason"] and df.loc[j, "Start"] < df.loc[i, "End"]:
                        df.loc[j, "Start"] = df.loc[i, "End"]

                df.to_csv(save_file_at, index=False)
    else:
        with open(save_file_at, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Not enough gameplay minutes to give a prediction'])
                print('Not enough gameplay minutes to give a prediction')
        
    
    statement, range1=get_summary(session)
    file_path = os.path.join(root_path1,'summary.csv')
    sum_data = {'summary': statement, 'condition': range1}

    # Create the DataFrame
    df_sum = pd.DataFrame(sum_data)
    df_sum.to_csv(file_path, index=False)
        

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Script to process a dataset')
#     parser.add_argument('--dataset', metavar='DATASET', type=str,
#                         help='Name of the dataset to process', default='dataset3.csv')
#     parser.add_argument('--session', metavar='SESSION', type=str,
#                         help='Name of the session that you want to explain')
#     parser.add_argument('--modelname', metavar='MODEL', type=str,
#                         help='Name of the model you need the explanation from', default='GradientBoosting')
#     parser.add_argument('--log', metavar='LOG', type=str,
#                         help='the log file you need the explanation from', default='log_1719971872')
#     args = parser.parse_args()
    
#     main(args.modelname, args.dataset, args.session, args.log)