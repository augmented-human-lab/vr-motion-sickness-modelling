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
from clean_v3 import *
from generate_explanation import *
from aggregation import *

#edit this as per the requirements and run pipeline.py to get the explanations for the session in the specified path
path_to_session='/home/dinithi/vr-motion-sickness-modelling/Modelling/data/44_1_Arena_Clash'
# path_to_session='/data/VR_NET/folders/Arena_Clash/44_1_Arena_Clash/' #the folder specified here shouldhave the data file saved as 'data_files.csv' inside.
save_file_at='/home/sharedFolder/pipeline' #please specify this path, where you want the CSVs and the cleaned file to be saved
dataset='dataset3.csv' #Which type pf agregation you want to perform 
root_path='/home/dinithi/vr-motion-sickness-modelling/Modelling' #root folder path in you local machine

aggre_point=int(dataset[-5])-1
cleaned_session=clean(path_to_session)
cleaned_session.to_csv(os.path.join(save_file_at, 'data_file_4.csv'), index=False)
df_s, indexes=create_frames_1(cleaned_session, aggre_point)
df_sum=aggregate_df(cleaned_session,aggre)
gen_explanation_new('XGBoost', dataset, save_file_at, 'log_1720005500', df_s, df_sum, root_path, indexes, aggre_point)#change the model file and the log file appropriately if you train better models