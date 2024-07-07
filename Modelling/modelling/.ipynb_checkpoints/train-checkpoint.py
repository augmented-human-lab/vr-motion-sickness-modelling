import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


# Set random seeds
np.random.seed(42)
random.seed(42)

global root_path
root_path='/home/vr-motion-sickness-modelling/Modelling'
# Ignore warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', '{:.10f}'.format)

time_now=str(int(time.time()))

models = {
    'LightGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
}

params = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'verbose' : [-1]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200, 400],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

train_games=['Earth_Gym', 'Wild_Quest', 'Roommate', 'Waffle_Restaurant', 'UFO_crash_site_venue', 'The_aquarium', 'Kowloon', 'Wake_the_Robot', 'Scifi_Sandbox', 'Citadel', 'Super_Rumble', 'Venues', 'Superhero_Arena', 'Army_Men', 'Bobber_Bay_Fishing', 'Mars_Miners', 'American_Idol', 'Metdonalds', '3D_Play_House', 'Kawaii_Daycare', 'Giant_Paddle_Golf', 'City_Parkour', 'Slash_RPG', 'Pirate_Life']
test_games=['Arena_Clash', 'Zombie', 'VR_Classroom', 'Horizon_Boxing', 'Titanic_Simulation']
val_games = ['Out_Of_Control', 'Geometry_Gunners', 'Barnyard', 'Canyon_Runners', 'Creature_Feature']

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

def model_grid(X_train, X_test, X_val, y_train, y_test, y_val, models, params, classification, path):
    best_models = {}
    for name in models:
        grid_search = GridSearchCV(models[name], params[name], cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
    # Evaluate the best models on the test set
    logger.info("Regression Results")
    for name, model in best_models.items():
        model_path=os.path.join(path, name, 'model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        y_prob = model.predict(X_test)
        mse = mean_squared_error(y_test, y_prob)
        logger.info(f"{name} Test MSE: {mse}")
        if classification=='Binary':
            evaluate_model_BC(model, X_train, X_val, X_test, y_train, y_test, y_val, path,name)
        if classification== 'Multiclass':
            evaluate_model_MC(model, X_train, X_val, X_test, y_train, y_test, y_val)
            
def log_results(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy= accuracy_score(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Recall: {recall}")
    logger.info(f"Precision: {precision}")
    logger.info(f"F1 Score: {f1}")
    return None

def log_results2(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average=None)
    precision = precision_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    accuracy= accuracy_score(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Recall: {recall}")
    logger.info(f"Precision: {precision}")
    logger.info(f"F1 Score: {f1}")
    return None
       
def evaluate_model_BC(model, X_train, X_val, X_test, y_train, y_test, y_val, path, name):
    y_prob_val= model.predict(X_val)
    threshold, f1=threshold_f1(y_prob_val,y_val)
    data = {'threshold': threshold}
    thesh_path=os.path.join(path, name, 'threshold.json')
    with open(thesh_path, 'w') as json_file:
        json.dump(data, json_file)
        # Predict class labels based on the threshold
    y_prob_test= model.predict(X_test)
    y_prob_train=model.predict(X_train)
    y_pred_train = (y_prob_train >= threshold).astype(int)
    y_pred_test = (y_prob_test >= threshold).astype(int)
    logger.info('*******----Stats for Train set----*******')
    log_results(y_train, y_pred_train)
    # Compute confusion matrix
    logger.info('*******----Stats for Test set----*******')
    log_results(y_test, y_pred_test)
    return None
    
def evaluate_model_MC(model, X_train, X_val, X_test, y_train, y_test, y_val):
    y_pred_train=np.round(model.predict(X_train))
    y_pred_test=np.round(model.predict(X_test))
    logger.info('*******----Stats for Train set----*******')
    log_results2(y_train, y_pred_train)
    # Compute confusion matrix
    logger.info('*******----Stats for Test set----*******')
    log_results2(y_test, y_pred_test)
    return None
     
def select_cols(dataset):
    cols=['IndTrig_L', 'IndTrig_R', 'HandTrig_L', 'HandTrig_R', 'Thumb_L_x',
       'Thumb_R_x', 'Thumb_L_y', 'Thumb_R_y', 'Head_Speed','LE_speed',
       'Head_Velocity_Change', 'Head_AngVel_Change', 'c_speed',
       'text_presence', 'change in brightness', 'diff_HE_yaw', 'diff_HE_roll',
       'diff_HE_pitch', 'c_acceleration1', 'c_acceleration2',
       'c_acceleration3', 'c_velocity1', 'c_velocity2', 'c_velocity3',
       'le_roll_change', 'le_pitch_change', 'le_yaw_change', 'h_roll_change',
       'h_pitch_change', 'h_yaw_change', 'c_roll_change', 'c_pitch_change',
       'c_yaw_change','time']
    if dataset=='dataset1.csv':
        # cols.remove('msr_1')
        cols.remove('time')
    # if dataset=='dataset3.csv':
    #     cols.append('msr_2')
    # if dataset=='dataset4.csv':
    #     cols.append('msr_2')
    #     cols.append('msr_3')
    return cols

def threshold_f1(y_prob, y_test):
    thresholds = np.arange(0, 1.01, 0.01)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_test, y_pred)
        f1_scores.append(score)

    # Convert to a DataFrame for easier manipulation
    thresholds_f1 = pd.DataFrame({'Threshold': thresholds, 'F1 Score': f1_scores})
    # Find the threshold with the lowest F1 score
    max_f1_row = thresholds_f1.loc[thresholds_f1['F1 Score'].idxmax()]
    max_threshold = max_f1_row['Threshold']
    max_f1_score = max_f1_row['F1 Score']
    return max_threshold, max_f1_score

def balance_classes(data, target_column, samples_per_class=30):
    balanced_data = pd.DataFrame()
    grouped = data.groupby(target_column)
    for class_label, group in grouped:
        sampled_group = group.sample(n=samples_per_class, random_state=42, replace=True)  # 
        balanced_data = pd.concat([balanced_data, sampled_group])
    return balanced_data

# Function to filter dataset by game list
def filter_by_games(df, session_data, game_list):
    sessions = []
    for game in game_list:
        if game in session_data:
            sessions.extend(session_data[game])
    return df[df['session'].isin(sessions)]

def classification(data, session_data, train_games, models, params, classtype, path, cols):
    data['msr_bin'] = (data['msr'] >= 3).astype(int)
    col_name='msr'
    # if classtype=='Binary':
    #     col_name='msr_bin'
    # else:
    #     col_name='msr'
    train_df = filter_by_games(data, session_data, train_games)
    value_counts = train_df[col_name].value_counts().min()
    train_df = balance_classes(train_df, target_column=col_name, samples_per_class=value_counts)
    
    test_df = filter_by_games(data, session_data, test_games)
    value_counts = test_df[col_name].value_counts().min()
    test_df = balance_classes(test_df, target_column=col_name, samples_per_class=value_counts)
    
    val_df = filter_by_games(data, session_data, val_games)
    value_counts = val_df[col_name].value_counts().min()
    val_df = balance_classes(val_df, target_column=col_name, samples_per_class=value_counts)
    
    if classtype=='Binary':
        col_name='msr_bin'
    
    X_train = train_df[cols].abs()
    y_train = train_df[col_name]

    X_test = test_df[cols].abs()
    y_test = test_df[col_name]

    X_val = val_df[cols].abs()
    y_val = val_df[col_name]
    
    model_grid(X_train, X_test, X_val, y_train, y_test, y_val, models, params, classtype, path)
    
    return None


def main(dataset):
    dataset_name=dataset[:-4]
    path=os.path.join(root_path,'modelling', dataset_name, 'log_'+time_now)
    log_path=os.path.join(path, 'models.log')
    global logger
    logger = setup_logger(log_path)
    dataset_path=os.path.join(root_path, 'data', dataset)
    data=pd.read_csv(dataset_path)

    cols=select_cols(dataset)

    with open(os.path.join(root_path,'sessions.json')) as f:
        session_data = json.load(f)
        
    logger.info('Multiclass Classification')
    # Split the data into training and testing sets
     
    classification(data, session_data, train_games, models, params, 'Multiclass', path, cols)
    
    logger.info('Binary Classification')
    
    classification(data, session_data, train_games, models, params, 'Binary', path, cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to process a dataset')
    parser.add_argument('dataset', metavar='DATASET', type=str,
                        help='Name of the dataset to process')
    args = parser.parse_args()
    main(args.dataset)