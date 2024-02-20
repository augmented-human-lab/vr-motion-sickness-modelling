import pandas as pd
import os
import shutil
import logging

output_dir="/data/VR_NET/folders/"
input_dir="/data/VR_NET/data"

arr_keys={
    "pose":['head_dir',
  'head_pos',
  'head_vel',
  'head_angvel',
  'left_eye_dir',
  'left_eye_pos',
  'left_eye_vel',
  'left_eye_angvel',
  'right_eye_dir',
  'right_eye_pos',
  'right_eye_vel',
  'right_eye_angvel'],
 "gaze":['left_eye', 'right_eye', 'confidence', 'is_valid'],
 "control": ['ConnectedControllerTypes',
  'Buttons',
  'Touches',
  'NearTouches',
  'IndexTrigger',
  'HandTrigger',
  'Thumbstick'],
 "scene":['object_name', 'bounds', 'm_matrix','camera_name', 'p_matrix', 'v_matrix'],
    "video":["video"],
"MS_rating":["MS_rating"]}

def setup_logger():
    # Configure logging
    logging.basicConfig(filename='data_arranging.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define logger
    logger = logging.getLogger(__name__)
    return logger



def remove_columns(arr,df):
    if arr[0]==0:
        df.drop(arr_keys["pose"], axis='columns', inplace=True)
    if arr[1]==0:
        df.drop(arr_keys["gaze"], axis='columns', inplace=True)
    if arr[2]==0:
        df.drop(arr_keys["control"], axis='columns', inplace=True)
    if arr[3]==0:
        df.drop(arr_keys["scene"], axis='columns', inplace=True)
    if arr[4]==0:
        df.drop(arr_keys["video"], axis='columns', inplace=True)
    if arr[5]==0:
        df.drop(arr_keys["MS_rating"], axis='columns', inplace=True)
    return df

# Get logger
logger = setup_logger()

df_ml=pd.read_csv("main_log_4.csv")

for i in range(len(df_ml)):
    logger.info('------------------------------------------------------------------------------------------------------------------')
    
    row=df_ml[i:i+1]
    file=row["name"].to_list()[0]
    try:
        logger.info(f'***************{file}********************')
        game_name="_".join(file.split("_")[2:])
        path1=os.path.join(output_dir, game_name,file,"data_file.csv")
        path2=os.path.join(input_dir, file, "data_file.csv")
        path3=os.path.join(input_dir, file,"video")
        path4=os.path.join(output_dir, game_name,file,"video")
        os.makedirs(path4)
        print(path4)
        cols_to_remove=[row["pose"].to_list()[0],row["gaze"].to_list()[0],row["control"].to_list()[0],row["scene_obj"].to_list()[0],row["video"].to_list()[0],row["MS_rating"].to_list()[0]]
        print(cols_to_remove)
        df_f=pd.read_csv(path2)
        df_f.drop(df_f.columns[df_f.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        df_f=remove_columns(cols_to_remove,df_f)
        df_f.to_csv(path1, index=False)
        shutil.copytree(path3, path4,dirs_exist_ok = True)
        logger.info('Completed')
        
    except:
        logger.info(f'Error! Please check {file}')
        continue