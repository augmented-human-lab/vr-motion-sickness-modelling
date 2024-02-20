# This file is the main file that can be used to extract data. 
# This takes File_extraction, logger_config, VR_log_Clean python files to completely run the program

import io
import multiprocessing
import os
import pickle
import struct
import zipfile
import csv
import shutil
import gzip
import pandas as pd
import time

from PIL import Image
from datetime import datetime
from VR_log_Clean import *
from File_Extraction import *

import logging

# Define logger
from logger_config import setup_logger

global m_log_arr

# Get logger
logger = setup_logger()

dataset_dir_path = "/data/VR_NET/zipped/"
output_dir = "/data/VR_NET/data/"

arr_k=['frame', 'head_dir', 'head_pos', 'head_vel', 'head_angvel', 'left_eye_dir', 'left_eye_pos', 'left_eye_vel', 'left_eye_angvel', 'right_eye_dir', 'right_eye_pos', 'right_eye_vel', 'right_eye_angvel',  'timestamp', 'left_eye', 'right_eye', 'confidence', 'is_valid', 'ConnectedControllerTypes', 'Buttons', 'Touches', 'NearTouches', 'IndexTrigger', 'HandTrigger', 'Thumbstick', 'object_name', 'bounds', 'm_matrix', 'camera_name', 'p_matrix', 'v_matrix', 'video', 'MS_rating']

arr_keys=[[
  'head_dir',
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
 ['timestamp','left_eye', 'right_eye', 'confidence', 'is_valid'],
 ['ConnectedControllerTypes',
  'Buttons',
  'Touches',
  'NearTouches',
  'IndexTrigger',
  'HandTrigger',
  'Thumbstick'],
 ['object_name', 'bounds', 'm_matrix'],
 ['camera_name', 'p_matrix', 'v_matrix']]

main_log_keys=["name","path","pose","gaze","control","scene_obj","scene_cam","video","MS_rating","face","manual"]



def check_face(list_n):
    ans=False
    for i in list_n:
        if "_face/data" in i:
            ans=True
    return ans

def check_count(archive):
    name_list=archive.namelist()
    count = len(name_list)
    logger.info(f'There are {count} files here')
    if check_face(name_list):
        valid_count=7
    else: 
        valid_count=6
    if count <= valid_count:
        # print('yes')
        return True
    else:
        logger.info(f'Following files are available: {name_list}')
        return False
    
def check_availability(gaze,control,obj_res,camera_res):
    global m_log_arr
    if gaze=={}:
        m_log_arr[3]=0
        logger.info("Gaze data is not available")
    if control=={}:
        m_log_arr[4]=0
        logger.info("Control data is not available")
    if obj_res=={}:
        m_log_arr[5]=0
        logger.info("Scene: obj data is not available")
    if camera_res=={}:
        m_log_arr[6]=0
        logger.info("Scene: camera data is not available")

    
def get_frames(json):
    frames=list(json.keys())
    return frames

def get_key_for_frame(df, target_frame):
    row = df[df['frame'] == target_frame]
    if not row.empty:
        return row['Key'].values[0]
    else:
        return None

def write_csv(pose,gaze,control,obj_res,camera_res,path,arr_k):
    global m_log_arr
    frames=get_frames(gaze)
    files=[pose, gaze, control, obj_res, camera_res]
    check_availability(gaze,control,obj_res,camera_res)
    video_list=[]
    if os.path.exists(os.path.join(path,"video")):
        video_list=[int(x[:-4]) for x in os.listdir(os.path.join(path,"video"))]
    if video_list==[]:
        logger.info("Video data is not available")
        m_log_arr[7]=0
    csv_path=os.path.join(path,"data_file.csv")
    print(csv_path)

    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(arr_k)
        # print(arr_k)
        for i in frames:
            arr=[i]
            for j in range(len(files)):
                file=files[j]
                keys_f=arr_keys[j]
                # print(j, "This is j")
                for p in keys_f:
                    # print(i,p)
                    if i in file:
                        if p in file[i]:
                            arr.append(file[i][p])
                        else:
                            arr.append(None)
                    else:
                        arr.append(None)
            if i in video_list:
                arr.append(os.path.join("video",str(i)))
            else:
                arr.append(None)
            csv_path=os.path.join(path,"VRMS_log.csv")
            
            if os.path.exists(csv_path):
                df=pd.read_csv(os.path.join(path,"VRMS_log.csv"))
                arr.append(get_key_for_frame(df, i))
                
            else:
                logging.info("Motion Sickness Ratings are not available")
                m_log_arr[8]=0
            csv_writer.writerow(arr)

def worker(dataset):
    global m_log_arr
    start_time = time.time()
    dataset_name = dataset[:-4]
    dataset_abs_path = os.path.join(dataset_dir_path, dataset)
    dataset_n=dataset_name.split("/")[-1]
    print(dataset_abs_path[:-4])
    path_out=os.path.join(output_dir,dataset_n)
    with open("main_log_3.csv", 'a') as main_log_file:
        csv_writer_M = csv.writer(main_log_file)
        with zipfile.ZipFile(dataset_abs_path, mode='r') as archive:
            m_log_arr=["","",0,1,1,1,1,1,1,0,0]
            # print(dataset, dataset_name,dataset_abs_path)
            m_log_arr[0]=dataset_n
            m_log_arr[1]=dataset_abs_path
            logger.info('------------------------------------------------------------------------------------------------------------------')
            logger.info(f'***************{dataset_n}********************')
            exp_res, pose, gaze, obj_res, camera_res, control= {},{},{},{},{},{}
            # if check_count(archive):
            logger.info("Extraction started...")
            for item in archive.namelist():
                if "_pose/data" in item:
                    exp_res, pose=extract_pose(archive, item, dataset_name)
                if "_video/data" in item:
                    extract_video(archive, item, dataset_n, output_dir)
                if "_gaze/data" in item:
                    gaze=extract_gaze(archive, item, dataset_name)
                if "_scene/data" in item:
                    obj_res, camera_res=extract_scene(archive, item, dataset_name)
                if "_control/data" in item:
                    control=extract_control(archive, item, dataset_name)
                if "VRLOG" in item:
                    extract_VRlog(archive, item, dataset_n,exp_res, output_dir)
                if "_face/data" in item:
                    m_log_arr[9]=1
                    logger.info("Face data present!- extract in the next phase")

            # print(pose)
                # if pose!={}:
            write_csv(pose,gaze,control,obj_res,camera_res,path_out,arr_k)
                # else:
                #     m_log_arr[2]=0
                #     logger.info("Can't proceed: no 'pose' data")
                # print("--- %s seconds ---" % (time.time() - start_time))
                    # shutil.rmtree('dataset_abs_path')
            # else:
            #     m_log_arr[10]=1
            #     logger.info('Can\'t extract, many files: Please proceed manually')
        csv_writer_M.writerow(m_log_arr)
        
        target_dir = os.path.join(output_dir, dataset_n)
        output_csv_path = os.path.join(target_dir, "VRMS_log.csv")
        print(output_csv_path)
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)
            
def main():
    tasks = []  
    # log_arr=[]
    with open("main_log_3.csv", 'a', newline='') as main_log_file:
        csv_writer_M = csv.writer(main_log_file)
        csv_writer_M.writerow(main_log_keys)
    
    logger.info('----------------------------------------------------Starting extraction again: files_without pose_correction--------------------------------------------------------------')
        
        
    # folders=os.listdir(dataset_dir_path)
    # # folders=folders[15:] #Don't use this unless you are running from a middle point
    # for folder in folders:
    #     dataset_dir_path1=os.path.join(dataset_dir_path, folder)
    #     datasets = os.listdir(dataset_dir_path1)
    #     # print(datasets)
    #     for i in range(len(datasets)):
    #         # print(datasets[i])
    #         if datasets[i].endswith(".zip"):
    #             tasks.append(os.path.join(dataset_dir_path1,datasets[i]))
    tasks= ['/data/VR_NET/zipped/22/3_2_Wild_Quest.zip',
 '/data/VR_NET/zipped/2/39_2_Citadel.zip',
 '/data/VR_NET/zipped/0/20_1_Mars_Miners.zip',
 '/data/VR_NET/zipped/3/61_2_UFO_crash_site_venue.zip',
 '/data/VR_NET/zipped/2/42_1_Super_Rumble.zip',
 '/data/VR_NET/zipped/18/304_2_Kawaii_Urgent_Care.zip',
 '/data/VR_NET/zipped/4/81_2_Venues.zip',
 '/data/VR_NET/zipped/1/29_1_Army_Men.zip',
 '/data/VR_NET/zipped/1/33_1_Citadel.zip']
    # tasks=tasks[11:] #Don't use this unless you are running from a middle point
    pool = multiprocessing.Pool(1)
    count = 0
    for res in pool.imap(worker, tasks):
        count += 1
        print(count)


if __name__ == "__main__":
    main()