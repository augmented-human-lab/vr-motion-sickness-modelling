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

dataset_dir_path = "/data/VR_NET/zipped/27"
output_dir = "/data/VR_NET/data/test1"

arr_k=['frame', 'timestamp', 'head_dir', 'head_pos', 'head_vel', 'head_angvel', 'left_eye_dir', 'left_eye_pos', 'left_eye_vel', 'left_eye_angvel', 'right_eye_dir', 'right_eye_pos', 'right_eye_vel', 'right_eye_angvel', 'left_eye', 'right_eye', 'confidence', 'is_valid', 'ConnectedControllerTypes', 'Buttons', 'Touches', 'NearTouches', 'IndexTrigger', 'HandTrigger', 'Thumbstick', 'object_name', 'bounds', 'm_matrix', 'camera_name', 'p_matrix', 'v_matrix', 'video', 'MS_rating']

arr_keys=[['timestamp',
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
 ['left_eye', 'right_eye', 'confidence', 'is_valid'],
 ['ConnectedControllerTypes',
  'Buttons',
  'Touches',
  'NearTouches',
  'IndexTrigger',
  'HandTrigger',
  'Thumbstick'],
 ['object_name', 'bounds', 'm_matrix'],
 ['camera_name', 'p_matrix', 'v_matrix']]

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
    frames=get_frames(pose)
    files=[pose, gaze, control, obj_res, camera_res]
    video_list=[int(x[:-4]) for x in os.listdir(os.path.join(path,"video"))]
    csv_path=os.path.join(path,"data_file.csv")

    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(arr_k)
        # print(arr_k)
        for i in frames:
            arr=[i]
            for j in range(len(files)):
                file=files[j]
                keys_f=arr_keys[j]
                # print(keys_f)
                for p in keys_f:
                    # print(p)
                    if i in file:
                        arr.append(file[i][p])
                    else:
                        arr.append(None)
            if i in video_list:
                arr.append(os.path.join("video",str(i)))
            else:
                arr.append(None)
            df=pd.read_csv(os.path.join(path,"VRMS_log.csv"))
            arr.append(get_key_for_frame(df, i))
            csv_writer.writerow(arr)

def worker(dataset):
    start_time = time.time()
    dataset_name = dataset[:-4]
    dataset_abs_path = os.path.join(dataset_dir_path, dataset)
    print(dataset_abs_path[:-4])
    path_out=os.path.join(output_dir,dataset_name)
    with zipfile.ZipFile(dataset_abs_path, mode='r') as archive:
        for item in archive.namelist():
            if "_pose/data" in item:
                exp_res, pose=extract_pose(archive, item, dataset_name)
            if "_video/data" in item:
                extract_video(archive, item, dataset_name, output_dir)
            if "_gaze/data" in item:
                gaze=extract_gaze(archive, item, dataset_name)
            if "_scene/data" in item:
                obj_res, camera_res=extract_scene(archive, item, dataset_name)
            if "_control/data" in item:
                control=extract_control(archive, item, dataset_name)
            if "VRLOG" in item:
                extract_VRlog(archive, item, dataset_name,exp_res, output_dir)
        # print(pose)
        write_csv(pose,gaze,control,obj_res,camera_res,path_out,arr_k)
        print("--- %s seconds ---" % (time.time() - start_time))
            # shutil.rmtree('dataset_abs_path')
                           
def main():
    tasks = []
    print()
    
    datasets = os.listdir(dataset_dir_path)
    for dataset in datasets:
        if dataset.endswith(".zip"):
            tasks.append(dataset)

    pool = multiprocessing.Pool(1)
    count = 0
    for res in pool.imap(worker, tasks):
        count += 1

if __name__ == "__main__":
    main()