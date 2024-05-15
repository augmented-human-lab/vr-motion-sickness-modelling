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

from PIL import Image
from datetime import datetime

from logger_config import setup_logger

global m_log_arr

# Get logger
logger = setup_logger()

dataset_dir_path="/data/VR_NET/data"

def extract_pose(archive, item, dataset_name):
    pose_data_pkg = archive.read(item)
    bio = io.BytesIO(pose_data_pkg)
    res = {}
    while True:
        data = bio.read(112)
        if not data or len(data) != 112:
            break

        metadata = data[-24:]
        unused, node, frame, timestamp = struct.unpack("iiLL", metadata)

        if node == 9:
            node_str = "head"
        elif node == 1:
            node_str = "left_eye"
        elif node == 2:
            node_str = "right_eye"
        elif node == 3:
            node_str = "left_controller"
        elif node == 4:
            node_str = "right_controller"
        else:
            continue

        if frame not in res:
            res[frame] = {}

        x, y, z, w, p1, p2, p3, v1, v2, v3, a1, a2, a3, angv1, angv2, angv3, anga1, anga2, anga3 = struct.unpack(
            "fffffffffffffffffff", data[0:76])
        
        res[frame]["timestamp"] = timestamp
        res[frame][node_str + "_dir"] = (x, y, z, w)
        res[frame][node_str + "_pos"] = (p1, p2, p3)
        res[frame][node_str + "_vel"] = (v1, v2, v3)
        res[frame][node_str + "_angvel"] = (angv1, angv2, angv3)

    # target_dir = os.path.join(output_dir, dataset_name)
    # os.makedirs(target_dir, exist_ok=True)
    # target_file = "pose.pickle"
    # with open(target_file, "wb") as outfp:
    #     pickle.dump(res, outfp)
    return res
 

def update_csv(path,res):
    df=pd.read_csv(path)
    frames=df['frame']
    lcont_dir=[]
    lcont_pos=[]
    lcont_vel=[]
    lcont_angvel=[]
    rcont_dir=[]
    rcont_pos=[]
    rcont_vel=[]
    rcont_angvel=[]
    array1=["left_controller_dir",'left_controller_pos','left_controller_vel','left_controller_angvel','right_controller_dir','right_controller_pos','right_controller_vel','right_controller_angvel']
    for i in frames:
        if i in res:
            if 'left_controller_dir' in res[i].keys():
                lcont_dir.append(res[i]['left_controller_dir'])
            else:
                lcont_dir.append(None)
            if 'left_controller_pos' in res[i].keys():
                lcont_pos.append(res[i]['left_controller_pos'])
            else:
                lcont_pos.append(None)
            if 'left_controller_vel' in res[i].keys():
                lcont_vel.append(res[i]['left_controller_vel'])
            else:
                lcont_vel.append(None)
            if 'left_controller_angvel' in res[i].keys():
                lcont_angvel.append(res[i]['left_controller_angvel'])
            else:
                lcont_angvel.append(None)
            if 'right_controller_dir' in res[i].keys():
                rcont_dir.append(res[i]['right_controller_dir'])
            else:
                rcont_dir.append(None)
            if 'right_controller_pos' in res[i].keys():
                rcont_pos.append(res[i]['right_controller_pos'])
            else:
                rcont_pos.append(None)
            if 'right_controller_vel' in res[i].keys():
                rcont_vel.append(res[i]['right_controller_vel'])
            else:
                rcont_vel.append(None)
            if 'right_controller_angvel' in res[i].keys():
                rcont_angvel.append(res[i]['right_controller_angvel'])
            else:
                rcont_angvel.append(None)
            # weights.append(res[i]["left_controller_dir"])
            # confidence.append(res[i]["confidence"])
            # validity.append(res[i]["validity"])
        else:
            # logger.info(i)
            lcont_dir.append(None)
            lcont_pos.append(None)
            lcont_vel.append(None)
            lcont_angvel.append(None)
            rcont_dir.append(None)
            rcont_pos.append(None)
            rcont_vel.append(None)
            rcont_angvel.append(None)
            # weights.append(None)
            # confidence.append(None)
            # validity.append(None)
            
    df["left_controller_dir"]=lcont_dir
    df["left_controller_pos"]=lcont_pos
    df["left_controller_vel"]=lcont_vel
    df["left_controller_angvel"]=lcont_angvel
    df["right_controller_dir"]=rcont_dir
    df["right_controller_pos"]=rcont_pos
    df["right_controller_vel"]=rcont_vel
    df["right_controller_angvel"]=rcont_angvel

    # df=df[df.columns[1:]]
    df.to_csv(path,index=False)
    



def worker(dataset):
    dataset_name = dataset.split("/")[-1][:-4]
    print(dataset_name)
    with zipfile.ZipFile(dataset, mode='r') as archive:
        for item in archive.namelist():
            if "_pose/data" in item:
                res=extract_pose(archive, item, dataset_name)
                print("res_extracted")
    path1=os.path.join(dataset_dir_path,dataset_name,"data_file.csv")
    print(dataset_name)
    fp2='_'.join(dataset_name.split('_')[2:])
    fp=os.path.join("/data/VR_NET/folders", fp2, dataset_name,'data_file.csv')
    logger.info('****************************************')
    logger.info(fp)
    # try:
    update_csv(fp, res)    
    logger.info('controller extracted')
    # except:
    #     logger.info('controller extraction failed')
    
def main():
    df=pd.read_csv("/home/dinithi/vr-motion-sickness-modelling/DataExtractionScripts/log_csvs/main_log_wpath.csv")
    # df1=df[df["face"]==1]
    tasks=df["path"].tolist()
    pool = multiprocessing.Pool(1)
    count = 0
    for res in pool.imap(worker, tasks[6:]):
        count += 1

if __name__ == "__main__":
    main()