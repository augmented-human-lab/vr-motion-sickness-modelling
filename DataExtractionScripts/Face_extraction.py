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

dataset_dir_path="/data/VR_NET/data"



def extract_face(archive, item, dataset_name):
    gaze_data_pkg = archive.read(item)
    bio = io.BytesIO(gaze_data_pkg)

    res = {}
    while True:
        data = bio.read(304)
        if not data or len(data) != 304:
            break

        exp_list=[]
        for exp in range(63):
            expression_w=data[exp*4:(exp+1)*4]
            expression_w=struct.unpack("f", expression_w)
            exp_list.append(expression_w[0])

        confidence=data[252:260]
        
        status = data[260:268]
        
        conf1, conf2 = struct.unpack("ff", confidence)
        
        IsValid, EyeFollowValid = struct.unpack("ii", status)

        metadata = data[-24:]
        unused, frame, timestamp = struct.unpack("iLL", metadata)

        if frame not in res:
            res[frame] = {}

        res[frame]["weights"] =exp_list
        res[frame]["confidence"]=(conf1, conf2)
        res[frame]["validity"]=(IsValid, EyeFollowValid)
        
    # logger.info("face_extracted")
    
    return res     

def update_csv(path,res):
    df=pd.read_csv(path)
    frames=df['frame']
    weights=[]
    confidence=[]
    validity=[]
    for i in frames:
        if i in res:
            weights.append(res[i]["weights"])
            confidence.append(res[i]["confidence"])
            validity.append(res[i]["validity"])
        else:
            weights.append(None)
            confidence.append(None)
            validity.append(None)
            
    df["exp_weights"]=weights
    df["exp_confidence"]=confidence
    df["exp_validity"]=validity
    # df=df[df.columns[1:]]
    df.to_csv(path,index=False)   



def worker(dataset):
    dataset_name = dataset.split("/")[-1][:-4]
    print(dataset_name)
    with zipfile.ZipFile(dataset, mode='r') as archive:
        for item in archive.namelist():
            if "_face/data" in item:
                res=extract_face(archive, item, dataset_name)

    path1=os.path.join(dataset_dir_path,dataset_name,"data_file.csv")
    print(path1)
    update_csv(path1, res)    
    
def main():
    df=pd.read_csv("/home/dinithi/vr-motion-sickness-modelling/DataExtractionScripts/main_log.csv")
    df1=df[df["face"]==1]
    tasks=df1["path"].tolist()
    pool = multiprocessing.Pool(1)
    count = 0
    for res in pool.imap(worker, tasks):
        count += 1

if __name__ == "__main__":
    main()