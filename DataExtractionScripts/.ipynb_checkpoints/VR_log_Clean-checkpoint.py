import csv
import json
import os
import shutil
from datetime import datetime
# import time

import logging

# Define logger
from logger_config import setup_logger

# Get logger
logger = setup_logger()

def remove_nonints(row):
    key= row['Key']
    # print(key)
    try:
        key=int(key)
        if key>0 and key<=5:
            return True
        else:
            logger.info(f'invalid MS rating: not between 1-5 : {key}')
            return False
    except:
        logger.info(f'invalid MS rating: not a number : {key}')
        return False

def find_nearest_frame(json_data, target_timestamp):
    nearest_frame = None
    min_time_difference = float('inf')

    for frame_number, data in json_data.items():
        timestamp = data.get("timestamp")
        if timestamp is not None:
            time_difference = abs(target_timestamp - timestamp)
            if time_difference < min_time_difference:
                min_time_difference = time_difference
                nearest_frame = frame_number

    return nearest_frame
        

def convert_timestamp(timestamp_str):
    timestamp_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    timestamp_ms = int(timestamp_dt.timestamp() * 1000)
    return timestamp_ms
    
def extract_VRlog(archive, item, dataset_name, exp_res, output_dir):
    target_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    output_csv_path = os.path.join(target_dir, "VRMS_log.csv")
    archive.extract(item, 'temp')
    csv_file_path = os.path.join('temp', item)
    with open(csv_file_path, 'r') as input_csv:
        csv_reader = csv.DictReader(input_csv)
        with open(output_csv_path, 'w', newline='') as output_csv:
            fieldnames = csv_reader.fieldnames  + ['frame']
            csv_writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
            csv_writer.writeheader()
            for row in csv_reader:
                if remove_nonints(row):
                    # print("came here")
                    # Assuming the timestamp column is named 'timestamp'
                    timestamp_str = row['Timestamp']
                    timestamp_ms = convert_timestamp(timestamp_str)
                    # print(timestamp_str, timestamp_ms)
                    row['Timestamp'] = timestamp_ms
                    row['frame']=find_nearest_frame(exp_res,timestamp_ms)
                    row['Key']=int(row['Key'])
                    csv_writer.writerow(row)
                else:
                    key_val=row['Key']
                    # print("kv = ", key_val,remove_nonints(row))
    shutil.rmtree('temp')
    logger.info("Motion sickness rating extracted")