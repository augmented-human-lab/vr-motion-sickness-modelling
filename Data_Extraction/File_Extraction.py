import io
import multiprocessing
import os
import pickle
import struct
import zipfile
import csv
import shutil
import gzip

from PIL import Image
from datetime import datetime
# from VR_log_Clean import *
import logging

# Define logger
from logger_config import setup_logger

# Get logger
logger = setup_logger()

def extract_gaze(archive, item, dataset_name):
    # exp_res={}
    gaze_data_pkg = archive.read(item)

    bio = io.BytesIO(gaze_data_pkg)

    res = {}
    while True:
        data = bio.read(104)
        if not data or len(data) != 104:
            break

        left_eye = data[:36]
        right_eye = data[36:72]

        l_x, l_y, l_z, l_w, l_p1, l_p2, l_p3 = struct.unpack(
            "fffffff", left_eye[:28])

        r_x, r_y, r_z, r_w, r_p1, r_p2, r_p3 = struct.unpack(
            "fffffff", right_eye[:28])
        
        l_c,l_i = struct.unpack("fi", left_eye[28:36])
        r_c,r_i = struct.unpack("fi", right_eye[28:36])

        metadata = data[-24:]
        unused, frame, timestamp = struct.unpack("iLL", metadata)

        if frame not in res:
            res[frame] = {}

        # res[frame] = (timestamp, l_x, l_y, l_z, l_w, l_p1, l_p2, l_p3, r_x, r_y, r_z, r_w, r_p1, r_p2, r_p3)
        # exp_res[frame]["timestamp"] = timestamp
        res[frame]["timestamp"]=timestamp
        res[frame]["left_eye"] =(l_x, l_y, l_z, l_w, l_p1, l_p2, l_p3)
        res[frame]["right_eye"]=(r_x, r_y, r_z, r_w, r_p1, r_p2, r_p3)
        res[frame]["confidence"] =(l_c, r_c)
        res[frame]["is_valid"]=(l_i, r_i) 
    logger.info("gaze_extracted")
    
    return res
        


def extract_pose(archive, item, dataset_name):
    pose_data_pkg = archive.read(item)
    bio = io.BytesIO(pose_data_pkg)
    res = {}
    exp_res={}
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
            exp_res[frame] = {}
        x, y, z, w, p1, p2, p3, v1, v2, v3, a1, a2, a3, angv1, angv2, angv3, anga1, anga2, anga3 = struct.unpack(
            "fffffffffffffffffff", data[0:76])
        
        res[frame]["timestamp"] = timestamp
        exp_res[frame]["timestamp"] = timestamp
        res[frame][node_str + "_dir"] = (x, y, z, w)
        res[frame][node_str + "_pos"] = (p1, p2, p3)
        res[frame][node_str + "_vel"] = (v1, v2, v3)
        res[frame][node_str + "_angvel"] = (angv1, angv2, angv3)

    logger.info("pose extracted")
    return exp_res, res

def extract_video(archive, item, dataset_name,output_dir):
    video_data_pkg = archive.read(item)
    bio = io.BytesIO(video_data_pkg)

    while True:
        data = bio.read(24)
        if not data or len(data) != 24:
            break
        frame, timestamp, size = struct.unpack("LLL", data)
        jpg = bio.read(size)

        if not jpg or len(jpg) != size:
            break

        original_jpg_image = Image.open(io.BytesIO(jpg))
        flipped = original_jpg_image.transpose(Image.FLIP_LEFT_RIGHT)
        rotated = flipped.rotate(180)
        target_dir = os.path.join(output_dir, dataset_name, "video")
        # print(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(target_dir, "%d.jpg" % (frame))
        rotated.save(target_file)
    logger.info("video extracted")
        
def extract_scene(archive, item, dataset_name):
    scene_data_pkg = archive.read(item)
    fp1 = io.BytesIO(scene_data_pkg)
    camera_res = {}
    obj_res = {}
    with gzip.open(fp1, "rb") as fp:
        while (True):
            try:
                data = fp.read(24)
                type1, frameIndex, timestamp = struct.unpack("QQQ", data)
                utfname = fp.read(128)
                name = utfname.decode("utf-16", errors="ignore")

                if type1 == 0xF1:
                    camera = fp.read(128)
                    p_matrix = struct.unpack("ffffffffffffffff", camera[:64])
                    v_matrix = struct.unpack("ffffffffffffffff", camera[64:])
                    if frameIndex not in camera_res:
                        camera_res[frameIndex] = {}
                        # camera_res[frameIndex]["timestamp"]= timestamp
                        camera_res[frameIndex]["camera_name"]=[name.rstrip('\x00')]
                        camera_res[frameIndex]["p_matrix"]= [p_matrix]
                        camera_res[frameIndex]["v_matrix"]= [v_matrix]

                    else:
                        camera_res[frameIndex]["camera_name"].append(name.rstrip('\x00'))
                        camera_res[frameIndex]["p_matrix"].append(p_matrix)
                        camera_res[frameIndex]["v_matrix"].append(v_matrix)
                    # if name.startswith("CenterEyeAnchor"):
                    #     camera_res[frameIndex] = v_matrix

                else:
                    renderer = fp.read(88)
                    bounds = struct.unpack("ffffff", renderer[:24])
                    m_matrix = struct.unpack("ffffffffffffffff", renderer[24:])
                    if frameIndex not in obj_res:
                        obj_res[frameIndex] = {}
                        # obj_res[frameIndex]["timestamp"]= timestamp
                        obj_res[frameIndex]["object_name"] = [name.rstrip('\x00')]
                        obj_res[frameIndex]["bounds"] = [bounds]
                        obj_res[frameIndex]["m_matrix"] = [m_matrix]
                    else:
                        obj_res[frameIndex]["object_name"].append(name.rstrip('\x00'))
                        obj_res[frameIndex]["bounds"].append(bounds)
                        obj_res[frameIndex]["m_matrix"].append(m_matrix)
            except:
                # logger.error('This is an error message')
                break
    logger.info("scene extracted")
    return obj_res,camera_res


def extract_control(archive, item, dataset_name):
    pose_data_pkg = archive.read(item)
    bio = io.BytesIO(pose_data_pkg)
    control = {}
    while True:
        data = bio.read(120)
        if not data or len(data) != 120:
            break

        metadata = data[-24:]
        unused, controllerMask, frame, timestamp = struct.unpack("iiLL", metadata)

        ConnectedControllerTypes, Buttons, Touches, NearTouches, IndexTrigger_1, IndexTrigger_2, HandTrigger_1, HandTrigger_2, Thumbstick_1_x, Thumbstick_1_y, Thumbstick_2_x, Thumbstick_2_y, Touchpad_1_x, Touchpad_1_y, Touchpad_2_x, Touchpad_2_y = struct.unpack("IIIIffffffffffff", data[0:64])
        
        # BatteryPercentRemaining_1, BatteryPercentRemaining_2 , RecenterCount_1, RecenterCount_2= struct.unpack(
        #     "BBBB", data[64:68])
   
        if frame not in control:
            control[frame] = {}
            
        # control[frame]["timestamp"] = timestamp
        control[frame]["ConnectedControllerTypes"] =ConnectedControllerTypes
        control[frame]["Buttons"] =Buttons
        control[frame]["Touches"] =  Touches
        control[frame]["NearTouches"] = NearTouches
        control[frame]["IndexTrigger"] = (IndexTrigger_1, IndexTrigger_2)
        control[frame]["HandTrigger"] = (HandTrigger_1, HandTrigger_2)
        control[frame]["Thumbstick"] = (Thumbstick_1_x, Thumbstick_1_y, Thumbstick_2_x, Thumbstick_2_y)
        # control[frame]["Touchpad"] = (Touchpad_1_x, Touchpad_1_y, Touchpad_2_x, Touchpad_2_y)
        # control[frame]["BatteryPercentRemaining"]=(BatteryPercentRemaining_1, BatteryPercentRemaining_2)
        # control[frame]["RecenterCount"]=(RecenterCount_1, RecenterCount_2)
    logger.info("control extracted")
    return control


        
