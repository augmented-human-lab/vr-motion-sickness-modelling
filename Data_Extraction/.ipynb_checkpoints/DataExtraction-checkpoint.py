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

dataset_dir_path = "/data/VR_NET/zipped"
output_dir = "/data/VR_NET/data/test1"

def convert_timestamp(timestamp_str):
    timestamp_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    timestamp_ms = int(timestamp_dt.timestamp() * 1000)
    return timestamp_ms

def extract_gaze(archive, item, dataset_name):
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
        
        res[frame]["timestamp"]=timestamp
        res[frame]["left_eye"] =(l_x, l_y, l_z, l_w, l_p1, l_p2, l_p3)
        res[frame]["right_eye"]=(r_x, r_y, r_z, r_w, r_p1, r_p2, r_p3)
        res[frame]["confidence"] =(l_c, r_c)
        res[frame]["is_valid"]=(l_i, r_i) 
        
    target_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "gaze.pickle")
    with open(target_file, "wb") as outfp:
        pickle.dump(res, outfp)


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

    target_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "pose.pickle")
    with open(target_file, "wb") as outfp:
        pickle.dump(res, outfp)


def extract_video(archive, item, dataset_name):
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
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(target_dir, "%d_%d.jpg" % (frame, timestamp))
        rotated.save(target_file)

        
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
                        camera_res[frameIndex]["timestamp"]= timestamp
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
                        obj_res[frameIndex]["timestamp"]= timestamp
                        obj_res[frameIndex]["object_name"] = [name.rstrip('\x00')]
                        obj_res[frameIndex]["bounds"] = [bounds]
                        obj_res[frameIndex]["m_matrix"] = [m_matrix]
                    else:
                        obj_res[frameIndex]["object_name"].append(name.rstrip('\x00'))
                        obj_res[frameIndex]["bounds"].append(bounds)
                        obj_res[frameIndex]["m_matrix"].append(m_matrix)
            except:
                break
            
        target_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(target_dir, exist_ok=True)
        target_file_1 = os.path.join(target_dir, "camera_res.pickle")
        target_file_2 = os.path.join(target_dir, "obj_res.pickle")
        with open(target_file_1, "wb") as outfp:
            pickle.dump(camera_res, outfp)
        with open(target_file_2, "wb") as outfp:
            pickle.dump(obj_res, outfp)


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
            
        control[frame]["timestamp"] = timestamp
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

    target_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "control.pickle")
    with open(target_file, "wb") as outfp:
        pickle.dump(control, outfp)
        
def extract_face(archive, item, dataset_name):
    gaze_data_pkg = archive.read(item)
    print("came here")
    bio = io.BytesIO(gaze_data_pkg)

    res = {}
    while True:
        data = bio.read(304)
        if not data or len(data) != 304:
            break

        # weights = data[:252]
        # status = data[252:16]
        exp_list=[]
        for exp in range(63):
            expression_w=data[exp*4:(exp+1)*4]
            expression_w=struct.unpack("f", expression_w)
            exp_list.append(expression_w[0])
        # print(exp_list)
        
        confidence=data[252:260]
        
        status = data[260:268]
        
        conf1, conf2 = struct.unpack("ff", confidence)
        
        IsValid, EyeFollowValid = struct.unpack("ii", status)
        
        # l_c,l_i = struct.unpack("fi", left_eye[28:36])
        # r_c,r_i = struct.unpack("fi", right_eye[28:36])

        metadata = data[-24:]
        unused, frame, timestamp = struct.unpack("iLL", metadata)

        if frame not in res:
            res[frame] = {}

        # res[frame] = (timestamp, l_x, l_y, l_z, l_w, l_p1, l_p2, l_p3, r_x, r_y, r_z, r_w, r_p1, r_p2, r_p3)
        
        # res[frame]["timestamp"]=timestamp
        res[frame]["weights"] =exp_list
        res[frame]["confidence"]=(conf1, conf2)
        res[frame]["validity"]=(IsValid, EyeFollowValid)
    # logger.info("face_extracted")
    
    return res        
        
def extract_VRlog(archive, item, dataset_name):
    target_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    output_csv_path = os.path.join(target_dir, "VRMS_log.csv")
    archive.extract(item, 'temp')
    csv_file_path = os.path.join('temp', item)
    with open(csv_file_path, 'r') as input_csv:
        csv_reader = csv.DictReader(input_csv)
        with open(output_csv_path, 'w', newline='') as output_csv:
            fieldnames = csv_reader.fieldnames
            csv_writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
            csv_writer.writeheader()

            for row in csv_reader:
                # Assuming the timestamp column is named 'timestamp'
                timestamp_str = row['Timestamp']
                timestamp_ms = convert_timestamp(timestamp_str)
                # print(timestamp_str, timestamp_ms)
                row['Timestamp'] = timestamp_ms

                csv_writer.writerow(row)
    shutil.rmtree('temp')
    
        
def worker(dataset):
    dataset_name = dataset.split("/")[-1][:-4]
    print(dataset_name)
    dataset_abs_path = os.path.join(dataset_dir_path, dataset)
    print(dataset_abs_path)
    with zipfile.ZipFile(dataset_abs_path, mode='r') as archive:
        for item in archive.namelist():
            if "_pose/data" in item:
                extract_pose(archive, item, dataset_name)
            if "_video/data" in item:
                extract_video(archive, item, dataset_name)
            if "_gaze/data" in item:
                extract_gaze(archive, item, dataset_name)
            if "_scene/data" in item:
                extract_scene(archive, item, dataset_name)
            if "_control/data" in item:
                extract_control(archive, item, dataset_name)
            if "_face/data" in item:
                res=extract_face(archive, item, dataset_name)
            if "VRLOG" in item:
                extract_VRlog(archive, item, dataset_name)
                # print(res[48835])


def main():
    tasks = []
    dataset2=[]
    folders=os.listdir(dataset_dir_path)
    print(folders)
    for folder in folders:
        dataset_dir_path1=os.path.join(dataset_dir_path, folder)
        datasets = os.listdir(dataset_dir_path1)
        # print(datasets)
        if datasets[0].endswith(".zip"):
            tasks.append(os.path.join(dataset_dir_path1,datasets[0]))
    tasks=tasks[0:2]
    pool = multiprocessing.Pool(1)
    count = 0
    for res in pool.imap(worker, tasks):
        count += 1


if __name__ == "__main__":
    main()