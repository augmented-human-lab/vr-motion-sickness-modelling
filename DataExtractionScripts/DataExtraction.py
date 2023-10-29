import io
import multiprocessing
import os
import pickle
import struct
import zipfile

from PIL import Image

dataset_dir_path = "raw_datasets"
output_dir = "datasets"


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

        metadata = data[-24:]
        unused, frame, timestamp = struct.unpack("iLL", metadata)

        if frame not in res:
            res[frame] = {}

        res[frame] = (l_x, l_y, l_z, l_w, l_p1, l_p2, l_p3, r_x, r_y, r_z, r_w, r_p1, r_p2, r_p3)

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

        res[frame][node_str + "_dir"] = (x, y, z, w)
        res[frame][node_str + "_pos"] = (p1, p2, p3)
        res[frame][node_str + "_vel"] = (v1, v2, v3)
        res[frame][node_str + "_angvel"] = (angv1, angv2, angv3)

    target_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, "control.pickle")
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
        target_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(target_dir, "%d.jpg" % frame)
        rotated.save(target_file)


def worker(dataset):
    dataset_name = dataset[:-4]
    dataset_abs_path = os.path.join(dataset_dir_path, dataset)
    with zipfile.ZipFile(dataset_abs_path, mode='r') as archive:
        for item in archive.namelist():
            if "_pose/data" in item:
                extract_pose(archive, item, dataset_name)
            if "_video/data" in item:
                extract_video(archive, item, dataset_name)
            if "_gaze/data" in item:
                extract_gaze(archive, item, dataset_name)


def main():
    tasks = []
    datasets = os.listdir(dataset_dir_path)
    for dataset in datasets:
        if dataset.endswith(".zip"):
            tasks.append(dataset)

    pool = multiprocessing.Pool(1)
    count = 0
    for res in pool.imap(worker, tasks):
        count += 1
        print(count)


if __name__ == "__main__":
    main()