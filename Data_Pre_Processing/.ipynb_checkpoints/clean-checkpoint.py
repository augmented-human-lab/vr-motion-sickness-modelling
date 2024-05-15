import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import csv
import math
import json
import ast
import cv2


dtypes = {
    'v_matrix': 'str',
    'camera_name': 'str',
    # 'object_name' : 'str',
    # 'bounds': 'str',
    # 'm_matrix': 'str'
    # define other columns as needed
}
# Selecting rows within the ratings 

# Define a function to calculate quaternion rotation change
def calculate_rotation_change(curr_quaternion, prev_quaternion):
    dot_products = np.sum(curr_quaternion * prev_quaternion, axis=1)
    angle_changes = 2 * np.arccos(np.abs(dot_products))
    return angle_changes

# Define a function to calculate magnitude change of position
def calculate_speed(curr_x, curr_y, curr_z, t1, prev_x, prev_y, prev_z, t2):
    pos_change= ((curr_x - prev_x)**2 + (curr_y - prev_y)**2 + (curr_z - prev_z)**2)**0.5
    speed= pos_change*1000/(t1-t2)
    return speed

def calculate_position_change(curr_x, curr_y, curr_z, prev_x, prev_y, prev_z):
    return ((curr_x - prev_x)**2 + (curr_y - prev_y)**2 + (curr_z - prev_z)**2)**0.5
    

def calculate_magnitude_dir(curr_x, curr_y, prev_x, prev_y):
    magnitude= ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
    dir_x= (curr_x - prev_x)/magnitude
    dir_y= (curr_y - prev_y)/magnitude
    return magnitude,dir_x, dir_y

def extract_position_and_euler_angles_from_matrix(matrix):
    matrix = np.array(matrix).reshape(4, 4).T
    # print(matrix)
    x,y,z = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    pitch = np.arcsin(-rotation_matrix[2, 0])
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    return x,y,z, yaw, pitch, roll

def check_for_text(lst):
    ans=0
    if 'TextMeshPro' in lst:
        ans=1
    else:
        ans=0
    return ans

def calculate_brightness(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate average pixel intensity
    avg_intensity = np.mean(gray_image)
    return avg_intensity

def detect_lighting_changes(images):
    # Calculate average pixel intensity for each image
    # image = image.astype('uint8')
    brightness_values = []
    for image in images:
        if image is not None:
            brightness_values.append(calculate_brightness(image.astype('uint8')))
        else:
            # Handle the case where the image is None
            brightness_values.append(0)
    brightness_changes = [abs(brightness_values[i] - brightness_values[i-1]) for i in range(1,len(brightness_values))]
    brightness_changes.insert(0, 0)
    return brightness_changes



def quaternion_to_euler(row):
    x, y, z, w = row
    # Calculate roll
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    # Calculate pitch
    sin_pitch = 2 * (w * y - z * x)
    if abs(sin_pitch) >= 1:
        pitch = math.copysign(math.pi / 2, sin_pitch)
    else:
        pitch = math.asin(sin_pitch)
    
    # Calculate yaw
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    return pd.Series({'le_roll': roll, 'le_pitch': pitch, 'le_yaw': yaw})


def clean(df_sess_path):
    df_cleaned=[]
    df_trimmed=[]
    df_sess_path_df=os.path.join(df_sess_path,'data_file.csv')
    df_sess=pd.read_csv(df_sess_path_df, dtype=dtypes)
    df_sess_path_vid=os.path.join(df_sess_path,'video')
    # df_cleaned=[]
    # Find the index of the first and last non-null rating
    first_rating_index = df_sess['MS_rating'].first_valid_index()
    last_rating_index = df_sess['MS_rating'].last_valid_index()

    # Slice the DataFrame to retain rows only between the first and last rating
    df_trimmed = df_sess.loc[first_rating_index-1:last_rating_index]
    # print("nan_count = ", df_trimmed['Buttons'].isna().sum())
    df_trimmed.reset_index(drop=True, inplace=True)

    #drop the columns without the control ratings/ fill the rating with mode of each column
    col_to_fill_mode=[ 'Buttons','IndexTrigger', 'HandTrigger','Thumbstick','left_eye','right_eye','camera_name','v_matrix','p_matrix','object_name']
    col_to_fill_median=[ 'Touches','NearTouches']
    df_cleaned=df_trimmed


    # declare the columns to be filled with mode and medians
    modes = df_cleaned[col_to_fill_mode].mode().iloc[0]
    medians = df_cleaned[col_to_fill_median].median()
    # print(modes)

    df_cleaned[col_to_fill_mode]=df_cleaned[col_to_fill_mode].fillna(modes)
    df_cleaned[col_to_fill_median]=df_cleaned[col_to_fill_median].fillna(medians)

    df_cleaned[['IndTrig_L', 'IndTrig_R']] = df_cleaned['IndexTrigger'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    df_cleaned[['HandTrig_L', 'HandTrig_R']] = df_cleaned['HandTrigger'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    # df_cleaned[['IndTrig_L', 'IndTrig_R','HandTrig_L', 'HandTrig_R']] = df_cleaned[['IndTrig_L', 'IndTrig_R','HandTrig_L', 'HandTrig_R']].round()

    # Apply the function to calculate magnitude for each row
    df_cleaned[['PrevX_L', 'PrevY_L', 'PrevX_R', 'PrevY_R']] = df_cleaned['Thumbstick'].apply(lambda x: pd.Series(ast.literal_eval(x)))

    # Calculate tmagnitude for left hand and right hand
    df_cleaned['Thumb_L'] = calculate_magnitude_dir(df_cleaned['PrevX_L'], df_cleaned['PrevY_L'], df_cleaned['PrevX_L'].shift(1), df_cleaned['PrevY_L'].shift(1))[0]
    df_cleaned['Thumb_R'] = calculate_magnitude_dir(df_cleaned['PrevX_R'], df_cleaned['PrevY_R'], df_cleaned['PrevX_R'].shift(1), df_cleaned['PrevY_R'].shift(1))[0]
    
    df_cleaned['Thumb_L_x'] = calculate_magnitude_dir(df_cleaned['PrevX_L'], df_cleaned['PrevY_L'], df_cleaned['PrevX_L'].shift(1), df_cleaned['PrevY_L'].shift(1))[1]
    df_cleaned['Thumb_R_x'] = calculate_magnitude_dir(df_cleaned['PrevX_R'], df_cleaned['PrevY_R'], df_cleaned['PrevX_R'].shift(1), df_cleaned['PrevY_R'].shift(1))[1]
    
    df_cleaned['Thumb_L_y'] = calculate_magnitude_dir(df_cleaned['PrevX_L'], df_cleaned['PrevY_L'], df_cleaned['PrevX_L'].shift(1), df_cleaned['PrevY_L'].shift(1))[2]
    df_cleaned['Thumb_R_y'] = calculate_magnitude_dir(df_cleaned['PrevX_R'], df_cleaned['PrevY_R'], df_cleaned['PrevX_R'].shift(1), df_cleaned['PrevY_R'].shift(1))[2]
    
    
    #Gaze

    # print("nan_count = ", df_cleaned['left_eye'].isna().sum())
    # Apply the function to extract quaternion and position components for each row
    df_cleaned[['le_x', 'le_y', 'le_z', 'le_w', 'le_p1', 'le_p2', 'le_p3']] = df_cleaned['left_eye'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    
    # df_cleaned[['le_roll', 'le_pitch', 'le_yaw']]= quaternion_to_euler(df_cleaned[['le_x', 'le_y', 'le_z', 'le_w']].values)
    df_cleaned[['le_roll', 'le_pitch', 'le_yaw']] = df_cleaned[['le_x', 'le_y', 'le_z', 'le_w']].apply(quaternion_to_euler, axis=1)

    # print(df_cleaned[['x', 'y', 'z', 'w']].values)
    # Calculate rotation change and position change
    df_cleaned['LE_speed'] = calculate_speed(df_cleaned['le_p1'], df_cleaned['le_p2'], df_cleaned['le_p3'], df_cleaned['timestamp'], df_cleaned['le_p1'].shift(1), df_cleaned['le_p2'].shift(1), df_cleaned['le_p3'].shift(1), df_cleaned['timestamp'].shift(1))
    df_cleaned['LE_pos_change']=calculate_position_change(df_cleaned['le_p1'], df_cleaned['le_p2'], df_cleaned['le_p3'], df_cleaned['le_p1'].shift(1), df_cleaned['le_p2'].shift(1), df_cleaned['le_p3'].shift(1))
    
    df_cleaned[['re_x', 're_y', 're_z', 're_w', 're_p1', 're_p2', 're_p3']] = df_cleaned['right_eye'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    # print(df_cleaned[['x', 'y', 'z', 'w']].values)
    # Calculate rotation change and position change
    df_cleaned[['re_roll', 're_pitch', 're_yaw']]= df_cleaned[['re_x', 're_y', 're_z', 're_w']].apply(quaternion_to_euler, axis=1)
    df_cleaned['RE_speed'] = calculate_speed(df_cleaned['re_p1'], df_cleaned['re_p2'], df_cleaned['re_p3'], df_cleaned['timestamp'], df_cleaned['re_p1'].shift(1), df_cleaned['re_p2'].shift(1), df_cleaned['re_p3'].shift(1), df_cleaned['timestamp'].shift(1))
    df_cleaned['RE_pos_change']=calculate_position_change(df_cleaned['re_p1'], df_cleaned['re_p2'], df_cleaned['re_p3'], df_cleaned['re_p1'].shift(1), df_cleaned['re_p2'].shift(1), df_cleaned['re_p3'].shift(1))


    #Pose: Head

    df_cleaned[['h_x', 'h_y', 'h_z', 'h_w']] = df_cleaned['head_dir'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    df_cleaned[['h_roll', 'h_pitch', 'h_yaw']] = df_cleaned[['h_x', 'h_y', 'h_z', 'h_w']].apply(quaternion_to_euler, axis=1)
    # df_cleaned['Head_Rot_Change'] = calculate_rotation_change(df_cleaned[['h_x', 'h_y', 'h_z', 'h_w']].values, df_cleaned[['h_x', 'h_y', 'h_z', 'h_w']].shift(1).values)

    df_cleaned[['h_p1', 'h_p2', 'h_p3']] = df_cleaned['head_pos'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    df_cleaned['Head_Speed'] = calculate_speed(df_cleaned['h_p1'], df_cleaned['h_p2'], df_cleaned['h_p3'],df_cleaned['timestamp'], df_cleaned['h_p1'].shift(1), df_cleaned['h_p2'].shift(1), df_cleaned['h_p3'].shift(1), df_cleaned['timestamp'].shift(1))
    df_cleaned['Head_pos_change']=calculate_position_change(df_cleaned['h_p1'], df_cleaned['h_p2'], df_cleaned['h_p3'], df_cleaned['h_p1'].shift(1), df_cleaned['h_p2'].shift(1), df_cleaned['h_p3'].shift(1))
    
    df_cleaned[['h_vp1', 'h_vp2', 'h_vp3']] = df_cleaned['head_vel'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    df_cleaned['Head_Velocity_Change'] = calculate_position_change(df_cleaned['h_vp1'], df_cleaned['h_vp2'], df_cleaned['h_vp3'], df_cleaned['h_vp1'].shift(1), df_cleaned['h_vp2'].shift(1), df_cleaned['h_vp3'].shift(1))

    df_cleaned[['h_avp1', 'h_avp2', 'h_avp3']] = df_cleaned['head_angvel'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    df_cleaned['Head_AngVel_Change'] = calculate_position_change(df_cleaned['h_avp1'], df_cleaned['h_avp2'], df_cleaned['h_avp3'], df_cleaned['h_avp1'].shift(1), df_cleaned['h_avp2'].shift(1), df_cleaned['h_avp3'].shift(1))

    
    #camera extractions
    
    df_cleaned['v_matrix'] = df_cleaned['v_matrix'].apply(ast.literal_eval)
    df_cleaned['camera_name'] = df_cleaned['camera_name'].apply(ast.literal_eval)
    df_cleaned['p_matrix'] = df_cleaned['p_matrix'].apply(ast.literal_eval)
    df_cleaned['object_name'] = df_cleaned['object_name'].apply(ast.literal_eval)


    cam_name=df_cleaned['camera_name'].to_list()
    timest=df_cleaned['timestamp'].to_list()
    v_matrix=df_cleaned['v_matrix'].to_list()
    p_matrix=df_cleaned['p_matrix'].to_list()

    c_roll=np.zeros(len(cam_name))
    c_pitch=np.zeros(len(cam_name))
    c_yaw=np.zeros(len(cam_name))
    c_x=np.zeros(len(cam_name))
    c_y=np.zeros(len(cam_name))
    c_z=np.zeros(len(cam_name))
    c_speed=np.zeros(len(cam_name))
    c_fov=np.zeros(len(cam_name))
    # c_fov_change=np.zeros(len(cam_name))

    for i in range(0,len(cam_name)):
        for y in range(len(cam_name[i])):
            # print(cam_name[i][y])
            if cam_name[i][y]=='CenterEyeAnchor':
                v_mat=v_matrix[i][y]
                # print(v_mat)
                c_x[i], c_y[i], c_z[i], c_yaw[i], c_pitch[i], c_roll[i]=extract_position_and_euler_angles_from_matrix(v_mat)
                c_fov[i]=p_matrix[i][y][0]
                
                if i!=0:
                    timediff=(timest[i]-timest[i-1])/1000
                    velocity=[(c_x[i]-c_x[i-1])/timediff, (c_y[i]-c_y[i-1])/timediff, (c_z[i]-c_z[i-1])/timediff]
                    c_speed[i]=np.linalg.norm(velocity)
                    # c_fov_change=(c_fov[i]-c_fov[i-1])*100000000

    # print(c_roll[0:5])
    df_cleaned['c_roll']=c_roll
    df_cleaned['c_pitch']=c_pitch
    df_cleaned['c_yaw']=c_yaw
    df_cleaned['c_x']=c_x
    df_cleaned['c_y']=c_y
    df_cleaned['c_z']=c_z
    df_cleaned['c_speed']=c_speed
    df_cleaned['c_fov']=c_fov
    # df_cleaned['c_fov_change']=c_fov_change
    df_cleaned['c_change_pitch']=abs(df_cleaned['c_pitch']-df_cleaned['c_pitch'].shift(1))
    df_cleaned['c_change_roll']=abs(df_cleaned['c_roll']-df_cleaned['c_roll'].shift(1))
    df_cleaned['c_change_yaw']=abs(df_cleaned['c_yaw']-df_cleaned['c_yaw'].shift(1))
    # Apply the function to each row of the 'object_name' column
    df_cleaned['text_presence'] = df_cleaned['object_name'].apply(check_for_text)
    # Display the modified DataFrame
    # print(df)
    # print("nan_count = ", df_cleaned['video'].isna().sum())
    # Example usage
    v_paths=df_cleaned['video'].to_list()
    # print(vpaths[0])
    vpaths=[]
    ind=0

    for i in range(len(v_paths)):
        # print(video_list[video])
        if str(v_paths[i]).startswith('vid'):
            vpaths.append( os.path.join(df_sess_path,str(v_paths[i]) + '.jpg'))
            ind=i
        else:
            # print('yes')
            # print(i)
            vpaths.append(os.path.join(df_sess_path,str(v_paths[ind]) + '.jpg'))
    # print(vpaths)
    
    images = [cv2.imread(image_path) for image_path in vpaths]
    df_cleaned['change in brightness']=detect_lighting_changes(images)
    # print(images)
    
    df_cleaned['diff_HE_yaw'] = abs(df_cleaned['le_yaw'] - df_cleaned['h_yaw'])
    df_cleaned['diff_HE_roll'] = abs(df_cleaned['le_roll'] - df_cleaned['h_roll'])
    df_cleaned['diff_HE_pitch'] = abs(df_cleaned['le_pitch'] - df_cleaned['h_pitch'])
    
    df_cleaned['hour'] = pd.to_datetime(df_cleaned['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai').dt.hour
    
    df_cleaned.reset_index(drop=True, inplace=True)
    
    df_final=df_cleaned[['frame', 'hour','Buttons', 'IndTrig_L', 'IndTrig_R', 'HandTrig_L',
       'HandTrig_R', 'Thumb_L', 'Thumb_R', 'Thumb_L_x', 'Thumb_R_x', 'Thumb_L_y', 'Thumb_R_y', 
       'LE_speed', 'LE_pos_change','Head_Speed', 'Head_pos_change', 'Head_Velocity_Change', 'Head_AngVel_Change', 
       'c_speed',  'text_presence', 'change in brightness','diff_HE_yaw', 'diff_HE_roll', 'diff_HE_pitch' , 
       'c_change_pitch', 'c_change_roll', 'c_change_yaw', 'MS_rating']]
    
    return df_final


