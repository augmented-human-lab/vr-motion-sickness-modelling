import pandas as pd
import numpy as np

def clean_buttons(buttons):
    if np.isnan(buttons):
        return 0
    buttons = int(buttons)
    buttons_bin = bin(buttons)[2:]
    button_presses = buttons_bin.count('1')
    return button_presses

def get_a_pressed(buttons):
    if np.isnan(buttons):
        return 0
    return 1 if int(buttons) & 1 else 0

def clean_triggers(trigger):
    if np.isnan(trigger):
        return 0
    return 1 if trigger != 0.0 else 0

def clean_thumbs(thumb):
    if np.isnan(thumb):
        return 0
    return 1 if thumb < 0.0 else 0

# df = pd.read_csv("shreyas.csv")

col_list = ['Buttons', 'IndTrig_L', 'IndTrig_R', 'HandTrig_L', 'HandTrig_R', 'Thumb_L', 'Thumb_R', 'Thumb_L_x', 'Thumb_R_x', 'Thumb_L_y', 'Thumb_R_y', 'LE_speed', 'LE_pos_change', 'Head_Speed', 'Head_pos_change', 'Head_Velocity_Change', 'Head_AngVel_Change', 'c_speed', 'text_presence', 'change in brightness', 'diff_HE_yaw', 'diff_HE_roll', 'diff_HE_pitch', 'c_change_pitch', 'c_change_roll', 'c_change_yaw', 'MS_rating', 'hour']

difficult_agg_list = ['Buttons', 'IndTrig_L', 'IndTrig_R', 'HandTrig_L', 'HandTrig_R', 'Thumb_L_x', 'Thumb_R_x', 'Thumb_L_y', 'Thumb_R_y']

easy_agg_list = [item for item in col_list if item not in difficult_agg_list]




def aggregate(df):
    col_list = ['Buttons', 'IndTrig_L', 'IndTrig_R', 'HandTrig_L', 'HandTrig_R', 'Thumb_L', 'Thumb_R', 'Thumb_L_x', 'Thumb_R_x', 'Thumb_L_y', 'Thumb_R_y', 'LE_speed', 'LE_pos_change', 'Head_Speed', 'Head_pos_change', 'Head_Velocity_Change', 'Head_AngVel_Change', 'c_speed', 'text_presence', 'change in brightness', 'diff_HE_yaw', 'diff_HE_roll', 'diff_HE_pitch', 'c_change_pitch', 'c_change_roll', 'c_change_yaw', 'MS_rating','hour']
    difficult_agg_list = ['Buttons', 'IndTrig_L', 'IndTrig_R', 'HandTrig_L', 'HandTrig_R', 'Thumb_L_x', 'Thumb_R_x', 'Thumb_L_y', 'Thumb_R_y']

    easy_agg_list = [item for item in col_list if item not in difficult_agg_list]
    df= df[col_list]
    
    # Find indices of non-NaN MS_rating values
    non_nan_indices = df[df['MS_rating'].notna()].index.tolist()

    # Initialize a dictionary to hold the groups
    groups = {}

    # Calculate bucket size for each group and extract the group
    for i, idx in enumerate(non_nan_indices):
        # For the first non-NaN value
        if i == 0:
            next_idx = non_nan_indices[i + 1] if i + 1 < len(non_nan_indices) else df.index.max()
            half_gap = (next_idx - idx) // 2
            start_idx = df.index.min()
            end_idx = idx + half_gap
        # For the last non-NaN value
        elif i == len(non_nan_indices) - 1:
            prev_idx = non_nan_indices[i - 1]
            half_gap = (idx - prev_idx) // 2
            start_idx = prev_idx + half_gap
            end_idx = df.index.max()
        # For non-NaN values in between
        else:
            prev_idx = non_nan_indices[i - 1]
            next_idx = non_nan_indices[i + 1]
            half_gap_prev = (idx - prev_idx) // 2
            half_gap_next = (next_idx - idx) // 2
            start_idx = idx - half_gap_prev
            end_idx = idx + half_gap_next

        # Extract the group of rows for the current index
        groups[idx] = (start_idx, end_idx)

    # Display the groups
    # for center_frame, group in groups.items():
    #     print(f"Group centered at frame {center_frame}:")
    #     print(group, "\n")

    easy_df = df[easy_agg_list]
    agg_easy_df = None

    easy_aggregations = {
        'Thumb_L': 'sum',
        'Thumb_R': 'sum',
        'LE_speed': ['mean', 'median'],
        'LE_pos_change': ['mean', 'median', 'sum'],
        'Head_Speed': ['mean', 'median'],
        'Head_pos_change': ['mean', 'median', 'sum'],
        'Head_Velocity_Change': ['mean', 'median'],
        'Head_AngVel_Change': ['mean', 'median'],
        'c_speed': ['mean', 'median'],
        'text_presence': 'sum',
        'change in brightness': ['mean', 'median'],
        'diff_HE_yaw': ['mean', 'median', 'sum'],
        'diff_HE_roll': ['mean', 'median', 'sum'],
        'diff_HE_pitch': ['mean', 'median', 'sum'],
        'c_change_pitch': ['mean', 'median', 'sum'],
        'c_change_roll': ['mean', 'median', 'sum'],
        'c_change_yaw': ['mean', 'median', 'sum'],
        'MS_rating': 'sum',
        'hour': 'median'
    }

    difficult_df = df[difficult_agg_list]
    agg_difficult_df = None

    difficult_agg_list = ['Buttons', 'IndTrig_L', 'IndTrig_R', 'HandTrig_L', 'HandTrig_R', 'Thumb_L_x', 'Thumb_R_x', 'Thumb_L_y', 'Thumb_R_y']
    difficult_df['Buttons_cleaned'] = difficult_df['Buttons'].apply(clean_buttons)
    difficult_df['A_Pressed'] = difficult_df['Buttons'].apply(get_a_pressed)
    difficult_df.drop(columns=['Buttons'], inplace=True)
    difficult_df.rename(columns={'Buttons_cleaned': 'Buttons'}, inplace=True)

    difficult_df['IndTrig_L_cleaned'] = difficult_df['IndTrig_L'].apply(clean_triggers)
    difficult_df.drop(columns=['IndTrig_L'], inplace=True)
    difficult_df.rename(columns={'IndTrig_L_cleaned': 'IndTrig_L'}, inplace=True)
    difficult_df['IndTrig_R_cleaned'] = difficult_df['IndTrig_R'].apply(clean_triggers)
    difficult_df.drop(columns=['IndTrig_R'], inplace=True)
    difficult_df.rename(columns={'IndTrig_R_cleaned': 'IndTrig_R'}, inplace=True)
    difficult_df['HandTrig_L_cleaned'] = difficult_df['HandTrig_L'].apply(clean_triggers)
    difficult_df.drop(columns=['HandTrig_L'], inplace=True)
    difficult_df.rename(columns={'HandTrig_L_cleaned': 'HandTrig_L'}, inplace=True)
    difficult_df['HandTrig_R_cleaned'] = difficult_df['HandTrig_R'].apply(clean_triggers)
    difficult_df.drop(columns=['HandTrig_R'], inplace=True)
    difficult_df.rename(columns={'HandTrig_R_cleaned': 'HandTrig_R'}, inplace=True)

    difficult_df['Thumb_L_x_cleaned'] = difficult_df['Thumb_L_x'].apply(clean_thumbs)
    difficult_df.drop(columns=['Thumb_L_x'], inplace=True)
    difficult_df.rename(columns={'Thumb_L_x_cleaned': 'Thumb_L_x'}, inplace=True)
    difficult_df['Thumb_R_x_cleaned'] = difficult_df['Thumb_R_x'].apply(clean_thumbs)
    difficult_df.drop(columns=['Thumb_R_x'], inplace=True)
    difficult_df.rename(columns={'Thumb_R_x_cleaned': 'Thumb_R_x'}, inplace=True)
    difficult_df['Thumb_L_y_cleaned'] = difficult_df['Thumb_L_y'].apply(clean_thumbs)
    difficult_df.drop(columns=['Thumb_L_y'], inplace=True)
    difficult_df.rename(columns={'Thumb_L_y_cleaned': 'Thumb_L_y'}, inplace=True)
    difficult_df['Thumb_R_y_cleaned'] = difficult_df['Thumb_R_y'].apply(clean_thumbs)
    difficult_df.drop(columns=['Thumb_R_y'], inplace=True)
    difficult_df.rename(columns={'Thumb_R_y_cleaned': 'Thumb_R_y'}, inplace=True)

    difficult_aggregations = {
        'Buttons': 'sum',
        'A_Pressed': 'sum',
        'IndTrig_L': 'sum',
        'IndTrig_R': 'sum',
        'HandTrig_L': 'sum',
        'HandTrig_R': 'sum',
        'Thumb_L_x': 'sum',
        'Thumb_R_x': 'sum',
        'Thumb_L_y': 'sum',
        'Thumb_R_y': 'sum'
    }

    e_flag, d_flag = 0, 0

    # Aggregate values in each interval
    for center_frame, group in groups.items():
        start, end = group
        interval_values_easy = easy_df.loc[start:end]
        new_easy_row = interval_values_easy.agg(easy_aggregations)

        stacked = new_easy_row.T.stack()
        stacked_df = stacked.to_frame('value')
        stacked_df.index = ['_'.join(idx) for idx in stacked.index]
        final_e_row = stacked_df.T

        if (e_flag == 0):
            agg_easy_df = pd.DataFrame(columns=final_e_row.index)
            e_flag = 1
        agg_easy_df = pd.concat([agg_easy_df, final_e_row], axis=0, ignore_index=True)

        interval_values_difficult = difficult_df.loc[start:end]
        new_difficult_row = interval_values_difficult.agg(difficult_aggregations)
        final_d_row = new_difficult_row.to_frame().T
        if (d_flag == 0):
            agg_difficult_df = pd.DataFrame(columns=final_d_row.columns)
            d_flag = 1
        agg_difficult_df = pd.concat([agg_difficult_df, final_d_row], axis=0, ignore_index=True)

    agg_easy_df.drop(columns=['value'], inplace=True)

    # Concatenate easy and difficult aggregation dataframes
    result = pd.concat([agg_easy_df, agg_difficult_df], axis=1)

    return result

