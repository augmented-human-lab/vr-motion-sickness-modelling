# Description
This folder contains the following files:
1. Arrange.py: remove the columns that are missing
2. Combined.py: All Combined extraction
3. Controller_extraction.py: Added missing controller exraction
4. Data Extraction.py: Extracting and creating pickle files
5. Face_Extraction.py: extracting facial expression data
6. Failed_files.py: Script tor un failed files in earlier extractions
7. Files_without_pose.py: Extracting files without pose information
8. File_Extraction.py: Combined extaction with face and aving as csv
8. logger_config.py: logger configuration
10. make_archive.py: archive the whole gaming sessions to one archive file
11. transfer.py: transfer GD data to server
12. zip_files.py: zip the files game wise
13. Extraction.ipynb: testing ground for scripting files

# VR_NET Dataset

The VR_NET dataset comprises records of game play sessions for various virtual reality (VR) games and associated motion sickness scores with ~1 minute intervals. The sessions are organized based on the specific game played. 

The folder structure is as follows:

- Each session is labeled in the format: `<Participant ID>_<Session ID>_<Game Name>`.
- Each participant has played two sessions, hence the inclusion of the session ID.
- Zipped files are provided for each of the different games, with the following folder structure:

```
Wild_West
│
├── 10_1_Wild_West
│   ├── data_file.csv
│   ├── video
│   │   ├── 230.jpg
│   │   ├── 240.jpg
│   │   ├── .
│   │   ├── .
│   │   ├── .
│   
├── 11_2_Wild_West
│   ├── data_file.csv
│   ├── video
│   │   ├── 230.jpg
│   │   ├── 240.jpg
│   │   ├── .
│   │   ├── .
│   │   ├── .
.
.
.
```


### Dataset Information

- **Name**: VR_NET
- **Format**: CSV and jpg
- **Total Entries**: 798 game play sessions
- **data_file.csv**: The meta data are represented by the columns mentioned below.
### Keys and Associated Columns

#### Pose
- **Columns**:
  - `head_dir`
  - `head_pos`
  - `head_vel`
  - `head_angvel`
  - `left_eye_dir`
  - `left_eye_pos`
  - `left_eye_vel`
  - `left_eye_angvel`
  - `right_eye_dir`
  - `right_eye_pos`
  - `right_eye_vel`
  - `right_eye_angvel`

#### Gaze
- **Columns**:
  - `left_eye`
  - `right_eye`
  - `confidence`
  - `is_valid`

#### Control
- **Columns**:
  - `ConnectedControllerTypes`
  - `Buttons`
  - `Touches`
  - `NearTouches`
  - `IndexTrigger`
  - `HandTrigger`
  - `Thumbstick`

#### Scene
- **Fields**:
  - `object_name`
  - `bounds`
  - `m_matrix`
  - `camera_name`
  - `p_matrix`
  - `v_matrix`

#### Video
- **Columns**:
  - `video` : contains the path to the relevant frame image

#### Motion Sickness Rating
- **Columns**:
  - `MS_rating`

#### Face
- **Columns**:
  - `weights`
  - `confidence`
  - `validity`

#### Important Notes:

1. **Missing Data Handling**: In instances where specific data is absent for a given frame, the corresponding position in the dataset contains a null value.

2. **Variability in Data Availability**: It's important to acknowledge that certain games within the dataset may lack scene and face information. In such cases, the relevant columns for these fields have been excluded from the CSV file. For a comprehensive understanding of data availability, please refer to the `main_log.csv`, which provides a detailed overview indicating the presence or absence of specific data types. For instance, if the "pose data" column is marked as 1, it signifies that the corresponding session includes pose data; conversely, a value of 0 indicates the absence of pose data entirely.

3. **Game and Session Statistics**: The dataset encompasses a total of 82 distinct games, each associated with a range of 7 to 11 valid sessions. For detailed information regarding available games and the corresponding number of sessions, please consult the `main_file.csv`.
