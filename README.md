# vr-motion-sickness-modelling
Modelling VR motion sickness from the VR.net Dataset

## DataExtractionScripts
**Data Extraction.py**: This contains scripts to extract json files from the VRLOG zip folders.

**Transfer.py**: Script to transfer zip files from Google Drive to the server

**Combined.py**: This script is used to run the whole extraction process. It will convert and store data in the following hierarchy. It uses several support files mentioned in the script and also creates a log file and a summary csv.

```
VR_NET
│
├── 10_1_Wild_West
│   ├── data_file.csv
│   ├── video
│   │   ├── 230.jpg
│   │   ├── 240.jpg
│   │   ├── .
│   │   ├── .
│   │   ├── .
├── 10_2_Earth_Gym
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
## ScriptsForDataVersion1
This folder contains some scripts used to analyse the data collected in the first round.

## Scripts for VideoMAE model training
Details of these scripts are in this repository:
```https://github.com/augmented-human-lab/vrnet.git```
We have trained this model previously to predict the number of objects in a video frame. A working version is in our lab server 137.132.83.109 at the location: 
```/home/chitra/Workspace/VRHook/VideoMAE```