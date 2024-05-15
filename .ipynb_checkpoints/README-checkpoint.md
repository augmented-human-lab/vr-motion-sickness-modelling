# VR Motion Sickness Dataset

Welcome to the VR Motion Sickness Dataset repository. This dataset is a comprehensive collection designed for researchers and developers working on understanding motion sickness in VR gaming environments.

## Dataset Overview

The dataset comprises data from 1,000 game sessions, of which 814 are currently available. Approximately 80% of these sessions are from Metaworld (Horizon), while the remaining 20% cover a variety of popular computer games. Each session lasts for a minimum of 12 minutes, providing a rich set of data for analysis.

### Data Points

During each gaming session, the following information is captured and logged at a granularity of roughly 60 frames per second:

- Camera movement
- Controller movement
- Eye gaze
- Facial expressions
- Pose
- Scene Data
- Motion Sickness score

In addition to the above, RGB data is captured at a rate of 10 frames per second to provide visual context to the recorded sessions.

### Motion Sickness Scores

Each session includes a user-labelled motion sickness score ranging from 1 to 5, where 1 indicates no motion sickness and 5 represents severe motion sickness. This subjective measure allows for the correlation of logged data with reported user discomfort.

## File Structure

The dataset is organized into individual CSV files, one for each game session. The layout of these CSV files is detailed in the image below. Please refer to this layout for understanding the structure of the data, including column headers and data types.
![DatasetLayout](images/datasetLayout.png)

## Usage

This dataset is intended for use in research related to gaming, virtual reality, and human-computer interaction, particularly in studies focused on identifying and mitigating motion sickness. It can also be valuable for developers of VR content aiming to create more comfortable gaming experiences.

### Prerequisites

To work with this dataset, you will need:

- A CSV file reader or a programming language capable of parsing CSV files (e.g., Python, R)
- Basic understanding of data analysis and possibly machine learning techniques

### Importing Data

Here is an example of how you might load a CSV file from this dataset using Python with pandas:

```python
import pandas as pd

# Replace 'session_001.csv' with the path to the CSV file you wish to load
session_data = pd.read_csv('session_001.csv')

print(session_data.head())
```

## Contributions and Feedback

We welcome contributions and feedback from the community. If you have suggestions for improving the dataset, encounter any issues, or have used the dataset in your research and would like to share your findings, please feel free to open an issue or a pull request in this repository.

## License

This dataset is provided for academic and research purposes only. Commercial use is strictly prohibited. Please review the LICENSE file for detailed licensing information.

## Acknowledgements

We would like to thank all the participants who contributed to the creation of this dataset and the research teams involved in the data collection and preprocessing efforts.

---

## DataExtractionScripts
**Data Extraction.py**: This contains scripts to extract csv files from the VRLOG zip folders.
**Transfer.py**: Script to transfer zip files from Google Drive to the server

## ScriptsForDataVersion1
This folder contains some scripts used to analyse the data collected in the first round.

## Scripts for VideoMAE model training
Details of these scripts are in this repository:
```https://github.com/augmented-human-lab/vrnet.git```
We have trained this model previously to predict the number of objects in a video frame. A working version is in our lab server 137.132.83.109 at the location: 
```/home/chitra/Workspace/VRHook/VideoMAE```
