# Description

This folder contans all the notebooks used for data preprocessing.

1. EDA: The EDA documentation is available [here](https://docs.google.com/document/d/1Frw1zzp3W0hTpj_qwa9rsJNaEwWeYSd_Pasuo6qYotk/edit?usp=sharing). This contains two notebooks which was used to generate the plots available in the document.
1.1 Documentation 1
1.2 Documentation 2
1.3 game_session.csv: All the games and the number of sessions we have for each game

2. Econ: Contains all the files used for the preliminery Statistical Analysis. 
2.1: data: contains the final data used for the Analysis 1, 2 and 3 in the report
2.2 Econ.ipynb: te notbook used to create data and do some further preliminery analysis
2.3 games_used: gaming sessions that were used for Analysis 1 and 2

3. notebooks: Notebooks used for the preliminery analsis: this is not cleaned, refer to the EDA noteboooks which are commented and cleaned

4. aggregation.py: Aggregating the data between two motion sickness ratings

5. clean.py: cleaning and pre processing the data and saving as a csv

6. pre_process.py: pre processing is the combination and the main file which uses aggregating and cleaning scripts. Ths combines and save the final file as a csv with all the sesions each having mostly 13 rows.