import csv
import pandas as pd
import numpy as np
def GetMotionSicknessScore(csvfile):
    scores = []
    timestamps = []
    flines = open(csvfile,'r').readlines()
    for line in flines:
        if 'auto' in line.split(',')[-1]: continue
        scores.append(int(line.split(',')[1]))
        timestamps.append(int(line.split(',')[0]))
    return scores,timestamps

import glob as glob
import os
import matplotlib.pylab as plt
mother = '../SelfReportData'
games = os.listdir('../DataVersion2')
if '.DS_Store' in games:
    games.remove('.DS_Store')
games = ['Beat Saber','Cartoon Network','Job Simulator', 'Monster Awakens', 'Traffic Cop', 'Voxel Shot','VR Rome']
shapes = ['ro','b*','c^','kx','gs','m+','rs','b+','co','k*','g^','mx','r^','bx','cs','k+','go','m*']
plt.figure()
gameind = 1
for game in games:   
    print(game)
    game_scores = []
    plt.subplot(2,5,gameind)
    i=0
    participants = []
    for report in glob.glob(mother+'/*.csv'):
        if game not in report: continue
        scores,timestamps = GetMotionSicknessScore(report)
        print(scores)
        participantid = int(report.split('/')[-1].split(' ')[0])
        plt.plot(scores,shapes[participantid]+'-')
        game_scores = game_scores+scores
        participants.append(participantid)
        i+=1
    plt.title(game)
    # plt.ylabel('Motion Sickness Level')
    # plt.xlabel('Reporting Instances')
    plt.ylim(0,5)
    plt.legend(participants)
    # print(game_scores)
    print(round(np.median(game_scores),2),round(np.mean(game_scores),2),round(np.std(game_scores),2),max(game_scores))
    gameind+=1
plt.show()