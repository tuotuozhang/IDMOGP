import numpy as np
import pickle
import re
import os
import glob

def bestInd(algorithmName,dataSetName):
    cwd = os.getcwd()
    filePath = os.pardir+ '/../'+'results/' + str(algorithmName) + '/' + dataSetName + '/'
    filePath = os.pardir+ '/../'+'results/' + str(algorithmName) + '/'
    files=os.chdir(filePath)
    #print(os.getcwd())
    txt_files=glob.glob('*FinalResultson'+dataSetName+'.txt')
    bestInd = list()
    #print(len(txt_files))
    for i in range(0, len(txt_files)):
        f=open(txt_files[i],'r')
        totalFinalResults=f.readlines()
        f.close()
        #print(totalFinalResults[15])
        bestInd.append(totalFinalResults[15])
    #print(bestInd)
    files=os.chdir(cwd)
    return bestInd
    #print(totalBestInd[0])
