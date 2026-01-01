#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, "../Src/")
import generateVoices as gv
import evaluateVoices as ev
import pandas as pd
import os
from os.path import exists
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pydub import AudioSegment
from TTS.api import TTS
from TTS.utils.manage import ModelManager
import torch
from TTS.utils.radam import RAdam
import numpy.core.multiarray
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import tempfile

torch.serialization.add_safe_globals([RAdam, numpy.core.multiarray.scalar])

get_ipython().run_line_magic('autosave', '5')


# In[2]:


# tts = gv.generateTTS('tts_models/en/ljspeech/vits--neon')
# tts.tts_to_file(
#     text='The anti-slavery movement took many forms.',
#     file_path='test.wav'
# )


# In[3]:


def getBestModelName():
    print('starting getBestModelName')
    wordErrorRateResults = pd.read_csv("../Data/wordErrorRateResults.csv")
    bestModel = wordErrorRateResults.sort_values(by="averageWER").iloc[0]
    print('finishing getBestModelName')
    return bestModel["model"]


# In[4]:


def getSourceVoice(speaker,speakerDf,datasetName="train"):
    print('starting getSourceVoice')
    if datasetName == "train":
        sourceVoiceFile = f"../Data/concatenatedInputs/{speaker}.wav"
    else:
        sourceVoiceFile = f"../Data/concatenatedTestInputs/{speaker}.wav"
    if exists(sourceVoiceFile):
        print('finishing getSourceVoice')
        return sourceVoiceFile
    gv.concatenateAudio(speaker, speakerDf,datasetName)
    if exists(sourceVoiceFile):
        print('finishing getSourceVoice')
        return sourceVoiceFile
    else:
        print(f"Source voice file for {speaker} not found after concatenation.")
        print('finishing getSourceVoice')
        return None


# In[5]:


def getFADJson(datasetName):
    print('starting getFADJson')
    fadJson = []
    if exists(f"../Data/fad_{datasetName}.json"):
        with open(f"../Data/fad_{datasetName}.json", "r") as f:
            fadJson = json.load(f)
    print('finishing getFADJson')
    return fadJson


# In[6]:


def saveFADJson(datasetName, fadJson):
    print('starting saveFADJson')
    with open(f"../Data/fad_{datasetName}.json", "w") as f:
        json.dump(fadJson, f)
    print('finishing saveFADJson')


# In[7]:


def createFakeAudio(speakerDf, sourceVoice, modelName, datasetName):
    print('starting createFakeAudio')
    fadJson = getFADJson(datasetName)
    textFiles = speakerDf[speakerDf['path_from_data_dir'].str.contains('.TXT', na=False)]
    textFileNames = textFiles['path_from_data_dir'].tolist()

    for textFileName in textFileNames:
        realFile = f"../Data/data/{textFileName.replace('.TXT', '.WAV.wav')}"
        fakeAudioFile = f"../Data/fakeAudio/{textFileName.replace('.TXT', '.wav')}"
        if exists(fakeAudioFile):
            print(f"Fake audio file {fakeAudioFile} already exists.")
            continue
        os.makedirs(os.path.dirname(fakeAudioFile), exist_ok=True)
        model = gv.generateTTS(modelName)
        sentence = gv.readSentenceFromFile(textFileName)
        gv.generateAndNormalizeAudio(model, sentence, sourceVoice, fakeAudioFile)
        if not exists(fakeAudioFile):
            raise FileNotFoundError(f"Failed to create fake audio file: {fakeAudioFile}")
        fadJson.append({
            "file": fakeAudioFile,
            "text": sentence,
            "isFake": True,
        })
        if not exists(realFile):
            raise FileNotFoundError(f"Failed to find real audio file: {realFile}")
        fadJson.append({
            "file": realFile,
            "text": sentence,
            "isFake": False,
        })
        saveFADJson(datasetName,fadJson)
        print('finishing createFakeAudio')



# In[8]:


def prepareFADData(modelName,datasetName):
    print('starting prepareFADData')
    print(f"Creating voices for {modelName} on {datasetName} dataset")
    
    df = gv.readCsv(datasetName)
    speakers = gv.getSpeakers(df)
    for speaker in speakers:
        speakerDf = gv.getFilesBySpeaker(df, speaker)
        sourceVoice = getSourceVoice(speaker, speakerDf,datasetName)
        if sourceVoice is None:
            print(f"Skipping {speaker} due to missing source voice.")
            continue
        createFakeAudio(speakerDf, sourceVoice, modelName, datasetName)
    fad_json = getFADJson(datasetName)
    unique_fad_json = [dict(t) for t in {tuple(sorted(d.items())) for d in fad_json}]
    voiceDF = pd.DataFrame(unique_fad_json)
    voiceDF.to_csv(f"../Data/fad_{datasetName}.csv", index=False)
    print('finishing prepareFADData')
    return voiceDF



# In[9]:


def main():
    print('starting main')
    bestModelName = getBestModelName()
    trainDF = prepareFADData(bestModelName, 'train')
    testDF =  prepareFADData(bestModelName, 'test')
    display(trainDF.head())
    display(testDF.head())
    print('done')


# In[10]:


if __name__ == '__main__':
    main()

