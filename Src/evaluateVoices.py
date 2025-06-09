#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TTS.utils.manage import ModelManager

import sys
sys.path.insert(0, "../Src/")
import generateVoices as gv
from os.path import exists
import pandas as pd
from pydub import AudioSegment
import speech_recognition as sr
import jiwer
import json
import numpy as np

get_ipython().run_line_magic('autosave', '5')


# In[2]:


def evaluateWER(generatedAudioFileName, sentence):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(generatedAudioFileName) as source:
            audio = recognizer.record(source)
            hypothesis = recognizer.recognize_google(audio)
        wer = jiwer.wer(sentence, hypothesis)
        print("WER:", wer)
        return wer
    except sr.UnknownValueError:
        print(f"Could not understand audio: {generatedAudioFileName}")
        return 1.0
    except Exception as e:
        print(f"Error processing {generatedAudioFileName}: {e}")
        return 1.0


# In[3]:


def evaluateWERForModel(model_dir_name, outputCSVFile):
    outputDF = pd.read_csv(outputCSVFile)
    speakers = outputDF['speakerId']
    wordErrorRateArray = []
    for speaker in speakers:
        sentence = outputDF[outputDF['speakerId'] == speaker]['generatedSentence'].values[0]
        generatedAudioFileName = f'../Data/ttsOutputs/{model_dir_name}/{speaker}.wav'
        #evaluate word error rate for geneatedAudioFileName vs sentence
        wer = evaluateWER(generatedAudioFileName, sentence)
        wordErrorRateArray.append(wer)
    if not wordErrorRateArray:
        print(f"No valid WER values for model {model_dir_name}")
        return 1.0, 0.0
    average = np.mean(wordErrorRateArray)
    std_error = np.std(wordErrorRateArray, ddof=1) / np.sqrt(len(wordErrorRateArray))
    return average, std_error


# In[4]:


def main():
    wordErrorRateResults = []
    if exists('../Data/wordErrorRateResults.json'):
        with open('../Data/wordErrorRateResults.json', 'r') as f:
            wordErrorRateResults = json.load(f) 
    
    englishModels = gv.getRawEnglishModelNames()
    for model in englishModels:
        print(f"Processing model: {model}")
        model_dir_name = model.replace("/", "_")
        outputCSVFile = f'../Data/ttsOutputs/{model_dir_name}_generatedSentences.csv'
        if not exists(outputCSVFile):
            continue
        averageWER, stdErrorWER = evaluateWERForModel(model_dir_name, outputCSVFile)
        wordErrorRateResults.append({
            "model": model,
            "averageWER": averageWER,
            "stdErrorWER": stdErrorWER
        })
        with open('../Data/wordErrorRateResults.json', 'w') as f:
            json.dump(wordErrorRateResults, f)
    wordErrorRateResults = pd.DataFrame(wordErrorRateResults)
    wordErrorRateResults = wordErrorRateResults.sort_values(by=['averageWER'], ascending=True)
    display(wordErrorRateResults)
    wordErrorRateResults.to_csv('../Data/wordErrorRateResults.csv', index=False)
    print('done')


# In[5]:


if __name__ == '__main__':
    main()

