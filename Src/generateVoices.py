#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from os.path import exists
from pydub import AudioSegment
from TTS.api import TTS
from TTS.utils.manage import ModelManager
import os
import torch
from TTS.utils.radam import RAdam
import numpy.core.multiarray
import shutil
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

torch.serialization.add_safe_globals([RAdam, numpy.core.multiarray.scalar])

import sys
sys.path.insert(0, "../Src/")

get_ipython().run_line_magic('autosave', '5')


# In[2]:


#dataSet = pd.read_csv('../Data/train_data.csv')
#dataSet[dataSet['speaker_id'] == 'MMDM0']
# outputDF = pd.read_csv('../Data/ttsOutputs/tts_models_en_ljspeech_tacotron2-DDC_generatedSentences.csv')
# print(outputDF.shape[0])
# uniqueSpeakers = outputDF['speakerId'].unique()
# print(len(uniqueSpeakers))


# In[3]:


def readCsv(dataset):
    return pd.read_csv(f'../Data/{dataset}_data.csv')


# In[4]:


def getSpeakers(df):
    speakerIds = df['speaker_id']
    return list(set(speakerIds))


# In[5]:


def getFilesBySpeaker(df,speakerId):
    return df[df['speaker_id']==speakerId]


# In[6]:


def concatenateAudio(speakerId, speakerDF,datasetName='train'):
    if speakerDF.empty:
        print(f"Empty DataFrame for speaker {speakerId}. Skipping.")
        return
    if datasetName == 'train':
        finalAudioFile = f'../Data/concatenatedInputs/{speakerId}.wav'
    else:
        finalAudioFile = f'../Data/concatenatedTestInputs/{speakerId}.wav'
    audioData = speakerDF[speakerDF['filename'].str.endswith('.wav', na=False)]

    if audioData.empty:
        print(f"No .wav files for speaker {speakerId}. Skipping.")
        return

    audioFileList = audioData['path_from_data_dir']

    if not exists(finalAudioFile):
        concat_audio = AudioSegment.empty()
    else:
        concat_audio = AudioSegment.from_wav(finalAudioFile)
    for audioFile in audioFileList:
        try:
            audio = AudioSegment.from_wav(f'../Data/data/{audioFile}') + AudioSegment.silent(duration=1000)
            concat_audio += audio
        except Exception as e:
            print(f"Failed to load {audioFile}: {e}")

    if len(concat_audio) > 0:
        concat_audio.export(finalAudioFile, format='wav')
    else:
        print(f"No valid audio for speaker {speakerId}, nothing exported.")
    


# In[7]:


def readSentenceFromFile(sentenceFile):
    try:
        with open(f'../Data/data/{sentenceFile}', 'r') as file:
            return " ".join(file.read().strip().split(" ")[2:])
    except Exception as e:
        print(f"Failed to read sentence file {sentenceFile}: {e}")
        return None




# In[8]:


def generateAndNormalizeAudio(tts, sentence, inputAudioFile, outputAudioFile):
    try:
        print(f"Generating audio for sentence: {sentence}")
        tts.tts_with_vc_to_file(
            text=sentence,
            file_path=outputAudioFile,
            speaker_wav=inputAudioFile,
        )

        # Normalize the generated audio
        original_audio = AudioSegment.from_wav(inputAudioFile)
        generated_audio = AudioSegment.from_wav(outputAudioFile)

        gain = original_audio.dBFS - generated_audio.dBFS
        normalized_audio = generated_audio.apply_gain(gain)

        # Export the normalized audio back to the same file
        normalized_audio.export(outputAudioFile, format='wav')
        if not exists(outputAudioFile):
            raise FileNotFoundError(f"Output file {outputAudioFile} was not created.")
        return True
    except Exception as e:
        print(f"Failed to generate or normalize audio: {e}")
        return False


# In[9]:


def saveGeneratedSentences(speakerSentences, model_dir_name, modelDirectory):
    outputFile = f'../Data/ttsOutputs/{model_dir_name}_generatedSentences.csv'
    speakerSentencesJsonFile = f'../Data/ttsOutputs/{model_dir_name}_generatedSentences.json'

    # Remove empty JSON file and directory if JSON is empty
    if exists(speakerSentencesJsonFile):
        try:
            with open(speakerSentencesJsonFile, 'r') as f:
                if not json.load(f):
                    os.remove(speakerSentencesJsonFile)
                    print(f"Removed empty JSON file: {speakerSentencesJsonFile}")
                    shutil.rmtree(modelDirectory, ignore_errors=True)
                    return
        except Exception as e:
            print(f"Error checking/removing JSON file: {e}")

    if speakerSentences:
        try:
            pd.DataFrame(speakerSentences).to_csv(outputFile, index=False)
        except Exception as e:
            print(f"Failed to save generated sentences: {e}")
            shutil.rmtree(modelDirectory, ignore_errors=True)
            return

    if not exists(outputFile):
        print(f"Warning: Output file {outputFile} was not created.")


# In[10]:


def loadGeneratedSentencesFromJson(speakerSentencesPath):
    if exists(speakerSentencesPath):
        with open(speakerSentencesPath, 'r') as f:
            speakerSentences = json.load(f)
    else:
        speakerSentences = []
    return speakerSentences


# In[11]:


def generateTTS(model):
    try:
        # Scoped override of torch.load
        original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: original_torch_load(*args, weights_only=False, **kwargs)
        tts = TTS(model_name=model, progress_bar=False, gpu=False)
        result = tts
    except Exception as e:
        print(f"Failed to load TTS model {model}: {e}")
        result = None
    finally:
        torch.load = original_torch_load  # Restore the original torch.load
    return result


# In[12]:


def process_speaker(speaker, trainDF, tts, modelDirectory):
    outputFilePath = f'{modelDirectory}/{speaker}.wav'
    if exists(outputFilePath):
        print(f"Output file {outputFilePath} already exists. Skipping.")
        return None
    speakerDF = getFilesBySpeaker(trainDF, speaker)
    if speakerDF.empty:
        print(f"No data for speaker {speaker}. Skipping.")
        return None

    textFiles = speakerDF[speakerDF['path_from_data_dir'].str.contains('.TXT', na=False)]
    if textFiles.empty:
        print(f"No valid sentences for speaker {speaker}. Skipping.")
        return None

    chosenSentenceFile = np.random.choice(textFiles['path_from_data_dir'])
    chosenSentence = readSentenceFromFile(chosenSentenceFile)
    if not chosenSentence:
        print(f"Chosen sentence for speaker {speaker} is empty. Skipping.")
        return None

    audioFile = f'../Data/concatenatedInputs/{speaker}.wav'
    if not exists(audioFile):
        print(f"Audio file for {speaker} does not exist. Skipping.")
        return None

    if generateAndNormalizeAudio(tts, chosenSentence, audioFile, outputFilePath):
        return {'speakerId': speaker, 'generatedSentence': chosenSentence}
    return None


# In[13]:


def getRawEnglishModelNames():
    manager = ModelManager()
    return [model for model in manager.list_models() if "/en/" in model]


# In[14]:


def generateAudioInBatches(speakers, trainDF, batch_size=10):
    if trainDF.empty or not speakers:
        print("Empty dataset or no speakers provided. Exiting.")
        return

    englishModels = getRawEnglishModelNames()

    if not englishModels:
        print("No English models found. Exiting.")
        return

    for model in englishModels:
        print(f"Processing model: {model}")
        model_dir_name = model.replace("/", "_")
        outputCSVFile = f'../Data/ttsOutputs/{model_dir_name}_generatedSentences.csv'
        missing_speakers = list(speakers)
        if exists(outputCSVFile):
            outputDF = pd.read_csv(outputCSVFile)
            processed_speakers = set(outputDF['speakerId'])
            missing_speakers = list(set(speakers) - processed_speakers)  # <-- fix here
            if not missing_speakers:
                print(f"All speakers processed for model {model}. Skipping.")
                continue
            print(f'missing {len(missing_speakers)} speakers for model {model}')
        
        modelDirectory = f'../Data/ttsOutputs/{model_dir_name}'
        os.makedirs(modelDirectory, exist_ok=True)
        speakerSentencesPath = f'../Data/ttsOutputs/{model_dir_name}_generatedSentences.json'
        speakerSentences = loadGeneratedSentencesFromJson(speakerSentencesPath)

        for i in range(0, len(missing_speakers), batch_size):
            batch = missing_speakers[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}: {batch}")

            tts = generateTTS(model)
            if tts is None:
                print(f"Failed to generate TTS for model {model}. Skipping.")
                continue

            results = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(process_speaker, speaker, trainDF, tts, modelDirectory)
                    for speaker in batch
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
            # Write results after batch
            speakerSentences.extend(results)
            with open(speakerSentencesPath, 'w') as f:
                json.dump(speakerSentences, f)

            saveGeneratedSentences(speakerSentences, model_dir_name, modelDirectory)


# In[15]:


def main():
    trainDF = readCsv('train')
    speakers = getSpeakers(trainDF)
    if not exists('../Data/concatenatedInputs'):
        os.makedirs('../Data/concatenatedInputs')
        for speaker in speakers:
            speakerDF = getFilesBySpeaker(trainDF, speaker)
            concatenateAudio(speaker, speakerDF)
    if not exists('../Data/ttsOutputs/'):
        os.makedirs('../Data/ttsOutputs/')
    
    # Process speakers in batches
    generateAudioInBatches(speakers, trainDF, batch_size=10)
    
    print("done")
    
    


# In[16]:


if __name__ == '__main__':
    main()

