# Voice Cloning and Fake Audio Detection
The goal of this project is to create 2 models, one is for Voice Cloning (VC), and the other is for Fake Audio Detection (FAD). Sequential model chosen has 4 Convolution-Pool layers followed by 16 Dense layers.

## Methodology
### Part 1: Voice Cloning
- ⁠concatenate sentences of each speaker, so they are long enough to be inputs (Vocoder.ipynb)
- ⁠Loop through English TTS models (Vocoder.ipynb) and compare word error rates. Use the model with the lowest word error rate for part 2 (EvaluateVoices.ipynb).
  - English models include: 
    - tts_models/en/ljspeech/vits--neon
    - tts_models/en/ljspeech/vits
    - tts_models/en/ljspeech/fast_pitch
    - tts_models/en/jenny/jenny
    - tts_models/en/ljspeech/overflow
    - tts_models/en/ljspeech/glow-tts
    - tts_models/en/ljspeech/neural_hmm
    - tts_models/en/ljspeech/speedy-speech
    - tts_models/en/ljspeech/tacotron2-DCA
    - tts_models/en/ljspeech/tacotron2-DDC_ph
    - tts_models/en/sam/tacotron-DDC
    - tts_models/en/ek1/tacotron2
    - tts_models/en/blizzard2013/capacitron-t2-c50
    - tts_models/en/ljspeech/tacotron2-DDC
    - tts_models/en/blizzard2013/capacitron-t2-c150_v2

### Part 2: Fake Audio Detection
- Use the selected model to generate sentences for training and testing for FAD.(PrepareFADData.ipynb)
- ⁠Balance train and test sets with real sentences (1) and generated sentences (0). (PrepareFADData.ipynb)
- ⁠Use MFCC on real and generated audio for X matrix inputs (CreateMFCCs.ipynb)
- ⁠Train and tune a sequential model for FAD. (DetectVoices.ipynb)
- After creating the best performing model, create a Gradio interface for uploading audio clips and receiving detection results. (predict.py)

## Repo Contents

### Data (not included due to .gitignore)

Data is from the TIMIT Corpus. The TIMIT corpus of read speech is designed to provide speech data for acoustic-phonetic studies and for the development and evaluation of automatic speech recognition systems. TIMIT contains a total of 6300 sentences, 10 sentences spoken by each of 630 speakers from 8 major dialect regions of the United States.

[Corpus Creation Repo Link](https://github.com/philipperemy/timit)

[Corpus Dataset on Kaggle Link](https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech)

### Models

The models are stored in the Models directory, which contains the following files:

#### DetectVoicesModelTrials

This directory contains the trial runs of the Fake Audio Detection model, which were used for hyperparameter tuning and model selection.

#### best_dv_model_.h5

This file contains the best performing Fake Audio Detection model, which was selected based on its performance on the validation set.

#### best_dv_model_params.json

This file contains the hyperparameters of the best performing Fake Audio Detection model, which were used for training and tuning the model.

### Notebooks (in order of creation and execution)

1. Vocoder.ipynb

This notebook contains the code for generating sentences using the selected TTS model and evaluating the generated audio using word error rates.

2. EvaluateVoices.ipynb

This notebook contains the code for evaluating different English TTS models and selecting the best performing model based on word error rates.

3. PrepareFADData.ipynb

This notebook contains the code for preparing the training and testing data for the Fake Audio Detection model, including balancing the dataset with real and generated sentences.

4. CreateMFCCs.ipynb

This notebook contains the code for creating Mel-Frequency Cepstral Coefficients (MFCCs) from the audio data, which are used as input features for the Fake Audio Detection model.

5. DetectVoices.ipynb

This notebook contains the code for training and tuning the Fake Audio Detection model, as well as evaluating its performance on the test set.

### Src (in order of creation and execution)

#### generateVoices.py

This script contains the code for generating sentences using the selected TTS model.

#### evaluateVoices.py

This script contains the code for evaluating different English TTS models and selecting the best performing model based on word error rates.

#### prepareFADData.py

This script contains the code for preparing the training and testing data for the Fake Audio Detection model, including balancing the dataset with real and generated sentences.

#### createMFCCs.py

This script contains the code for creating Mel-Frequency Cepstral Coefficients (MFCCs) from the audio data, which are used as input features for the Fake Audio Detection model.

#### trainModel.py

This script contains the code for training and tuning the Fake Audio Detection model, as well as evaluating its performance on the test set.

#### predict.py

This script contains the code for making predictions using the trained Fake Audio Detection model. It includes a Gradio interface for uploading audio clips and receiving detection results. Run this script to launch the Gradio app and test the model with new audio clips.

### License

This project is licensed under the MIT License. See the LICENSE file for more details.

### requirements.txt

This file contains the list of Python packages required to run the code in this repo. You can install the required packages using pip:

```bash
pip install -r requirements.txt
```
