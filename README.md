# Voice Cloning and Fake Audio Detection
The goal of this project is to create 2 models, one is for Voice Cloning (VC), and the other is for Fake Audio Detection (FAD). 

## Methodology
- ⁠concatenate sentences of each speaker, so they are long enough to be inputs
- ⁠Loop through English TTS models and compare word error rates. Use the model with the lowest word error rate for part 2
- Use the selected model to generate sentences for training and testing for FAD.
- ⁠Balance train and test sets with real sentences (1) and generated sentences (0).
- ⁠Use MFCC on real and generated audio for X inputs
- ⁠Train and tune a sequential model for FAD.

## Repo Contents

### Data (not included due to .gitignore)

Data is from the VCTK Corpus, which contains speech data from 109 native English speakers with various accents. Each speaker reads out approximately 400 sentences. The dataset is commonly used for voice cloning and speech synthesis research.

This repo also contains generated audio data from the best performing TTS model, which is used for training and testing the Fake Audio Detection model.

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
