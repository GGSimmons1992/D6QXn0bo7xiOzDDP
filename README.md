# Voice Cloning and Fake Audio Detection
The goal of this project is to create 2 models, one is for Voice Cloning (VC), and the other is for Fake Audio Detection (FAD). 

## Methodology
- ⁠concatenate sentences of each speaker, so they are long enough to be inputs
- ⁠Loop through English TTS models and compare word error rates. Use the model with the lowest word error rate for part 2
- Use the selected model to generate sentences for training and testing for FAD.
- ⁠Balance train and test sets with real sentences (1) and generated sentences (0).
- ⁠Use MFCC on real and generated audio for X inputs
- ⁠Train and tune a sequential model for FAD.
