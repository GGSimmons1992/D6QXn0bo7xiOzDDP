#!/usr/bin/env python
# coding: utf-8

# # Create MFCC Matrices
# This notebook cell provides a function to generate MFCC tensors (X) and labels (y) from the CSVs `Data/fad_train.csv` and `Data/fad_test.csv`. The outputs are saved under `Data/mfcc/` as compressed NumPy archives (`.npz`).
# 
# Notes: Uses `librosa` to load audio and compute MFCCs. Each audio file is converted into a fixed-length frame sequence by padding/truncating to `max_len` frames so the result is an array shaped `(N, max_len, n_mfcc)`. The `isFake` column is used as `y`.

# In[1]:


# Imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import librosa
except Exception as e:
    raise ImportError('librosa is required. Install with `pip install librosa soundfile`') from e


# In[2]:


def _resolve_path(csv_dir, file_entry):
    # If the path is absolute, return as-is. Otherwise resolve relative to the CSV's directory
    if os.path.isabs(file_entry):
        return file_entry
    return os.path.normpath(os.path.join(csv_dir, file_entry))


# In[3]:


def _process_csv(csv_path, n_mfcc, max_len, sr):
    df = pd.read_csv(csv_path)
    X_list = []
    y_list = []
    csv_dir = os.path.dirname(csv_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Process {os.path.basename(csv_path)}'):
        file_entry = row.get('file') or row.get('filename') or row.get('path')
        if pd.isna(file_entry):
            continue
        audio_path = _resolve_path(csv_dir, str(file_entry))
        if not os.path.exists(audio_path):
            # warn and skip missing files
            print(f'Warning: file not found {audio_path}; skipping')
            continue
        try:
            y, _ = librosa.load(audio_path, sr=sr, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # shape (n_mfcc, frames)
            mfcc = mfcc.T  # (frames, n_mfcc)
            frames = mfcc.shape[0]
            if frames >= max_len:
                mfcc_fixed = mfcc[:max_len, :]
            else:
                pad = np.zeros((max_len - frames, n_mfcc), dtype=mfcc.dtype)
                mfcc_fixed = np.vstack([mfcc, pad])
            X_list.append(mfcc_fixed.astype(np.float32))
            # isFake column may be boolean or string; coerce to int (1 fake, 0 real)
            is_fake = row.get('isFake')
            if pd.isna(is_fake):
                y_list.append(0)
            else:
                # handle boolean/string/1/0
                if isinstance(is_fake, str):
                    val = is_fake.lower() in ['true', '1', 't', 'yes']
                else:
                    val = bool(is_fake)
                y_list.append(int(val))
        except Exception as e:
            print(f'Error processing {audio_path}: {e}; skipping')
            continue
    if len(X_list) == 0:
        return np.zeros((0, max_len, n_mfcc), dtype=np.float32), np.array([], dtype=np.int64)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# In[4]:


def createMFCCMatricies(train_csv='../Data/fad_train.csv', test_csv='../Data/fad_test.csv', output_dir='../Data/mfcc', n_mfcc=40, max_len=200, sr=22050):
    """
    Generate MFCC tensors for train and test CSVs and save them under `output_dir`.

    Parameters:
    - train_csv/test_csv: paths to the CSV files (relative paths are supported).
    - output_dir: where to save `mfcc_train.npz` and `mfcc_test.npz`.
    - n_mfcc: number of MFCC coefficients per frame.
    - max_len: fixed number of frames per example (pad/truncate).
    - sr: sampling rate for audio loading.
    Returns: tuple of saved file paths (train_path, test_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    print('Processing train CSV...')
    X_train, y_train = _process_csv(train_csv, n_mfcc=n_mfcc, max_len=max_len, sr=sr)
    train_out = os.path.join(output_dir, 'mfcc_train.npz')
    np.savez_compressed(train_out, X=X_train, y=y_train)
    print(f'Saved train MFCCs: {train_out} -> X shape {X_train.shape}, y shape {y_train.shape}')

    print('Processing test CSV...')
    X_test, y_test = _process_csv(test_csv, n_mfcc=n_mfcc, max_len=max_len, sr=sr)
    test_out = os.path.join(output_dir, 'mfcc_test.npz')
    np.savez_compressed(test_out, X=X_test, y=y_test)
    print(f'Saved test MFCCs: {test_out} -> X shape {X_test.shape}, y shape {y_test.shape}')

    return train_out, test_out


# In[5]:


def main():
    # Default run: write MFCCs into Data/mfcc/ with 40 coefficients and 200 frames per example
    out_train, out_test = createMFCCMatricies()
    print('Completed. Files:', out_train, out_test)


# In[6]:


if __name__ == "__main__":
    main()

