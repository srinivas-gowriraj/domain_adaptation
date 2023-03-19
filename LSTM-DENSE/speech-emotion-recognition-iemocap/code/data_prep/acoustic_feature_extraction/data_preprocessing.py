import re
import os
import librosa
import numpy as np
import pandas as pd
import argparse
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.style as ms
from tqdm import tqdm
import pickle
import random

import IPython.display
import librosa.display
import time
import joblib
from joblib import Parallel, delayed
import math

def read_labels(args):

    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
    start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []

    for sess in range(1, 6):
        emo_evaluation_dir = os.path.join(args.data_dir, 'Session{}/dialog/EmoEvaluation/'.format(sess))
        evaluation_files = [l for l in sorted(os.listdir(emo_evaluation_dir)) if 'Ses' in l]
        for file in evaluation_files:
            with open(emo_evaluation_dir + file, encoding="utf8", errors='ignore') as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)

    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['wav_file'] = wav_file_names
    df_iemocap['emotion'] = emotions
    df_iemocap['val'] = vals
    df_iemocap['act'] = acts
    df_iemocap['dom'] = doms

    print(df_iemocap.head())
    print(df_iemocap["emotion"].value_counts())

    df_iemocap.to_csv(args.output_dir + 'df_iemocap.csv', index=False)

def extract_features(args):

    labels_df = pd.read_csv(args.output_dir + 'df_iemocap.csv')
    sr = 22050


    audio_vectors = {}
    for sess in range(1, 6):
    # for sess in range(4, 5):
        print('Started Session {} Pickling'.format(sess))
        wav_file_path = '{}Session{}/dialog/wav/'.format(args.data_dir, sess)
        orig_wav_files = os.listdir(wav_file_path)
        for orig_wav_file in tqdm(orig_wav_files):
            try:
                #breakpoint()
                orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
                orig_wav_file, file_format = orig_wav_file.split('.')
                for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                    start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
                    start_frame = math.floor(start_time * sr)
                    end_frame = math.floor(end_time * sr)
                    truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                    audio_vectors[truncated_wav_file_name] = truncated_wav_vector
            except:
                #breakpoint()
                print('An exception occured for {}'.format(orig_wav_file))
        with open(args.output_dir + 'audio_vectors_{}.pkl'.format(sess), 'wb') as f:
            pickle.dump(audio_vectors, f)

    emotion_dict = {'ang': 0,
                'hap': 1,
                'exc': 2,
                'sad': 3,
                'fru': 4,
                'fea': 5,
                'sur': 6,
                'neu': 7,
                'xxx': 8,
                'oth': 8,
                'dis': 8}
    columns = ['wav_file', 'label', 'mfccs', 'spec_db']
    #df_features = pd.DataFrame(columns=columns, dtypes = ['str', 'int', 'float', 'float'])
    #df_features = pd.DataFrame({'wav_file': [], 'label': np.array([], dtype = int), 'mfccs': np.array([], dtype = np.float32), 'spec_db': np.array([], dtype = np.float32)})
    #df_features = np.array([[], [], [], []])
    # wav_file_array = np.array([])
    # label_array = np.array([], dtype = int)
    # mfccs_array = np.array([], dtype = np.float32)
    # spec_db_array = np.array([], dtype = np.float32)
    wav_file_array = []
    label_array = []
    mfccs_array = []
    spec_db_array = []

    audio_vectors_path= args.output_dir + 'audio_vectors_'

    for sess in range(1, 6):
        audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path,sess), 'rb'))
        for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains('Ses0{}'.format(sess))].iterrows()):

            wav_file_name = row['wav_file']
        
            label = emotion_dict[row['emotion']]
            y = audio_vectors[wav_file_name]

            #features_all = list(features(y, sr))
            # Extract the Mel-frequency cepstral coefficients (MFCC) features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # Extract the spectrogram
            spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

            # Convert power spectrogram to dB-scaled spectrogram
            spec_db = librosa.power_to_db(spec, ref=np.max)
            #breakpoint()

            features_all = [mfccs, spec_db]        
            feature_list = [wav_file_name, label] + features_all

            wav_file_array.append(wav_file_name)
            label_array.append(label)
            mfccs_array.append(np.array(mfccs))
            
            spec_db_array.append(np.array(spec_db))
            #breakpoint()

            
            #df_features = df_features.append(pd.DataFrame(feature_list, index=columns).transpose(), ignore_index=True)
            #print(df_features.head())
            #breakpoint()
            #sys.exit()
            #break
            
        
        print("Session Finished {}".format(sess))
    #breakpoint()
    #df_features = pd.DataFrame({'wav_file': wav_file_array, 'label': label_array, 'mfccs': mfccs_array, 'spec_db': spec_db_array})
    #breakpoint()
    '''with open(args.output_dir + 'df_features.txt', 'w+') as f:
        f.write('wav_file,label,mfccs,spec_db'+ "\n")
         
        for i in range(len(wav_file_array)):
            f.write('{}, {}, {}, {}n'.format(wav_file_array[i], label_array[i], mfccs_array[i], spec_db_array[i])+ "\n")'''
    output_dictionary = {'wav_file': wav_file_array, 'label': label_array, 'mfccs': mfccs_array, 'spec_db': spec_db_array}
    with open(args.output_dir + 'feature_vectors.pkl', 'wb') as f:
        pickle.dump(output_dictionary, f)
    
    #breakpoint()

        
    
    #df_features.to_csv(args.output_dir + 'df_features.csv', index=False)

def read_df_features(args):
    #df_features = pd.read_csv(args.output_dir + 'df_features.csv')
    #df_features = np.loadtxt(args.output_dir + 'df_features.txt', delimiter=',', skiprows=1, dtype=str)
    audio_vectors = pickle.load(open((args.output_dir+'feature_vectors.pkl'), 'rb'))
    breakpoint()
    #test = np.array(df_features['spec_db'], dtype=float)
    #print(test.shape)
    '''print(test.dtype)
    breakpoint()'''
    print(df_features.head())
    print(df_features.columns)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/sgowrira/domain_adaptation/IEMOCAP_full_release/')
    parser.add_argument('--output_dir', type=str, default='/home/sgowrira/domain_adaptation/LSTM-DENSE/speech-emotion-recognition-iemocap/preprocess_info/')
    args = parser.parse_args()
    #read_labels(args)
    extract_features(args)
    read_df_features(args)
    