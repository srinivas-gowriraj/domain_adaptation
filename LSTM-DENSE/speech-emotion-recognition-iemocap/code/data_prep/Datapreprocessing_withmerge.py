import re
import os
import librosa
import numpy as np
import pandas as pd
import argparse
#import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.style as ms
from tqdm import tqdm
import pickle
import random
from collections import Counter

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

    # labels_df = pd.read_csv(args.output_dir + 'df_iemocap.csv')
    labels_df = os.listdir(args.data_dir)
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

            #features_all = list(dataset(y, sr))
            # Extract the Mel-frequency cepstral coefficients (MFCC) dataset
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
         
            
        
        print("Session Finished {}".format(sess))

    output_dictionary = {'wav_file': wav_file_array, 'label': label_array, 'mfccs': mfccs_array, 'spec_db': spec_db_array}
    with open(args.output_dir + 'feature_vectors.pkl', 'wb') as f:
        pickle.dump(output_dictionary, f)
   
def read_df_features(args):
    audio_vectors = pickle.load(open((args.output_dir+'feature_vectors.pkl'), 'rb'))

def createImageData(args):
    dataset = pickle.load(open((args.output_dir+'feature_vectors.pkl'), 'rb'))
    #filter the dataset with label values between 0 and 3
    spectograms = []
    labels = []
    filenames = []
    for i in range(len(dataset['label'])):
        if dataset['label'][i] in [0,1,3,7]:
            spectograms.append(dataset['spec_db'][i])
            labels.append(dataset['label'][i])
            filenames.append(dataset['wav_file'][i])

    emotion_full_dict = {0:'angry', 1:'happiness', 3:'excited', 7:'neutral'}
    # iterate over the spectograms and convert them to images
    sr = 22050
    for i in range(len(spectograms)):
        spectogram = spectograms[i]
        librosa.display.specshow(spectogram, y_axis='mel', sr=sr, x_axis='time')
        fig1 = plt.gcf()
        plt.axis('off')
        plt.draw()
        save_dir = args.output_dir+'images_new/'+emotion_full_dict[labels[i]]+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir+filenames[i]+'.jpg'
        fig1.savefig(save_path, dpi=50)

def domain_adapted_data_Preprocess(args, label):
    audio_vectors = {}
    orig_wav_files = os.listdir(args.data_dir)
    filename=[]
    audiovec=[]
    mfcc=[]
    lab=[]
    specs=[]
    for i in orig_wav_files:
        
        wavpath = os.path.join(args.data_dir, i)
        orig_wav_vector, _sr = librosa.load(wavpath, sr=22050)
        orig_wav_file, file_format = wavpath.split('.')
        print(orig_wav_file)
        
        #audio_vectors[orig_wav_file] = orig_wav_vector
        mfccs = librosa.feature.mfcc(y = orig_wav_vector, sr=22050, n_mfcc=13)
        spec = librosa.feature.melspectrogram(y = orig_wav_vector, sr=22050, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        filename.append(wavpath)
        mfcc.append(np.array(mfccs))
        audiovec.append(orig_wav_vector)
        #lab.append(label)
        if "F" in orig_wav_file:
            lab.append(1)
        if "W" in orig_wav_file:
            lab.append(0)
        if "T" in orig_wav_file:
            lab.append(3)
        if "N" in orig_wav_file:
            lab.append(7)
            print("something working")
        
        specs.append(np.array(spec_db))

    final_dictionary = {'wav_file': filename, 'label': lab, 'mfccs': mfcc, 'spec_db': specs}
    with open(args.output_dir + 'feature_vectors_emodb.pkl', 'wb') as f:
        pickle.dump(final_dictionary, f)

def combine(args):
    data_angry= pd.read_pickle(args.output_dir + 'feature_vectors_Angry.pkl')
    data_happy = pd.read_pickle(args.output_dir + 'feature_vectors_Happy.pkl')
    data_sad= pd.read_pickle(args.output_dir + 'feature_vectors_Sad.pkl')
    data_neutral = pd.read_pickle(args.output_dir + 'feature_vectors_Neutral.pkl')

    all_labels=[]
    k=0
    for i in data_neutral['label']:
        all_labels.append(i)
        k=k+1
        if k==100:
            break
    for i in data_angry['label']:
        all_labels.append(i)
    for i in data_happy['label']:
        all_labels.append(i)
    for i in data_sad['label']:
        all_labels.append(i)


    all_spec_db_images=[]
    k=0
    for i in data_neutral['spec_db']:
        all_spec_db_images.append(i)
        k=k+1
        if k==100:
            break
    for i in data_angry['spec_db']:
        all_spec_db_images.append(i)
    for i in data_happy['spec_db']:
        all_spec_db_images.append(i)
    for i in data_sad['spec_db']:
        all_spec_db_images.append(i)
    

    
    all_wav_files=[]
    k=0
    for i in data_neutral['wav_file']:
        all_wav_files.append(i)
        k=k+1
        if k==100:
            break
    for i in data_angry['wav_file']:
        all_wav_files.append(i)
    for i in data_happy['wav_file']:
        all_wav_files.append(i)
    for i in data_sad['wav_file']:
        all_wav_files.append(i)
    

    

    all_mfccs=[]
    k=0
    for i in data_neutral['mfccs']:
        all_mfccs.append(i)
        k=k+1
        if k==100:
            break
    for i in data_angry['mfccs']:
        all_mfccs.append(i)
    for i in data_happy['mfccs']:
        all_mfccs.append(i)
    for i in data_sad['mfccs']:
        all_mfccs.append(i)


    final_dictionary = {'wav_file': all_wav_files, 'label': all_labels, 'mfccs': all_mfccs, 'spec_db': all_spec_db_images}
    with open(args.output_dir + 'NSC_datasetset_balaanced.pkl', 'wb') as f:
        pickle.dump(final_dictionary, f)


def test_pickle(path):
    #dataset=np.load(path)
    dataset = pickle.load(open(path,'rb'))
    a=dataset['label']
    print(dataset['mfccs'][0])
    print(Counter(a))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    import sys
    parser.add_argument('--output_dir', type=str, default='LSTM-DENSE/speech-emotion-recognition-iemocap/preprocess_info')
    sys.path.append("wav")
    parser.add_argument('--data_dir', type=str, default='wav')
    args = parser.parse_args()
    #read_labels(args)
    #extract_features(args)
    #read_df_features(args)
    #createImageData(args)
    #domain_adapted_data_Preprocess(args, label=7)
    # combine(args)
    
    test_pickle("LSTM-DENSE/speech-emotion-recognition-iemocap/preprocess_infofeature_vectors_emodb.pkl")
    



    