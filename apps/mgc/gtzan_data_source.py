#
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import sklearn.model_selection as skms # import train_test_split

class GtzanDataSource:
    def __init__(self):
        self.refl = 'apps.mgc.GtzanDs'
        self.ds_folder = './datas/gtzan/genres_original/'
        self.json_folder = './work/gtzan.json'
        self.sample_rate = 22050
        self.duration = 30
        self.samples_per_track = self.sample_rate * self.duration

    def load_ds(self):
        #self.show_wav_file()
        #self.generate_mfcc_feats(self.ds_folder, self.json_folder, num_segments=10)
        inputs, targets = self.load_from_json(self.json_folder)
        v = min(np.unique(targets))
        for i in range(len(targets)):
            if targets[i] == v:
                targets[i] = 0
            else:
                new_val = targets[i] - v
                targets[i] = new_val
        # If you want to apply Convolution NN the remove the comment from the below line
        #inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], inputs.shape[2], 1))
        inputs_train, inputs_test, targets_train, targets_test = \
                    skms.train_test_split(inputs, targets, test_size=0.25)
        # Adding Noise 
        for i in range(inputs_train.shape[0]):
            s = np.random.rand(inputs_train.shape[1], inputs_train.shape[2])
            inputs_train[i] = inputs_train[i] + s
        return inputs_train, targets_train, inputs_test, targets_test

    def load_from_json(self, json_folder):
        with open(json_folder, 'r') as fp:
            data = json.load(fp)
        #Convert lists into numpy arrays
        inputs = data['mfcc']
        targets = data['labels'] 
        return np.array(inputs, dtype=np.float32), np.array(targets)

    def show_wav_file(self):
        wav_file = '{0}blues/blues.00000.wav'.format(self.ds_folder)
        signal, sr = librosa.load(wav_file, sr=22050)
        print('length of signal: {0};'.format(len(signal)))
        print('sample rate: {0};'.format(sr))
        print('duration of the file: {0}s;'.format(len(signal)/sr))
        librosa.display.waveplot(signal)
        plt.show()

    def generate_mfcc_feats(self, dataset_path, json_path, 
                n_mfcc=13, n_fft=4084, hop_length=1024, 
                num_segments=10):
        data = {
            'mapping' : [],
            'mfcc' : [],
            'labels' : []
        }
        count = 0
        num_samples_per_segment = int(self.samples_per_track / num_segments) 
        expected_num_mfcc_vectors_per_segment = math.ceil(
                    num_samples_per_segment / hop_length)
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
            if dirpath not in dataset_path:
                dirpath_components = dirpath.split('/')
                semantic_label = dirpath_components[-1]
                data['mapping'].append(semantic_label)
                print('\nProcessing {}'.format(semantic_label))
                for f in filenames:
                    if f.endswith('.wav') and f != 'jazz.00054.wav':
                        file_path = os.path.join(dirpath,f)
                        #print(file_path)
                        signal, sr = sf.read(file_path)
                        for s in range(num_segments):
                            start_sample = num_samples_per_segment * s  
                            finish_sample = num_samples_per_segment + start_sample
                            mfcc = librosa.feature.mfcc(signal[start_sample : finish_sample],
                                                   sr = sr,
                                                   n_fft = n_fft,
                                                   n_mfcc = n_mfcc,
                                                   hop_length = hop_length)
                            mfcc = mfcc.T
                            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                                print(mfcc.shape)
                                data['mfcc'].append(mfcc.tolist())
                                data['labels'].append(i)
                                print('Processing {}, segment:{}'.format(file_path, s))
                                count += 1
                                print(count)
        with open(json_path, 'w') as fp:
            json.dump(data, fp, indent=4)