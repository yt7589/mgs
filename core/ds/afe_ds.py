#
import csv
from pyAudioAnalysis import ShortTermFeatures as aSF
from pyAudioAnalysis import MidTermFeatures as aMF
from pyAudioAnalysis import audioBasicIO as aIO 
import numpy as np 
import plotly.graph_objs as go 
import plotly

class AfeDs:
    @staticmethod
    def load_ds(class_num, base_folder, csv_file):
        '''
        载入数据集
        @param class_num 类别数
        @param base_folder 数据集根目录，绝对路径或相对app_main.py路径
        @param csv_file 文件名,类别编号（从0开始）
        '''
        X_raw = [[]]
        y_raw = []
        m = 0
        dim = 2
        with open(csv_file, newline='', encoding='utf-8') as fd:
            rows = csv.reader(fd, delimiter=',', quotechar='|')
            for row in rows:
                feats = AfeDs.extract_afs('{0}{1}'.format(base_folder, row[0]))
                X_raw = np.append(X_raw, feats)
                y_raw = np.append(y_raw, int(row[1]))
                m += 1
        X = np.array(X_raw)
        X = np.reshape(X, (m, dim))
        y = np.array(y_raw)
        return X, y

    @staticmethod
    def extract_afs(wav_file):
        fs, s = aIO.read_audio_file(wav_file)
        mt, st, mt_n = aMF.mid_feature_extraction(s, fs, 1 * fs, 1 * fs, 
                                         0.05 * fs, 0.05 * fs)
        '''
        print(f'signal duration {len(s)/fs} seconds')
        print(f'{st.shape[1]} {st.shape[0]}-D short-term feature vectors extracted')
        print(f'{mt.shape[1]} {mt.shape[0]}-D segment feature statistic vectors extracted')
        print('mid-term feature names')
        for i, mi in enumerate(mt_n):
            print(f'{i}:{mi}')
        '''
        mtf = np.mean(mt, axis=1)
        feats = np.array([
            mtf[mt_n.index('spectral_centroid_mean')],
            mtf[mt_n.index('energy_entropy_mean')]
        ])
        return feats
