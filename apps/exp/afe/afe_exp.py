#
from pyAudioAnalysis import ShortTermFeatures as aSF
from pyAudioAnalysis import MidTermFeatures as aMF
from pyAudioAnalysis import audioBasicIO as aIO 
import numpy as np 
import plotly.graph_objs as go 
import plotly
#import IPython

class AfeExp:
    data_folder = './deps/pyAudioAnalysis/pyAudioAnalysis/data/'
    wav_file = './deps/pyAudioAnalysis/pyAudioAnalysis/data/scottish.wav'

    @staticmethod
    def exp1():
        fs, s = aIO.read_audio_file(AfeExp.wav_file)
        #IPython.display.display(IPython.display.Audio(wav_file))
        duration = len(s) / float(fs)
        print(f'duration = {duration} seconds')
        win, step = 0.050, 0.050
        [f, fn] = aSF.feature_extraction(s, fs, int(fs * win), 
                                        int(fs * step))
        print(f'{f.shape[1]} frames, {f.shape[0]} short-term features')
        print('Feature names:')
        for i, nam in enumerate(fn):
            print(f'{i}:{nam}')
        time = np.arange(0, duration - step, win) 
        energy = f[fn.index('energy'), :]
        mylayout = go.Layout(yaxis=dict(title="frame energy value"),
                            xaxis=dict(title="time (sec)"))
        '''
        plotly.offline.iplot(go.Figure(data=[go.Scatter(x=time, 
                                                        y=energy)], 
                                    layout=mylayout))
        '''
        plotly.offline.plot({
            'data': [go.Scatter(x=time, y=energy)],
            'layout': mylayout
        }, auto_open=True)

    def exp2():
        fs, s = aIO.read_audio_file(AfeExp.wav_file)
        duration = len(s) / float(fs)
        print(f'duration = {duration} seconds')
        win, step = 0.050, 0.050
        [f, fn] = aSF.feature_extraction(s, fs, int(fs * win), 
                                        int(fs * step))
        print(f'{f.shape[1]} frames, {f.shape[0]} short-term features')
        time = np.arange(0, duration - step, win) 
        energy = f[fn.index('spectral_centroid'), :]
        mylayout = go.Layout(yaxis=dict(title="spectral_centroid value"),
                            xaxis=dict(title="time (sec)"))
        '''
        plotly.offline.iplot(go.Figure(data=[go.Scatter(x=time, 
                                                        y=energy)], 
                                    layout=mylayout))
        '''
        plotly.offline.plot({
            'data': [go.Scatter(x=time, y=energy)],
            'layout': mylayout
        }, auto_open=True)

    def exp3():
        fs, s = aIO.read_audio_file(AfeExp.wav_file)
        mt, st, mt_n = aMF.mid_feature_extraction(s, fs, 1 * fs, 1 * fs, 
                                         0.05 * fs, 0.05 * fs)
        print(f'signal duration {len(s)/fs} seconds')
        print(f'{st.shape[1]} {st.shape[0]}-D short-term feature vectors extracted')
        print(f'{mt.shape[1]} {mt.shape[0]}-D segment feature statistic vectors extracted')
        print('mid-term feature names')
        for i, mi in enumerate(mt_n):
            print(f'{i}:{mi}')

    def exp4():
        print('pyAudioAnalysis example 4')
        dirs = [
            '{0}music/classical'.format(AfeExp.data_folder), 
            '{0}music/metal'.format(AfeExp.data_folder)] 
        class_names = ['classical', 'metal']
        m_win, m_step, s_win, s_step = 1, 1, 0.1, 0.05 
        features = [] 
        for d in dirs: # get feature matrix for each directory (class) 
            f, files, fn = aMF.directory_feature_extraction(d, m_win, m_step, 
                                                        s_win, s_step) 
            features.append(f)
        print(features[0].shape, features[1].shape)
        f1 = np.array([features[0][:, fn.index('spectral_centroid_mean')],
                    features[0][:, fn.index('energy_entropy_mean')]])
        f2 = np.array([features[1][:, fn.index('spectral_centroid_mean')],
                    features[1][:, fn.index('energy_entropy_mean')]])
        plots = [go.Scatter(x=f1[0, :],  y=f1[1, :], 
                            name=class_names[0], mode='markers'),
                go.Scatter(x=f2[0, :], y=f2[1, :], 
                            name=class_names[1], mode='markers')]
        mylayout = go.Layout(xaxis=dict(title="spectral_centroid_mean"),
                            yaxis=dict(title="energy_entropy_mean"))
        #plotly.offline.iplot(go.Figure(data=plots, layout=mylayout))
        plotly.offline.plot({
            'data': plots,
            'layout': mylayout
        }, auto_open=True)