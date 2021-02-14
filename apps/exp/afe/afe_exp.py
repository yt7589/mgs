#
from pyAudioAnalysis import ShortTermFeatures as aSF
from pyAudioAnalysis import MidTermFeatures as aMF
from pyAudioAnalysis import audioBasicIO as aIO 
import numpy as np 
import plotly.graph_objs as go 
import plotly
#import IPython

class AfeExp:
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
        pass