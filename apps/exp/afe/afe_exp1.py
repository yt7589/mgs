#
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO 
import numpy as np 
import plotly.graph_objs as go 
import plotly
#import IPython

class AfeExp1:
    @staticmethod
    def exp1():
        wav_file = 'E:/temp/pyAudioAnalysis/pyAudioAnalysis/data/scottish.wav'
        fs, s = aIO.read_audio_file(wav_file)
        #IPython.display.display(IPython.display.Audio(wav_file))
        duration = len(s) / float(fs)
        print(f'duration = {duration} seconds')
        win, step = 0.050, 0.050
        [f, fn] = aF.feature_extraction(s, fs, int(fs * win), 
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