#
from pyAudioAnalysis import ShortTermFeatures as aSF
from pyAudioAnalysis import MidTermFeatures as aMF
from pyAudioAnalysis import audioBasicIO as aIO 
import numpy as np 
import plotly.graph_objs as go 
import plotly
import sklearn.svm as sks # import SVC
#import IPython
from core.ds.afe_ds import AfeDs

class AfeExp:
    @staticmethod
    def test_afe_ds():
        class_num = 2
        base_folder = './deps/pyAudioAnalysis/pyAudioAnalysis/data/music/'
        csv_file = './datas/acds.csv'
        X, y = AfeDs.load_ds(class_num=class_num, base_folder=base_folder, csv_file=csv_file)
        cl = sks.SVC(kernel='rbf', C=20) 
        cl.fit(X, y)
        X_t = np.array([X[0], X[15], X[2], X[11], X[3]])
        print('X_t: {0};'.format(X_t.shape))
        rst = cl.predict(X_t)
        print(rst)


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

    def exp5():
        print('pyAudioAnalysis example 5')
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

        print('f1 type:{0}; shape:{1}; value:{2};'.format(type(f1), f1.shape, f1))
        print('f2 type:{0}; shape:{1}; value:{2};'.format(type(f2), f2.shape, f2))

        y = np.concatenate((np.zeros(f1.shape[1]), np.ones(f2.shape[1]))) 
        f = np.concatenate((f1.T, f2.T), axis = 0)
        print('y: {0}; {1};'.format(y.shape, y))
        print('X: {0}; {1};'.format(f.shape, f))
        # train the svm classifier
        cl = sks.SVC(kernel='rbf', C=20) 
        cl.fit(f, y)


        p1 = go.Scatter(x=f1[0, :],  y=f1[1, :], name=class_names[0],
                        marker=dict(size=10,color='rgba(255, 182, 193, .9)'),
                        mode='markers')
        p2 = go.Scatter(x=f2[0, :], y=f2[1, :],  name=class_names[1], 
                        marker=dict(size=10,color='rgba(100, 100, 220, .9)'),
                        mode='markers')
        mylayout = go.Layout(xaxis=dict(title="spectral_centroid_mean"),
                            yaxis=dict(title="energy_entropy_mean"))

        # apply the trained model on the points of a grid
        x_ = np.arange(f[:, 0].min(), f[:, 0].max(), 0.002) 
        y_ = np.arange(f[:, 1].min(), f[:, 1].max(), 0.002) 
        xx, yy = np.meshgrid(x_, y_) 
        X_t = np.c_[xx.ravel(), yy.ravel()]
        print('X_t: {0};'.format(X_t.shape))
        Z = cl.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) / 2 
        # and visualize the grid on the same plot (decision surfaces)
        cs = go.Heatmap(x=x_, y=y_, z=Z, showscale=False, 
                    colorscale= [[0, 'rgba(255, 182, 193, .3)'], 
                                [1, 'rgba(100, 100, 220, .3)']]) 
        mylayout = go.Layout(xaxis=dict(title="spectral_centroid_mean"),
                            yaxis=dict(title="energy_entropy_mean"))
        #plotly.offline.iplot(go.Figure(data=[p1, p2, cs], layout=mylayout))
        plotly.offline.plot({
            'data': [p1, p2, cs],
            'layout': mylayout
        }, auto_open=True)