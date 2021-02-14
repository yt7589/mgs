#
#import scipy
#from scipy import io as sio
import scipy.io.wavfile
from ext.spafe.utils import vis
from ext.spafe.features.bfcc import bfcc

class AfeBfcc:
    @staticmethod
    def extract_bfcc(wav_file):
        print('获取BFCC特征')
        num_ceps = 13
        low_freq = 0
        high_freq = 2000
        nfilts = 24
        nfft = 512
        dct_type = 2,
        use_energy = False,
        lifter = 5
        normalize = False
            
        # read wav 
        fs, sig_raw = scipy.io.wavfile.read(wav_file)
        sig = sig_raw #[:, :1] #.reshape((sig_raw.shape[0],))
        print('fs: {0}\r\n{1}\r\n***********'.format(type(fs), fs))
        print('sig: {0}\r\n{1}\r\n******************'.format(sig.shape, sig))
        # compute features
        bfccs = bfcc(sig=sig,
                    fs=fs,
                    num_ceps=num_ceps,
                    nfilts=nfilts,
                    nfft=nfft,
                    low_freq=low_freq,
                    high_freq=high_freq,
                    dct_type=dct_type,
                    use_energy=use_energy,
                    lifter=lifter,
                    normalize=normalize)
        print('step 1')
        # visualize spectogram
        vis.spectogram(sig, fs)
        print('step 2')
        # visualize features
        vis.visualize_features(bfccs, 'BFCC Index', 'Frame Index')
        print('step 3')