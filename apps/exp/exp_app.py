#
import subprocess as spc
from apps.exp.afe.afe_exp import AfeExp

class ExpApp:
    def __init__(self):
        self.refl = 'apps.exp.ExpApp'
    
    def startup(self, args={}):
        print('音频试验程序 v0.0.1')
        #AfeExp.exp1()
        #AfeExp.exp2()
        #AfeExp.exp3()
        AfeExp.exp4()

    def convert_mp3_wav(mp3_file, wav_file):
        '''
        使用ffmpeg将mp3文件转换为wav文件
        '''
        cmd = 'd:/software/ffmpeg/ffmpeg.exe -i {0} {1}'.format(mp3_file, wav_file)
        p = spc.Popen(cmd, shell=True, stdout=spc.PIPE, universal_newlines=True)
        p.wait()
        result_lines = p.stdout.readlines()
        for line in result_lines:
            print(line.strip())