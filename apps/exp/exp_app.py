#
import subprocess as spc

class ExpApp:
    def __init__(self):
        self.refl = 'apps.exp.ExpApp'
    
    def startup(self, args={}):
        print('音频试验程序 v0.0.1')
        p = spc.Popen("d:/software/ffmpeg/ffmpeg.exe", shell=True, stdout=spc.PIPE, universal_newlines=True)
        p.wait()
        result_lines = p.stdout.readlines()
        for line in result_lines:
            print(line.strip())