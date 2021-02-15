# mgs
#from apps.exp.exp_app import ExpApp
from apps.mgc.mgc_app import MgcApp

def main(args={}):
    print('音乐生成系统 v0.0.1')
    #app = ExpApp()
    app = MgcApp()
    app.startup(args)

if '__main__' == __name__:
    main()
