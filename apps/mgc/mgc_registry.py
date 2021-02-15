#
from apps.mgc.mgc_const import GruConst

class MgcRegistry:
    instance = None

    def __init__(self):
        self.params = {}

    def put(self, key, val):
        self.params[key] = val

    def get(self, key):
        return self.params[key]

    @staticmethod
    def get_instance():
        if MgcRegistry.instance is None:
            MgcRegistry.instance = MgcRegistry()
        return MgcRegistry.instance

    # 定义相关参数
    # GRU1相关参数定义
    GRU1 = {
        GruConst.L: 3,
        GruConst.N: 6,
        GruConst.H_in: 65*13,
        GruConst.num_directions: 1,
        GruConst.hidden_size: 100,
        GruConst.num_layers: 1
    }
    GRU2 = {
        GruConst.L: 3,
        GruConst.N: 6,
        GruConst.H_in: 100,
        GruConst.num_directions: 1,
        GruConst.hidden_size: 500,
        GruConst.num_layers: 1
    }
    GRU3 = {
        GruConst.L: 3,
        GruConst.N: 6,
        GruConst.H_in: 500,
        GruConst.num_directions: 1,
        GruConst.hidden_size: 1000,
        GruConst.num_layers: 1
    }