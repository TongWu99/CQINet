# 导出实际存在的模型模块
from .TransNet import *
from .TransNet_MoE import *
from .TransNet_H import *
from .TransNet_QModPlusD import *
from .crnet import *
from .clnet import *
from .CQIFineTuner import *

# CQI 嵌入模型（用于微调）
from .crnet_cqi import *
from .crnet_cqi_film import *  # 新增：FiLM 调制方式
from .clnet_cqi import *
from .transnet_qmodplusd_cqi import *
