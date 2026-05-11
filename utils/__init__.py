# logger 模块
from .logger import debug, info, emph, warning, error, fatal
from .logger import log_level, line_seg

# 创建 logger 对象供兼容使用
class _LoggerProxy:
    """logger 代理类，兼容 logger.info() 等调用方式"""
    debug = staticmethod(debug)
    info = staticmethod(info)
    emph = staticmethod(emph)
    warning = staticmethod(warning)
    error = staticmethod(error)
    fatal = staticmethod(fatal)

logger = _LoggerProxy()

# 导出其他模块
from .init import *
from .scheduler import WarmUpCosineAnnealingLR, FakeLR, LinearDecayLR
from .solver import *
