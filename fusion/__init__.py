# fusion/__init__.py
from .base_fusion import BaseFusion
from .concatenation_fusion import ConcatenationFusion
from .coattention_fusion import CoAttentionFusion
from .bilinear_fusion import BilinearFusion

__all__ = [
    'BaseFusion',
    'ConcatenationFusion',
    'CoAttentionFusion', 
    'BilinearFusion'
] 