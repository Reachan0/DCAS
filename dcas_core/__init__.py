"""
DCAS核心模块
Dynamic Curriculum Alignment System Core Modules
"""

__version__ = "1.0.0-simplified"
__author__ = "DCAS Team"
__description__ = "基于多智能体的动态课程对齐系统"

# 导入核心组件
try:
    from .data_models import (
        JobMarketData, CourseData, LearnerData, 
        AlignmentResult, ContentRecommendation, SimulationResult
    )
    from .config import get_config, setup_logging
    from .content_generator import PersonalizedContentGenerator
    from .orchestrator import DCASOrchestrator
    
    __all__ = [
        'JobMarketData', 'CourseData', 'LearnerData',
        'AlignmentResult', 'ContentRecommendation', 'SimulationResult',
        'get_config', 'setup_logging',
        'PersonalizedContentGenerator', 'DCASOrchestrator'
    ]
    
except ImportError as e:
    # 在开发阶段可能出现循环导入，这里提供备选
    __all__ = []
    import warnings
    warnings.warn(f"DCAS核心模块导入警告: {e}", ImportWarning)