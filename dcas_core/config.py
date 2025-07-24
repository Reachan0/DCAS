#!/usr/bin/env python3
"""
DCAS配置管理系统
统一管理所有组件的配置参数
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DCASConfig:
    """DCAS系统配置管理器"""
    
    def __init__(self, config_file: str = "dcas_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_default_config()
        
        # 如果配置文件存在，则加载并覆盖默认配置
        if self.config_file.exists():
            self._load_config_file()
        else:
            self._save_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "system": {
                "version": "1.0.0-simplified",
                "debug": True,
                "log_level": "INFO",
                "data_dir": "dcas_data",
                "temp_dir": "dcas_temp"
            },
            "models": {
                "job_analysis": {
                    "model_type": "local",  # local, api, mock
                    "model_name": "Qwen/Qwen2.5-7B-Instruct",
                    "api_endpoint": None,
                    "max_tokens": 500,
                    "temperature": 0.1
                },
                "content_generation": {
                    "model_type": "local",
                    "model_name": "Qwen/Qwen2.5-7B-Instruct", 
                    "max_tokens": 1000,
                    "temperature": 0.3
                },
                "embedding": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 32
                }
            },
            "agents": {
                "job_market_analyst": {
                    "enabled": True,
                    "use_mock": False,
                    "cache_results": True,
                    "cache_duration_hours": 24
                },
                "learner_profiling": {
                    "enabled": True,
                    "profile_storage_dir": "learner_profiles",
                    "update_frequency_minutes": 60
                },
                "dynamic_alignment": {
                    "enabled": True,
                    "similarity_threshold": 0.7,
                    "strategy_storage_dir": "alignment_strategies"
                },
                "content_generator": {
                    "enabled": True,
                    "template_dir": "content_templates",
                    "output_dir": "generated_content"
                },
                "simulation_reflection": {
                    "enabled": True,
                    "num_virtual_students": 50,  # 简化版减少数量
                    "simulation_days": 14,
                    "storage_dir": "simulation_results"
                }
            },
            "knowledge_graph": {
                "similarity_threshold": 0.6,
                "max_edges": 10000,  # 简化版限制边数
                "embedding_dim": 384,
                "cache_embeddings": True
            },
            "web_interface": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 8501,
                "debug": True
            },
            "data_sources": {
                "course_data_dir": "datasets/Course Details/General",
                "job_postings_file": "datasets/Job Descptions/postings.csv",
                "output_base_dir": "dcas_output"
            }
        }
    
    def _load_config_file(self):
        """从文件加载配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                self._deep_update(self.config, file_config)
                logger.info(f"配置文件加载成功: {self.config_file}")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
                logger.info(f"配置文件保存成功: {self.config_file}")
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        使用点号分隔的路径获取配置值
        例: config.get("models.job_analysis.temperature")
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        使用点号分隔的路径设置配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 导航到最后一级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
        self._save_config()
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取特定模型的配置"""
        return self.get(f"models.{model_type}", {})
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """获取特定智能体的配置"""
        return self.get(f"agents.{agent_name}", {})
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """检查智能体是否启用"""
        return self.get(f"agents.{agent_name}.enabled", False)
    
    def get_data_dir(self, dir_type: str = "base") -> Path:
        """获取数据目录路径"""
        if dir_type == "base":
            return Path(self.get("data_sources.output_base_dir"))
        elif dir_type == "course":
            return Path(self.get("data_sources.course_data_dir"))
        elif dir_type == "temp":
            return Path(self.get("system.temp_dir"))
        else:
            return Path(self.get("system.data_dir"))
    
    def ensure_directories(self):
        """确保所有必要的目录存在"""
        dirs_to_create = [
            self.get("system.data_dir"),
            self.get("system.temp_dir"), 
            self.get("data_sources.output_base_dir"),
            self.get("agents.learner_profiling.profile_storage_dir"),
            self.get("agents.dynamic_alignment.strategy_storage_dir"),
            self.get("agents.content_generator.output_dir"),
            self.get("agents.simulation_reflection.storage_dir")
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info("所有必要目录已创建")
    
    def get_full_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config.copy()
    
    def reload(self):
        """重新加载配置文件"""
        if self.config_file.exists():
            self._load_config_file()
            logger.info("配置已重新加载")
        else:
            logger.warning("配置文件不存在，使用默认配置")

# 全局配置实例
_global_config = None

def get_config() -> DCASConfig:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = DCASConfig()
    return _global_config

def setup_logging():
    """根据配置设置日志"""
    config = get_config()
    log_level = config.get("system.log_level", "INFO")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dcas.log', encoding='utf-8')
        ]
    )

if __name__ == "__main__":
    # 测试配置系统
    config = DCASConfig()
    
    print("🔧 配置系统测试")
    print(f"系统版本: {config.get('system.version')}")
    print(f"作业分析模型: {config.get('models.job_analysis.model_name')}")
    print(f"调试模式: {config.get('system.debug')}")
    
    # 测试设置配置
    config.set("test.value", "hello")
    print(f"测试值: {config.get('test.value')}")
    
    # 确保目录存在
    config.ensure_directories()
    
    print("✅ 配置系统测试通过")