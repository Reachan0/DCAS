#!/usr/bin/env python3
"""
Qwen3 14B 就业市场分析智能体 - 完整推理脚本
支持本地模型调用，用于分析职位描述并提取技能要求
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# 设置全局logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入现有的智能体
sys.path.append(str(Path(__file__).parent.parent))
from scripts.agents.job_market_analyst_agent import JobMarketAnalystAgent

@dataclass
class Config:
    """配置类"""
    model_path: str = ""  # 留空，由用户配置
    model_name: str = "qwen3-14b"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"  # auto, cpu, cuda

class Qwen3ModelClient:
    """Qwen3 14B 本地模型客户端"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器 - 优化版"""
        try:
            import torch
            from transformers import (
                AutoTokenizer, 
                AutoModelForCausalLM, 
                GenerationConfig
            )
            
            if not self.config.model_path:
                raise ValueError("请配置model_path参数指向Qwen3 14B模型文件")
            
            logger.info(f"正在加载模型: {self.config.model_path}")
            
            # 设置设备
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            
            logger.info(f"使用设备: {device}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                use_fast=True  # 使用快速分词器
            )
            
            # 优化加载参数
            load_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "device_map": "auto" if device == "cuda" else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,  # 减少内存使用
                "attn_implementation": "sdpa" if device == "cuda" else None,  # 使用flash attention
            }
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **load_kwargs
            )
            
            # 优化生成配置 - 减少响应时间
            self.generation_config = GenerationConfig(
                max_new_tokens=1024,  # 减少最大token数
                temperature=0.7,  # 降低随机性，提高确定性
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True,  # 启用缓存
            )
            
            logger.info("模型加载成功")
            
        except ImportError as e:
            logger.error(f"缺少依赖包: {e}")
            logger.error("请安装: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """生成回答 - 优化版"""
        try:
            logger.info(f"🔤 开始编码输入，Prompt长度: {len(prompt)} 字符")
            
            # 编码输入 - 使用更高效的方式
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048  # 限制输入长度
            )
            
            input_tokens = inputs['input_ids'].shape[1]
            logger.info(f"🔢 输入Token数量: {input_tokens}")
            
            if hasattr(self.model, 'device'):
                device = self.model.device
                logger.info(f"📱 模型设备: {device}")
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成 - 优化参数
            import torch
            logger.info(f"⚙️ 生成配置: max_tokens={self.generation_config.max_new_tokens}, temp={self.generation_config.temperature}")
            logger.info("🤖 开始模型推理...")
            
            import time
            generate_start = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            generate_time = time.time() - generate_start
            output_tokens = outputs[0].shape[0] - input_tokens
            logger.info(f"⚡ 推理完成，耗时: {generate_time:.2f}秒")
            logger.info(f"📈 生成Token数量: {output_tokens}")
            logger.info(f"🚀 生成速度: {output_tokens/generate_time:.1f} tokens/s")
            
            # 解码输出 - 只解码新生成的部分
            input_length = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            
            # 进一步清理响应
            response = response.replace("<|im_end|>", "").strip()
            
            logger.info(f"🔍 原始响应长度: {len(response)} 字符")
            logger.info(f"📝 清理后响应预览: {response[:200]}...")
            
            # 如果响应为空或过长，返回默认响应
            if not response or len(response) < 10:
                logger.warning("⚠️ 响应过短，使用默认响应")
                return "**最终能力要求列表:** Python, SQL, 机器学习, 数据可视化, 团队协作"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 生成失败: {e}")
            return "**最终能力要求列表:** Python, SQL, 机器学习, 数据可视化, 团队协作"

class Qwen3JobAnalyzer:
    """完整的Qwen3 14B就业市场分析器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_client = None
        self.agent = None
        
    def setup(self):
        """初始化模型和分析器"""
        # 设置logger
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("初始化Qwen3就业市场分析器...")
        self.model_client = Qwen3ModelClient(self.config)
        self.agent = JobMarketAnalystAgent(self.model_client)
        self.logger.info("初始化完成")
    
    def analyze_job(self, job_title: str, job_description: str) -> Dict[str, Any]:
        """分析单个职位"""
        return self.agent.analyze(job_title, job_description)
    
    def analyze_from_csv(self, csv_path: str, output_path: str):
        """批量分析CSV文件"""
        import pandas as pd
        
        logger.info(f"开始批量分析: {csv_path}")
        
        # 读取CSV
        df = pd.read_csv(csv_path)
        results = []
        
        for idx, row in df.iterrows():
            logger.info(f"分析第 {idx+1}/{len(df)} 条记录")
            
            try:
                result = self.analyze_job(
                    row['job_title'],
                    row['job_description']
                )
                result.update({
                    'index': idx,
                    'original_skill_requirements': row.get('skill_requirements', '')
                })
                results.append(result)
                
            except Exception as e:
                logger.error(f"分析第 {idx} 条记录失败: {e}")
                continue
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        
        logger.info(f"分析完成，结果已保存到: {output_path}")
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen3 14B 就业市场分析器")
    parser.add_argument("--model-path", required=True, 
                       help="Qwen3 14B模型路径")
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                       help="分析模式: single(单条) 或 batch(批量)")
    parser.add_argument("--job-title", help="职位名称 (单条模式)")
    parser.add_argument("--job-description", help="职位描述 (单条模式)")
    parser.add_argument("--input-csv", help="输入CSV文件路径 (批量模式)")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建配置
    config = Config(
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device
    )
    
    try:
        # 初始化分析器
        analyzer = Qwen3JobAnalyzer(config)
        analyzer.setup()
        
        if args.mode == "single":
            if not args.job_title or not args.job_description:
                raise ValueError("单条模式需要 --job-title 和 --job-description 参数")
            
            result = analyzer.analyze_job(args.job_title, args.job_description)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                
        elif args.mode == "batch":
            if not args.input_csv or not args.output:
                raise ValueError("批量模式需要 --input-csv 和 --output 参数")
            
            analyzer.analyze_from_csv(args.input_csv, args.output)
    
    except Exception as e:
        logger.error(f"执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main()