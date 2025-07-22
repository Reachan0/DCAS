#!/usr/bin/env python3
"""
基于微调Qwen3-14B的就业市场分析API服务
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

from job_market_analyst_agent import JobMarketAnalystAgent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI应用
app = FastAPI(
    title="就业市场分析API",
    description="基于微调Qwen3-14B的就业市场技能提取服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRequest(BaseModel):
    job_title: str
    job_description: str

class BatchJobRequest(BaseModel):
    jobs: list[JobRequest]

class JobResponse(BaseModel):
    job_title: str
    job_description: str
    final_skills: str
    cot_response: str
    success: bool
    timestamp: str

class FineTunedModelClient:
    """微调模型客户端"""
    
    def __init__(self, model_path: str = None):
        """
        初始化微调模型客户端
        
        Args:
            model_path: 微调模型路径，默认为本地微调后的Qwen3-14B
        """
        self.model_path = model_path or self._get_finetuned_model_path()
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _get_finetuned_model_path(self) -> str:
        """获取微调模型路径"""
        # 检查本地微调模型路径
        possible_paths = [
            "./models/qwen3-14b-finetuned",
            "./checkpoints/qwen3-14b-finetuned",
            "./output/qwen3-14b-finetuned",
            "qwen3-14b-finetuned"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        logger.warning("未找到微调模型，使用基础Qwen3-14B")
        return "Qwen/Qwen3-14B"
    
    def _load_model(self):
        """加载微调模型"""
        try:
            logger.info(f"正在加载微调模型: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 确保有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            logger.info(f"微调模型加载成功: {self.model_path}")
            
        except Exception as e:
            logger.error(f"加载微调模型失败: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """
        生成响应
        
        Args:
            prompt: 输入提示
            
        Returns:
            模型生成的响应
        """
        try:
            # 使用聊天模板
            messages = [
                {"role": "system", "content": "你是一个经过微调的专业职位分析师，请按照要求提供详细的思维链分析。"},
                {"role": "user", "content": prompt}
            ]
            
            # 格式化输入
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = prompt
            
            # 编码输入
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )
            
            # 移动到GPU（如果有）
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.95,
                    top_k=50
                )
            
            # 解码响应
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"模型生成失败: {e}")
            raise

# 全局变量
analyst_agent = None
model_client = None

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global analyst_agent, model_client
    
    logger.info("正在启动就业市场分析服务...")
    
    try:
        # 初始化微调模型客户端
        model_client = FineTunedModelClient()
        
        # 初始化分析智能体
        analyst_agent = JobMarketAnalystAgent(model_client)
        
        logger.info("就业市场分析服务启动成功！")
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "就业市场分析API服务已启动",
        "version": "1.0.0",
        "model": "Qwen3-14B-FineTuned",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": analyst_agent is not None
    }

@app.post("/analyze", response_model=JobResponse)
async def analyze_job(request: JobRequest):
    """分析单个职位"""
    if not analyst_agent:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        result = analyst_agent.analyze(request.job_title, request.job_description)
        
        return JobResponse(
            job_title=result["job_title"],
            job_description=result["job_description"],
            final_skills=result["final_skills"],
            cot_response=result["cot_response"],
            success=result["success"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def analyze_batch(request: BatchJobRequest):
    """批量分析职位"""
    if not analyst_agent:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        results = []
        
        for job in request.jobs:
            result = analyst_agent.analyze(job.job_title, job.job_description)
            results.append(JobResponse(
                job_title=result["job_title"],
                job_description=result["job_description"],
                final_skills=result["final_skills"],
                cot_response=result["cot_response"],
                success=result["success"],
                timestamp=datetime.now().isoformat()
            ))
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"批量分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def get_model_info():
    """获取模型信息"""
    if not model_client:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "model_path": model_client.model_path,
        "model_name": getattr(model_client, 'model_name', 'unknown'),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    uvicorn.run(
        "job_market_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # 单进程，避免模型重复加载
    )