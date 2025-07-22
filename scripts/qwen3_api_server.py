#!/usr/bin/env python3
"""
基于微调Qwen3-14B的就业市场分析API服务器
结合qwen3_analyzer.py和job_market_analyst_agent.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import List
import json
from datetime import datetime

# 导入现有模块
from qwen3_analyzer import Qwen3JobAnalyzer, Config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI应用
app = FastAPI(
    title="Qwen3就业市场分析API",
    description="基于微调Qwen3-14B的就业市场技能提取服务",
    version="2.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class JobRequest(BaseModel):
    job_title: str
    job_description: str

class BatchJobRequest(BaseModel):
    jobs: List[JobRequest]

class JobResponse(BaseModel):
    job_title: str
    job_description: str
    final_skills: str
    cot_response: str
    success: bool
    timestamp: str

class BatchJobResponse(BaseModel):
    results: List[JobResponse]
    count: int
    processing_time: float

# 全局分析器实例
analyzer = None

@app.on_event("startup")
async def startup_event():
    """启动时初始化分析器"""
    global analyzer
    
    logger.info("正在启动Qwen3就业市场分析服务...")
    
    try:
        # 配置微调模型路径
        model_path = os.getenv("QWEN3_MODEL_PATH", "./models/qwen3-14b-finetuned")
        
        config = Config(
            model_path=model_path,
            max_tokens=2048,
            temperature=0.7,
            device="auto"
        )
        
        analyzer = Qwen3JobAnalyzer(config)
        analyzer.setup()
        
        logger.info("Qwen3就业市场分析服务启动成功！")
        logger.info(f"使用模型: {model_path}")
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Qwen3就业市场分析API已启动",
        "version": "2.0.0",
        "model": "Qwen3-14B-FineTuned",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": analyzer is not None
    }

@app.post("/analyze", response_model=JobResponse)
async def analyze_job(request: JobRequest):
    """分析单个职位"""
    if not analyzer:
        raise HTTPException(status_code=503, detail="分析器未初始化")
    
    try:
        start_time = datetime.now()
        
        result = analyzer.analyze_job(request.job_title, request.job_description)
        
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

@app.post("/analyze/batch", response_model=BatchJobResponse)
async def analyze_batch(request: BatchJobRequest):
    """批量分析职位"""
    if not analyzer:
        raise HTTPException(status_code=503, detail="分析器未初始化")
    
    try:
        start_time = datetime.now()
        results = []
        
        for job in request.jobs:
            result = analyzer.analyze_job(job.job_title, job.job_description)
            
            response = JobResponse(
                job_title=result["job_title"],
                job_description=result["job_description"],
                final_skills=result["final_skills"],
                cot_response=result["cot_response"],
                success=result["success"],
                timestamp=datetime.now().isoformat()
            )
            results.append(response)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchJobResponse(
            results=results,
            count=len(results),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"批量分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def get_info():
    """获取系统信息"""
    return {
        "service": "Qwen3就业市场分析API",
        "version": "2.0.0",
        "features": [
            "基于微调Qwen3-14B模型",
            "思维链(CoT)分析能力",
            "技能提取与分类",
            "批量分析支持"
        ],
        "endpoints": {
            "/analyze": "单条职位分析",
            "/analyze/batch": "批量职位分析",
            "/health": "健康检查",
            "/info": "系统信息"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "qwen3_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # 单进程避免模型重复加载
    )