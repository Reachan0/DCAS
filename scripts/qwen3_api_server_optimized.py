#!/usr/bin/env python3
"""
优化版Qwen3 API服务器 - 解决性能和超时问题
"""

import os
import sys
import gc
import psutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import List, Optional
import json
import asyncio
from datetime import datetime
import threading
import time
from contextlib import asynccontextmanager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据模型
class JobRequest(BaseModel):
    job_title: str
    job_description: str
    timeout: Optional[int] = 30  # 可配置超时时间

class JobResponse(BaseModel):
    job_title: str
    job_description: str
    final_skills: str
    cot_response: str
    success: bool
    timestamp: str
    processing_time: Optional[float] = None

class SystemStats(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    model_loaded: bool
    queue_size: int

# 优化的模型管理器
class OptimizedModelManager:
    """优化的模型管理器"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.is_loading = False
        self.load_error = None
        self.analyzer = None
        self.processing_queue = []
        self.max_queue_size = 10
        
    def load_model_with_optimization(self):
        """优化的模型加载"""
        if self.is_loading:
            logger.info("模型正在加载中...")
            return False
            
        self.is_loading = True
        
        try:
            logger.info("开始加载优化版Qwen3模型...")
            
            # 内存检查
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            logger.info(f"可用内存: {available_gb:.1f}GB")
            
            if available_gb < 8:
                logger.warning("⚠️ 可用内存不足8GB，可能影响性能")
            
            # 清理内存
            gc.collect()
            
            # 配置优化参数
            model_path = os.getenv("QWEN3_MODEL_PATH", "/Users/chenxuanchong/fsdownload/Qwen3-8B-SFT/")
            
            # 检查模型路径
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                # 使用Mock模式
                self.analyzer = MockAnalyzer()
                logger.info("⚠️ 切换到Mock模式")
                self.is_loading = False
                return True
            
            # 尝试加载真实模型
            try:
                from qwen3_analyzer import Qwen3JobAnalyzer, Config
                
                config = Config(
                    model_path=model_path,
                    max_tokens=256,  # 大幅减少token数
                    temperature=0.1,  # 降低温度提高速度
                    device="auto"
                )
                
                # 设置环境变量优化推理
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                
                self.analyzer = Qwen3JobAnalyzer(config)
                self.analyzer.setup()
                
                logger.info("✅ Qwen3模型加载成功")
                
            except Exception as model_error:
                logger.error(f"模型加载失败: {model_error}")
                logger.info("切换到Mock模式...")
                self.analyzer = MockAnalyzer()
                
            self.is_loading = False
            return True
            
        except Exception as e:
            logger.error(f"模型管理器初始化失败: {e}")
            self.load_error = str(e)
            self.is_loading = False
            return False
    
    def get_system_stats(self) -> SystemStats:
        """获取系统状态"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return SystemStats(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            model_loaded=self.analyzer is not None,
            queue_size=len(self.processing_queue)
        )
    
    async def analyze_with_timeout(self, job_title: str, job_description: str, timeout: int = 30):
        """带超时的异步分析"""
        if not self.analyzer:
            raise HTTPException(status_code=503, detail="分析器未初始化")
        
        # 检查队列长度
        if len(self.processing_queue) >= self.max_queue_size:
            raise HTTPException(status_code=429, detail="服务器忙，请稍后重试")
        
        # 添加到处理队列
        task_id = f"task_{int(time.time() * 1000)}"
        self.processing_queue.append(task_id)
        
        try:
            # 创建异步任务
            loop = asyncio.get_event_loop()
            
            # 使用超时控制
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    self.analyzer.analyze_job, 
                    job_title, 
                    job_description
                ),
                timeout=timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"分析超时 ({timeout}s): {job_title}")
            raise HTTPException(status_code=408, detail=f"分析超时，请尝试缩短职位描述或增加超时时间")
        except Exception as e:
            logger.error(f"分析失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # 从队列中移除
            if task_id in self.processing_queue:
                self.processing_queue.remove(task_id)

class MockAnalyzer:
    """Mock分析器，用于性能不足时的快速响应"""
    
    def analyze_job(self, job_title: str, job_description: str):
        """模拟分析"""
        # 简单的关键词提取
        skills = []
        
        # 技术技能关键词
        tech_keywords = {
            'Python': ['python', 'py'],
            '数据分析': ['数据分析', '数据处理', 'data analysis'],
            '机器学习': ['机器学习', 'machine learning', 'ml'],
            'SQL': ['sql', '数据库'],
            'Java': ['java'],
            'JavaScript': ['javascript', 'js'],
            '深度学习': ['深度学习', 'deep learning'],
            '云计算': ['云计算', 'aws', 'azure', '阿里云'],
            '项目管理': ['项目管理', 'project management'],
            '团队协作': ['团队', '协作', '沟通'],
        }
        
        text = (job_title + " " + job_description).lower()
        
        for skill, keywords in tech_keywords.items():
            if any(keyword in text for keyword in keywords):
                skills.append(skill)
        
        if not skills:
            skills = ['编程', '团队协作', '问题解决']
        
        return {
            "job_title": job_title,
            "job_description": job_description,
            "final_skills": ", ".join(skills[:5]),  # 限制5个技能
            "cot_response": f"基于关键词匹配分析职位: {job_title}",
            "success": True
        }

# 全局模型管理器
model_manager = OptimizedModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 正在启动优化版Qwen3服务...")
    
    # 异步加载模型
    load_success = model_manager.load_model_with_optimization()
    
    if not load_success:
        logger.warning("⚠️ 模型加载失败，服务将以降级模式运行")
    
    logger.info("✅ 服务启动完成")
    
    yield
    
    # 清理资源
    logger.info("🔧 正在清理资源...")
    gc.collect()

# FastAPI应用
app = FastAPI(
    title="优化版Qwen3就业市场分析API",
    description="性能优化版，支持超时控制和降级服务",
    version="2.1.0-optimized",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "优化版Qwen3就业市场分析API",
        "version": "2.1.0-optimized",
        "status": "running",
        "optimizations": [
            "异步处理",
            "超时控制", 
            "内存优化",
            "队列管理",
            "降级服务"
        ]
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    stats = model_manager.get_system_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_stats": stats.dict()
    }

@app.get("/stats")
async def get_stats():
    """获取详细系统统计"""
    stats = model_manager.get_system_stats()
    
    return {
        "system": stats.dict(),
        "process": {
            "pid": os.getpid(),
            "threads": threading.active_count(),
            "memory_info": dict(psutil.Process().memory_info()._asdict())
        },
        "model": {
            "loaded": model_manager.analyzer is not None,
            "loading": model_manager.is_loading,
            "error": model_manager.load_error
        }
    }

@app.post("/analyze", response_model=JobResponse)
async def analyze_job(request: JobRequest):
    """优化版单个职位分析"""
    start_time = time.time()
    
    try:
        result = await model_manager.analyze_with_timeout(
            request.job_title, 
            request.job_description,
            timeout=request.timeout
        )
        
        processing_time = time.time() - start_time
        
        return JobResponse(
            job_title=result["job_title"],
            job_description=result["job_description"],
            final_skills=result["final_skills"],
            cot_response=result["cot_response"],
            success=result["success"],
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/quick")
async def quick_analyze(request: JobRequest):
    """快速分析模式 - 使用Mock分析器"""
    start_time = time.time()
    
    try:
        # 直接使用Mock分析器进行快速分析
        mock_analyzer = MockAnalyzer()
        result = mock_analyzer.analyze_job(request.job_title, request.job_description)
        
        processing_time = time.time() - start_time
        
        return JobResponse(
            job_title=result["job_title"],
            job_description=result["job_description"],
            final_skills=result["final_skills"],
            cot_response=result["cot_response"] + " [快速模式]",
            success=result["success"],
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"快速分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload_model():
    """重新加载模型"""
    if model_manager.is_loading:
        return {"message": "模型正在加载中，请稍后"}
    
    success = model_manager.load_model_with_optimization()
    
    return {
        "success": success,
        "message": "模型重新加载完成" if success else "模型加载失败",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "qwen3_api_server_optimized:app",
        host="0.0.0.0",
        port=8001,  # 使用不同端口避免冲突
        reload=False,
        workers=1,
        log_level="info",
        timeout_keep_alive=30,
        limit_concurrency=5  # 限制并发数
    )