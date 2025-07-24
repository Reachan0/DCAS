#!/usr/bin/env python3
"""
DCAS核心数据模型
统一系统中各个智能体的数据接口和格式
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

@dataclass
class JobMarketData:
    """就业市场数据统一格式"""
    job_title: str
    required_skills: List[str] 
    skill_importance: Dict[str, float]  # 0-1
    market_trend: str  # growing, stable, declining
    salary_info: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CourseData:
    """课程数据统一格式"""
    course_id: str
    course_name: str
    modules: List[Dict[str, Any]]
    skill_coverage: Dict[str, float]  # skill -> coverage (0-1)
    difficulty_level: str  # beginner, intermediate, advanced
    estimated_hours: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass 
class LearnerData:
    """学习者数据统一格式"""
    user_id: str
    knowledge_state: Dict[str, float]  # concept -> mastery (0-1)
    learning_style: Dict[str, str]
    preferences: Dict[str, Any]
    skill_gaps: List[str]
    engagement_score: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AlignmentResult:
    """对齐分析结果统一格式"""
    analysis_id: str
    timestamp: str
    gaps_identified: List[Dict[str, Any]]
    strategies_generated: List[Dict[str, Any]]
    priority_actions: List[str]
    confidence_score: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ContentRecommendation:
    """内容推荐统一格式"""
    recommendation_id: str
    target_skills: List[str]
    content_modules: List[Dict[str, Any]]
    learning_path: List[str]
    personalization_notes: List[str]
    estimated_completion_time: float
    knowledge_graph_data: Optional[Dict[str, Any]] = None  # 知识图谱增强数据
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SimulationResult:
    """模拟结果统一格式"""
    simulation_id: str
    predicted_outcomes: Dict[str, float]
    success_probability: float  # 0-1
    risk_factors: List[str]
    optimization_suggestions: List[str]
    confidence_level: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_job_market_data(data: Dict[str, Any]) -> bool:
        """验证就业市场数据格式"""
        required_fields = ['job_title', 'required_skills', 'skill_importance']
        return all(field in data for field in required_fields)
    
    @staticmethod  
    def validate_course_data(data: Dict[str, Any]) -> bool:
        """验证课程数据格式"""
        required_fields = ['course_id', 'course_name', 'modules']
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_learner_data(data: Dict[str, Any]) -> bool:
        """验证学习者数据格式"""
        required_fields = ['user_id', 'knowledge_state', 'learning_style']
        return all(field in data for field in required_fields)

class DataTransformer:
    """数据转换器 - 将现有格式转换为统一格式"""
    
    @staticmethod
    def transform_job_analysis_result(raw_result: Dict[str, Any]) -> JobMarketData:
        """转换就业市场分析结果"""
        skills = raw_result.get('final_skills', '').split(', ')
        
        return JobMarketData(
            job_title=raw_result.get('job_title', ''),
            required_skills=[s.strip() for s in skills if s.strip()],
            skill_importance={skill: 0.8 for skill in skills if skill.strip()},  # 默认重要性
            market_trend="stable",  # 默认值
            salary_info={"min": 0, "max": 0, "avg": 0}  # 默认值
        )
    
    @staticmethod
    def transform_learner_profile(profile_dict: Dict[str, Any]) -> LearnerData:
        """转换学习者画像"""
        knowledge_state = profile_dict.get('knowledge_state', {}).get('concept_mastery', {})
        
        # 将mastered/learning/not_started转换为数值
        knowledge_numeric = {}
        for concept, status in knowledge_state.items():
            if status == 'mastered':
                knowledge_numeric[concept] = 0.9
            elif status == 'learning':
                knowledge_numeric[concept] = 0.5
            else:
                knowledge_numeric[concept] = 0.1
        
        return LearnerData(
            user_id=profile_dict.get('user_id', ''),
            knowledge_state=knowledge_numeric,
            learning_style=profile_dict.get('learning_style', {}),
            preferences=profile_dict.get('behavioral_patterns', {}),
            skill_gaps=[],  # 需要从其他数据推导
            engagement_score=profile_dict.get('behavioral_patterns', {}).get('engagement_score', 0.5)
        )
    
    @staticmethod
    def transform_course_modules(modules_list: List[Dict[str, Any]]) -> CourseData:
        """转换课程模块数据"""
        if not modules_list:
            return CourseData(
                course_id="default",
                course_name="Default Course",
                modules=[],
                skill_coverage={},
                difficulty_level="intermediate",
                estimated_hours=0
            )
        
        # 提取技能覆盖率
        skill_coverage = {}
        total_hours = 0
        
        for module in modules_list:
            concepts = module.get('concepts', [])
            duration = module.get('duration_hours', 0)
            total_hours += duration
            
            for concept in concepts:
                skill_coverage[concept] = skill_coverage.get(concept, 0) + 0.3
        
        # 限制覆盖率在0-1之间
        skill_coverage = {k: min(1.0, v) for k, v in skill_coverage.items()}
        
        return CourseData(
            course_id="course_001",
            course_name="Integrated Course",
            modules=modules_list,
            skill_coverage=skill_coverage,
            difficulty_level="intermediate",
            estimated_hours=total_hours
        )

if __name__ == "__main__":
    # 测试数据模型
    job_data = JobMarketData(
        job_title="数据科学家",
        required_skills=["Python", "机器学习", "SQL"],
        skill_importance={"Python": 0.9, "机器学习": 0.8, "SQL": 0.7},
        market_trend="growing",
        salary_info={"min": 15000, "max": 30000, "avg": 22000}
    )
    
    print("✅ 数据模型测试通过")
    print("Job Data:", json.dumps(job_data.to_dict(), ensure_ascii=False, indent=2))