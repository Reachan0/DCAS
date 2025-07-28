#!/usr/bin/env python3
"""
DCAS系统协调器
统一协调所有智能体，实现端到端工作流
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dcas_core.data_models import (
    JobMarketData, CourseData, LearnerData, AlignmentResult, 
    ContentRecommendation, SimulationResult, DataTransformer
)
from dcas_core.config import get_config, setup_logging
from dcas_core.content_generator import PersonalizedContentGenerator
from dcas_core.knowledge_graph_integrator import KnowledgeGraphIntegrator, EnhancedContentGenerator

# 导入现有智能体
from scripts.agents.job_market_analyst_agent import JobMarketAnalystAgent
from scripts.agents.learner_profiling_agent import LearnerProfilingAgent  
from scripts.agents.dynamic_alignment_agent import DynamicAlignmentAgent, MarketDemand, CourseStatus, LearnerSnapshot
from scripts.agents.simulation_reflection_agent import SimulationReflectionAgent

logger = logging.getLogger(__name__)

class ModelClientManager:
    """模型客户端管理器"""
    
    def __init__(self):
        self.config = get_config()
        self._clients = {}
    
    def get_client(self, model_type: str):
        """获取模型客户端"""
        if model_type not in self._clients:
            model_config = self.config.get_model_config(model_type)
            
            if model_config.get("model_type") == "mock":
                self._clients[model_type] = MockLLMClient()
            else:
                # 在实际部署中，这里会加载真实的模型客户端
                self._clients[model_type] = MockLLMClient()
                logger.warning(f"使用Mock客户端代替 {model_type}")
        
        return self._clients[model_type]

class MockLLMClient:
    """统一的Mock LLM客户端"""
    
    def generate(self, prompt: str) -> str:
        """模拟LLM生成"""
        if "职位名称" in prompt and "技能" in prompt:
            return """**职位类型识别:**
这是一个技术类职位，需要综合的编程和分析技能。

**技术技能深度分析:**
需要掌握编程语言、数据处理工具和分析方法。

**业务技能映射:**
需要理解业务场景，具备问题解决和项目管理能力。

**软技能与协作要求:**
需要良好的沟通能力和团队协作精神。

**最终能力要求列表:** [根据职位分析得出的技能清单]"""
        
        elif "学习风格" in prompt:
            return '{"processing": "active", "perception": "sensory", "understanding": "sequential"}'
        
        else:
            return "基于分析，建议采用渐进式学习方法，重点关注实践应用。"

class DCASOrchestrator:
    """DCAS系统协调器"""
    
    def __init__(self):
        self.config = get_config()
        setup_logging()
        
        # 确保必要目录存在
        self.config.ensure_directories()
        
        # 初始化模型客户端管理器
        self.model_manager = ModelClientManager()
        
        # 初始化各个智能体
        self._initialize_agents()
        
        # 数据转换器
        self.data_transformer = DataTransformer()
        
        logger.info("DCAS系统协调器初始化完成")
    
    def _initialize_agents(self):
        """初始化所有智能体"""
        try:
            # 1. 就业市场分析智能体
            if self.config.is_agent_enabled("job_market_analyst"):
                llm_client = self.model_manager.get_client("job_analysis")
                self.job_analyst = JobMarketAnalystAgent(llm_client)
                logger.info("✅ 就业市场分析智能体已初始化")
            else:
                self.job_analyst = None
                logger.info("❌ 就业市场分析智能体已禁用")
            
            # 2. 学习者画像智能体
            if self.config.is_agent_enabled("learner_profiling"):
                llm_client = self.model_manager.get_client("job_analysis")
                profile_dir = self.config.get("agents.learner_profiling.profile_storage_dir")
                self.learner_profiler = LearnerProfilingAgent(llm_client, profile_dir)
                logger.info("✅ 学习者画像智能体已初始化")
            else:
                self.learner_profiler = None
                logger.info("❌ 学习者画像智能体已禁用")
            
            # 3. 动态对齐策略智能体
            if self.config.is_agent_enabled("dynamic_alignment"):
                llm_client = self.model_manager.get_client("job_analysis")
                strategy_dir = self.config.get("agents.dynamic_alignment.strategy_storage_dir")
                self.alignment_agent = DynamicAlignmentAgent(llm_client, strategy_dir)
                logger.info("✅ 动态对齐策略智能体已初始化")
            else:
                self.alignment_agent = None
                logger.info("❌ 动态对齐策略智能体已禁用")
            
            # 4. 知识图谱集成器
            if self.config.is_agent_enabled("knowledge_graph"):
                kg_dir = self.config.get("knowledge_graph.output_dir", None)
                self.kg_integrator = KnowledgeGraphIntegrator(kg_dir)
                logger.info("✅ 知识图谱集成器已初始化")
            else:
                self.kg_integrator = KnowledgeGraphIntegrator()  # 使用默认配置
                logger.info("🔧 知识图谱集成器使用默认配置")
            
            # 5. 内容生成器（增强版）
            if self.config.is_agent_enabled("content_generator"):
                llm_client = self.model_manager.get_client("content_generation")
                self.content_generator = PersonalizedContentGenerator(llm_client)
                self.enhanced_content_generator = EnhancedContentGenerator(self.kg_integrator)
                logger.info("✅ 课程内容生成器已初始化（含知识图谱增强）")
            else:
                self.content_generator = None
                self.enhanced_content_generator = None
                logger.info("❌ 课程内容生成器已禁用")
            
            # 6. 模拟与反思智能体
            if self.config.is_agent_enabled("simulation_reflection"):
                llm_client = self.model_manager.get_client("job_analysis")
                simulation_dir = self.config.get("agents.simulation_reflection.storage_dir")
                self.simulation_agent = SimulationReflectionAgent(llm_client, simulation_dir)
                logger.info("✅ 模拟与反思智能体已初始化")
            else:
                self.simulation_agent = None
                logger.info("❌ 模拟与反思智能体已禁用")
                
        except Exception as e:
            logger.error(f"智能体初始化失败: {e}")
            raise
    
    def process_learning_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理学习请求的完整工作流
        
        Args:
            request_data: 包含职位信息、学习者信息等的请求数据
            
        Returns:
            完整的处理结果，包括分析、推荐、模拟等
        """
        logger.info("开始处理学习请求")
        
        try:
            # 第1步：分析就业市场需求
            logger.info("📊 第1步：分析就业市场需求")
            job_market_data = self._analyze_job_market(request_data.get("job_info", {}))
            
            # 第2步：分析学习者画像
            logger.info("👤 第2步：分析学习者画像")  
            learner_data = self._analyze_learner_profile(request_data.get("learner_info", {}))
            
            # 第3步：执行动态对齐分析
            logger.info("🎯 第3步：执行动态对齐分析")
            alignment_result = self._perform_alignment_analysis(
                job_market_data, learner_data, request_data.get("course_info", {})
            )
            
            # 第4步：生成个性化内容推荐
            logger.info("📚 第4步：生成个性化内容推荐")
            content_recommendation = self._generate_content_recommendation(
                job_market_data, learner_data
            )
            
            # 第5步：执行效果模拟
            logger.info("🎮 第5步：执行效果模拟")
            simulation_result = self._simulate_learning_outcome(
                learner_data, content_recommendation, alignment_result
            )
            
            # 第6步：构建完整响应
            logger.info("📝 第6步：构建完整响应")
            response = self._build_response(
                job_market_data, learner_data, alignment_result,
                content_recommendation, simulation_result
            )
            
            logger.info("✅ 学习请求处理完成")
            return response
            
        except Exception as e:
            logger.error(f"处理学习请求失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_job_market(self, job_info: Dict[str, Any]) -> JobMarketData:
        """分析就业市场需求"""
        if not self.job_analyst:
            # 使用默认数据
            return JobMarketData(
                job_title=job_info.get("title", "软件工程师"),
                required_skills=["编程", "团队协作", "问题解决"],
                skill_importance={"编程": 0.9, "团队协作": 0.7, "问题解决": 0.8},
                market_trend="stable",
                salary_info={"min": 10000, "max": 20000, "avg": 15000}
            )
        
        try:
            # 调用就业市场分析智能体
            result = self.job_analyst.analyze(
                job_info.get("title", ""),
                job_info.get("description", "")
            )
            
            # 转换为统一格式
            return self.data_transformer.transform_job_analysis_result(result)
        except Exception as e:
            logger.error(f"就业市场分析失败: {e}")
            raise
    
    def _analyze_learner_profile(self, learner_info: Dict[str, Any]) -> LearnerData:
        """分析学习者画像"""
        user_id = learner_info.get("user_id", f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if not self.learner_profiler:
            # 使用默认数据
            return LearnerData(
                user_id=user_id,
                knowledge_state=learner_info.get("current_skills", {}),
                learning_style={"processing": "active", "perception": "sensory"},
                preferences={"engagement_score": 0.7},
                skill_gaps=[],
                engagement_score=0.7
            )
        
        try:
            # 检查是否已有画像
            try:
                profile = self.learner_profiler.get_profile(user_id)
            except FileNotFoundError:
                # 创建新画像
                initial_data = {
                    "interests": learner_info.get("interests", []),
                    "self_description": learner_info.get("description", "")
                }
                self.learner_profiler.create_profile(user_id, initial_data)
                profile = self.learner_profiler.get_profile(user_id)
            
            # 转换为统一格式
            profile_dict = {
                "user_id": profile.user_id,
                "knowledge_state": {"concept_mastery": profile.knowledge_state.concept_mastery},
                "learning_style": {
                    "processing": profile.learning_style.processing,
                    "perception": profile.learning_style.perception,
                    "understanding": profile.learning_style.understanding
                },
                "behavioral_patterns": {
                    "engagement_score": profile.behavioral_patterns.engagement_score
                }
            }
            
            return self.data_transformer.transform_learner_profile(profile_dict)
        except Exception as e:
            logger.error(f"学习者画像分析失败: {e}")
            raise
    
    def _perform_alignment_analysis(self, job_market_data: JobMarketData,
                                  learner_data: LearnerData,
                                  course_info: Dict[str, Any]) -> AlignmentResult:
        """执行动态对齐分析"""
        if not self.alignment_agent:
            # 返回默认结果
            return AlignmentResult(
                analysis_id=f"alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now().isoformat(),
                gaps_identified=[{"type": "skill_gap", "description": "需要加强技术技能"}],
                strategies_generated=[{"type": "add_content", "description": "增加相关技能模块"}],
                priority_actions=["重点学习核心技能"],
                confidence_score=0.8
            )
        
        try:
            # 转换为智能体需要的格式
            market_demand = MarketDemand(
                job_title=job_market_data.job_title,
                required_skills=job_market_data.required_skills,
                skill_importance=job_market_data.skill_importance,
                market_trend=job_market_data.market_trend,
                salary_range=job_market_data.salary_info
            )
            
            course_status = CourseStatus(
                course_id="current_course",
                course_name=course_info.get("name", "当前课程"),
                covered_concepts=list(learner_data.knowledge_state.keys()),
                skill_coverage=learner_data.knowledge_state,
                difficulty_level="intermediate",
                student_enrollment=1,
                completion_rate=0.8
            )
            
            learner_snapshot = LearnerSnapshot(
                user_id=learner_data.user_id,
                knowledge_mastery={k: "mastered" if v > 0.8 else "learning" if v > 0.3 else "not_started" 
                                 for k, v in learner_data.knowledge_state.items()},
                learning_style=learner_data.learning_style,
                behavioral_patterns=learner_data.preferences,
                skill_gaps=learner_data.skill_gaps
            )
            
            # 执行对齐分析
            report = self.alignment_agent.analyze_and_plan(
                market_demand, course_status, learner_snapshot
            )
            
            # 转换结果格式
            return AlignmentResult(
                analysis_id=report["analysis_id"],
                timestamp=report["timestamp"],
                gaps_identified=report["alignment_issues"],
                strategies_generated=report["adjustment_strategies"],
                priority_actions=[s["expected_outcome"] for s in report["adjustment_strategies"][:3]],
                confidence_score=0.8
            )
            
        except Exception as e:
            logger.error(f"对齐分析失败: {e}")
            raise
    
    def _generate_content_recommendation(self, job_market_data: JobMarketData,
                                       learner_data: LearnerData) -> ContentRecommendation:
        """生成个性化内容推荐（集成知识图谱）"""
        if not self.content_generator:
            # 返回默认推荐
            return ContentRecommendation(
                recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                target_skills=job_market_data.required_skills[:3],
                content_modules=[],
                learning_path=[],
                personalization_notes=["基于默认模板生成的推荐"],
                estimated_completion_time=20.0
            )
        
        try:
            # 首先生成基础推荐
            base_recommendation = self.content_generator.generate_personalized_content(
                job_market_data, learner_data
            )
            
            # 如果有知识图谱增强生成器，则使用增强功能
            if self.enhanced_content_generator:
                logger.info("🔗 使用知识图谱增强内容生成")
                kg_enhanced = self.enhanced_content_generator.generate_kg_enhanced_content(
                    job_market_data, learner_data
                )
                
                # 合并基础推荐和知识图谱增强
                enhanced_recommendation = self._merge_recommendations(
                    base_recommendation, kg_enhanced
                )
                return enhanced_recommendation
            
            return base_recommendation
            
        except Exception as e:
            logger.error(f"内容推荐生成失败: {e}")
            raise
    
    def _merge_recommendations(self, base_recommendation: ContentRecommendation, 
                             kg_enhanced: Dict[str, Any]) -> ContentRecommendation:
        """合并基础推荐和知识图谱增强推荐"""
        enhanced_data = kg_enhanced.get('enhanced_recommendation', {})
        
        # 扩展个性化说明
        enhanced_notes = base_recommendation.personalization_notes.copy()
        enhanced_notes.extend(enhanced_data.get('kg_integration_notes', []))
        
        # 如果有知识图谱学习路径，添加相关信息
        kg_path = enhanced_data.get('personalized_learning_path', {})
        if kg_path.get('recommended_courses'):
            enhanced_notes.append(
                f"🎓 基于知识图谱推荐了{len(kg_path['recommended_courses'])}门相关课程"
            )
        
        # 创建增强的推荐
        enhanced_recommendation = ContentRecommendation(
            recommendation_id=base_recommendation.recommendation_id + "_kg_enhanced",
            target_skills=base_recommendation.target_skills,
            content_modules=base_recommendation.content_modules,
            learning_path=base_recommendation.learning_path,
            personalization_notes=enhanced_notes,
            estimated_completion_time=base_recommendation.estimated_completion_time,
            # 添加知识图谱特有信息
            knowledge_graph_data=enhanced_data
        )
        
        return enhanced_recommendation
    
    def _simulate_learning_outcome(self, learner_data: LearnerData,
                                 content_recommendation: ContentRecommendation,
                                 alignment_result: AlignmentResult) -> SimulationResult:
        """模拟学习效果"""
        if not self.simulation_agent:
            # 返回默认模拟结果
            return SimulationResult(
                simulation_id=f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                predicted_outcomes={skill: 0.8 for skill in content_recommendation.target_skills},
                success_probability=0.75,
                risk_factors=["时间管理", "难度适应"],
                optimization_suggestions=["制定学习计划", "寻求指导"],
                confidence_level=0.7
            )
        
        try:
            # 构建模拟输入数据
            learner_profiles = [{
                "user_id": learner_data.user_id,
                "knowledge_mastery": learner_data.knowledge_state,
                "learning_style": learner_data.learning_style,
                "behavioral_patterns": learner_data.preferences
            }]
            
            course_modules = [{
                "module_id": module.get("module_id", f"module_{i}"),
                "title": module.get("title", f"模块{i+1}"),
                "concepts": [module.get("skill", "通用技能")],
                "difficulty": 0.6,
                "duration_hours": module.get("estimated_minutes", 60) / 60.0,
                "content_type": module.get("content_type", "lecture")
            } for i, module in enumerate(content_recommendation.content_modules[:5])]  # 限制模块数量
            
            adjustment_strategies = [{
                "strategy_type": "add_content",
                "target_skills": content_recommendation.target_skills[:3]
            }]
            
            expected_outcomes = {
                skill: 0.8 for skill in content_recommendation.target_skills
            }
            
            # 执行模拟
            result = self.simulation_agent.run_simulation_cycle(
                learner_profiles, course_modules, adjustment_strategies, expected_outcomes
            )
            
            # 提取关键信息
            insights = result["reflection_insights"]
            
            return SimulationResult(
                simulation_id=result["cycle_id"],
                predicted_outcomes=expected_outcomes,
                success_probability=insights["strategy_effectiveness"],
                risk_factors=insights["identified_issues"][:3],
                optimization_suggestions=insights["next_iteration_recommendations"][:3],
                confidence_level=insights["confidence_level"]
            )
            
        except Exception as e:
            logger.error(f"学习效果模拟失败: {e}")
            raise
    
    def _build_response(self, job_market_data: JobMarketData,
                       learner_data: LearnerData,
                       alignment_result: AlignmentResult,
                       content_recommendation: ContentRecommendation,
                       simulation_result: SimulationResult) -> Dict[str, Any]:
        """构建完整响应"""
        
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            
            # 核心结果
            "job_market_analysis": job_market_data.to_dict(),
            "learner_profile": learner_data.to_dict(),
            "alignment_analysis": alignment_result.to_dict(),
            "content_recommendation": content_recommendation.to_dict(),
            "simulation_prediction": simulation_result.to_dict(),
            
            # 摘要信息
            "summary": {
                "target_job": job_market_data.job_title,
                "key_skill_gaps": content_recommendation.target_skills[:3],
                "learning_time_estimate": f"{content_recommendation.estimated_completion_time:.1f}小时",
                "success_probability": f"{simulation_result.success_probability:.1%}",
                "priority_actions": alignment_result.priority_actions[:2],
                "personalization_notes": content_recommendation.personalization_notes[:2]
            },
            
            # 系统信息
            "system_info": {
                "version": self.config.get("system.version"),
                "agents_used": [
                    "job_market_analyst" if self.job_analyst else None,
                    "learner_profiling" if self.learner_profiler else None,
                    "dynamic_alignment" if self.alignment_agent else None,
                    "content_generator" if self.content_generator else None,
                    "simulation_reflection" if self.simulation_agent else None
                ],
                "processing_time": "模拟处理时间"
            }
        }
        
        # 保存处理结果
        self._save_session_result(response)
        
        return response
    
    def _save_session_result(self, result: Dict[str, Any]):
        """保存会话结果"""
        try:
            output_dir = self.config.get_data_dir("base")
            output_dir.mkdir(exist_ok=True)
            
            session_file = output_dir / f"{result['session_id']}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"会话结果已保存: {session_file}")
        except Exception as e:
            logger.error(f"保存会话结果失败: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "status": "running",
            "version": self.config.get("system.version"),
            "agents": {
                "job_market_analyst": self.job_analyst is not None,
                "learner_profiling": self.learner_profiler is not None,
                "dynamic_alignment": self.alignment_agent is not None,
                "content_generator": self.content_generator is not None,
                "simulation_reflection": self.simulation_agent is not None
            },
            "config": {
                "debug_mode": self.config.get("system.debug"),
                "data_dir": str(self.config.get_data_dir()),
                "log_level": self.config.get("system.log_level")
            }
        }

# 测试代码
if __name__ == "__main__":
    print("🚀 DCAS系统协调器测试")
    print("=" * 50)
    
    # 初始化协调器
    orchestrator = DCASOrchestrator()
    
    # 测试系统状态
    status = orchestrator.get_system_status()
    print("系统状态:", json.dumps(status, ensure_ascii=False, indent=2))
    
    # 测试完整工作流
    test_request = {
        "job_info": {
            "title": "数据科学家",
            "description": "负责数据分析、机器学习模型开发，需要Python、SQL、统计学基础"
        },
        "learner_info": {
            "user_id": "test_user_001",
            "interests": ["数据分析", "机器学习"],
            "description": "有一定编程基础，希望转入数据科学领域",
            "current_skills": {"Python": 0.6, "统计学": 0.4}
        },
        "course_info": {
            "name": "数据科学基础课程"
        }
    }
    
    print("\n🔄 执行完整工作流...")
    result = orchestrator.process_learning_request(test_request)
    
    if result["success"]:
        print("✅ 工作流执行成功")
        print(f"目标职位: {result['summary']['target_job']}")
        print(f"技能缺口: {', '.join(result['summary']['key_skill_gaps'])}")
        print(f"学习时间: {result['summary']['learning_time_estimate']}")
        print(f"成功概率: {result['summary']['success_probability']}")
    else:
        print("❌ 工作流执行失败:", result.get("error"))
    
    print("\n🎯 DCAS系统协调器测试完成")