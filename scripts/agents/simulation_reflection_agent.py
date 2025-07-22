#!/usr/bin/env python3
"""
模拟与反思智能体 (Simulation & Reflection Agent)
基于数字孪生和TIR(可迁移迭代反思)的课程效果预测与优化系统
"""

import json
import logging
import random
import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VirtualStudent:
    """虚拟学生数字孪生"""
    learner_id: str
    knowledge_state: Dict[str, float]  # concept -> mastery level (0-1)
    learning_style: Dict[str, str]
    behavioral_patterns: Dict[str, Any]
    engagement_factor: float  # 0.8-1.2 学习投入系数
    difficulty_tolerance: float  # 0.7-1.3 难度容忍度
    
    def __post_init__(self):
        # 确保数值在合理范围内
        self.engagement_factor = max(0.8, min(1.2, self.engagement_factor))
        self.difficulty_tolerance = max(0.7, min(1.3, self.difficulty_tolerance))

@dataclass
class CourseModule:
    """课程模块"""
    module_id: str
    title: str
    concepts: List[str]
    difficulty: float  # 0-1
    duration_hours: float
    content_type: str  # lecture, exercise, project, case_study
    
@dataclass
class SimulationConfig:
    """模拟配置"""
    num_virtual_students: int = 100
    simulation_duration_days: int = 30
    assessment_frequency: int = 7  # 每7天评估一次
    noise_factor: float = 0.1  # 模拟现实中的不确定性

@dataclass
class SimulationResult:
    """模拟结果"""
    simulation_id: str
    virtual_students: List[VirtualStudent]
    course_progress: Dict[str, Any]
    skill_achievements: Dict[str, float]
    engagement_metrics: Dict[str, float]
    completion_rates: Dict[str, float]
    predicted_outcomes: List[Dict[str, Any]]

@dataclass
class ReflectionInsights:
    """反思洞察"""
    insights_id: str
    strategy_effectiveness: float  # 0-1
    identified_issues: List[str]
    optimization_suggestions: List[Dict[str, Any]]
    confidence_level: float
    next_iteration_recommendations: List[str]

class VirtualStudentGenerator:
    """虚拟学生生成器"""
    
    def __init__(self):
        self.base_profiles = [
            {
                "learning_style": {"processing": "active", "perception": "sensory", "understanding": "sequential"},
                "behavioral_patterns": {"activity_level": "high", "engagement_score": 0.9},
                "engagement_factor": 1.1,
                "difficulty_tolerance": 1.0
            },
            {
                "learning_style": {"processing": "reflective", "perception": "intuitive", "understanding": "global"},
                "behavioral_patterns": {"activity_level": "medium", "engagement_score": 0.7},
                "engagement_factor": 0.9,
                "difficulty_tolerance": 1.2
            },
            {
                "learning_style": {"processing": "active", "perception": "intuitive", "understanding": "sequential"},
                "behavioral_patterns": {"activity_level": "low", "engagement_score": 0.6},
                "engagement_factor": 0.8,
                "difficulty_tolerance": 0.8
            }
        ]
    
    def generate_virtual_students(self, learner_profiles: List[Dict[str, Any]], 
                                config: SimulationConfig) -> List[VirtualStudent]:
        """基于真实学习者画像生成虚拟学生"""
        virtual_students = []
        
        for i, real_profile in enumerate(learner_profiles):
            # 创建基于真实画像的虚拟学生
            knowledge_state = {}
            for concept, mastery in real_profile.get("knowledge_mastery", {}).items():
                if mastery == "mastered":
                    knowledge_state[concept] = 0.9 + random.uniform(-0.1, 0.1)
                elif mastery == "learning":
                    knowledge_state[concept] = 0.5 + random.uniform(-0.2, 0.2)
                else:
                    knowledge_state[concept] = 0.1 + random.uniform(-0.1, 0.1)
            
            # 添加个体差异
            base_profile = random.choice(self.base_profiles)
            
            virtual_student = VirtualStudent(
                learner_id=f"virtual_{real_profile.get('user_id', f'learner_{i}')}_{i}",
                knowledge_state=knowledge_state,
                learning_style=real_profile.get("learning_style", base_profile["learning_style"]),
                behavioral_patterns=real_profile.get("behavioral_patterns", base_profile["behavioral_patterns"]),
                engagement_factor=base_profile["engagement_factor"] + random.uniform(-0.1, 0.1),
                difficulty_tolerance=base_profile["difficulty_tolerance"] + random.uniform(-0.1, 0.1)
            )
            
            virtual_students.append(virtual_student)
        
        # 补充随机生成的学生以达到配置数量
        while len(virtual_students) < config.num_virtual_students:
            base = random.choice(self.base_profiles)
            virtual_student = VirtualStudent(
                learner_id=f"virtual_synthetic_{len(virtual_students)}",
                knowledge_state={},
                learning_style=base["learning_style"],
                behavioral_patterns=base["behavioral_patterns"],
                engagement_factor=base["engagement_factor"],
                difficulty_tolerance=base["difficulty_tolerance"]
            )
            virtual_students.append(virtual_student)
        
        return virtual_students

class LearningSimulationEngine:
    """学习模拟引擎"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def calculate_learning_rate(self, student: VirtualStudent, module: CourseModule) -> float:
        """计算学习速率"""
        # 基于学习风格和模块特性的学习速率计算
        style_bonus = 1.0
        
        # 学习风格匹配度
        if student.learning_style["processing"] == "active" and module.content_type in ["exercise", "project"]:
            style_bonus *= 1.2
        elif student.learning_style["processing"] == "reflective" and module.content_type == "lecture":
            style_bonus *= 1.15
        
        # 难度适应度
        difficulty_factor = 1.0 - abs(module.difficulty - 0.5) * 0.5
        
        # 综合学习速率
        base_rate = 0.1
        learning_rate = base_rate * student.engagement_factor * style_bonus * difficulty_factor
        
        # 添加随机噪声
        learning_rate *= (1 + random.uniform(-self.config.noise_factor, self.config.noise_factor))
        
        return max(0.05, min(0.3, learning_rate))
    
    def simulate_dropout_risk(self, student: VirtualStudent, cumulative_difficulty: float) -> float:
        """模拟辍学风险"""
        # 基于难度累积和学习者容忍度的辍学概率
        risk_threshold = 1.0 - student.difficulty_tolerance
        risk = max(0, (cumulative_difficulty - risk_threshold) * 0.5)
        
        # 参与度影响
        risk *= (2.0 - student.engagement_factor)
        
        return min(0.9, max(0.0, risk))
    
    def run_simulation(self, virtual_students: List[VirtualStudent], 
                      course_modules: List[CourseModule]) -> SimulationResult:
        """运行完整模拟"""
        logger.info(f"开始模拟 {len(virtual_students)} 名虚拟学生的学习过程")
        
        # 初始化结果
        simulation_id = f"sim_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        progress_tracker = {}
        skill_achievements = {}
        engagement_metrics = {"avg_engagement": 0.0, "dropout_rate": 0.0}
        
        # 为每个学生初始化进度
        for student in virtual_students:
            progress_tracker[student.learner_id] = {
                "current_module": 0,
                "completed_modules": [],
                "skill_progress": {},
                "cumulative_difficulty": 0.0,
                "active": True,
                "engagement_history": []
            }
            
            # 初始化技能进度
            for module in course_modules:
                for concept in module.concepts:
                    if concept not in progress_tracker[student.learner_id]["skill_progress"]:
                        progress_tracker[student.learner_id]["skill_progress"][concept] = {
                            "mastery": student.knowledge_state.get(concept, 0.0),
                            "last_update": 0
                        }
        
        # 模拟学习过程
        total_days = self.config.simulation_duration_days
        assessment_points = list(range(0, total_days, self.config.assessment_frequency))
        predicted_outcomes = []
        
        for day in range(total_days):
            daily_outcomes = {"day": day, "progress": {}, "events": []}
            
            for student in virtual_students:
                student_id = student.learner_id
                tracker = progress_tracker[student_id]
                
                if not tracker["active"]:
                    continue
                
                # 检查当前模块
                if tracker["current_module"] < len(course_modules):
                    current_module = course_modules[tracker["current_module"]]
                    
                    # 计算学习进度
                    learning_rate = self.calculate_learning_rate(student, current_module)
                    
                    # 更新技能掌握度
                    for concept in current_module.concepts:
                        old_mastery = tracker["skill_progress"][concept]["mastery"]
                        new_mastery = min(1.0, old_mastery + learning_rate * (current_module.duration_hours / 24))
                        tracker["skill_progress"][concept]["mastery"] = new_mastery
                        tracker["skill_progress"][concept]["last_update"] = day
                    
                    # 检查模块完成
                    module_concepts = current_module.concepts
                    avg_mastery = sum(tracker["skill_progress"][c]["mastery"] for c in module_concepts) / len(module_concepts)
                    
                    if avg_mastery >= 0.8:  # 80%掌握度视为完成
                        tracker["completed_modules"].append(current_module.module_id)
                        tracker["current_module"] += 1
                        tracker["cumulative_difficulty"] += current_module.difficulty
                        
                        daily_outcomes["events"].append({
                            "type": "module_completed",
                            "student": student_id,
                            "module": current_module.module_id,
                            "mastery": avg_mastery
                        })
                    
                    # 检查辍学风险
                    dropout_risk = self.simulate_dropout_risk(student, tracker["cumulative_difficulty"])
                    if random.random() < dropout_risk:
                        tracker["active"] = False
                        daily_outcomes["events"].append({
                            "type": "student_dropout",
                            "student": student_id,
                            "reason": "difficulty_overwhelming"
                        })
            
            # 记录每日结果
            if day in assessment_points:
                active_students = [s for s in virtual_students if progress_tracker[s.learner_id]["active"]]
                completion_rates = {}
                
                for module in course_modules:
                    completed_count = sum(1 for s in virtual_students 
                                        if module.module_id in progress_tracker[s.learner_id]["completed_modules"])
                    completion_rates[module.module_id] = completed_count / len(virtual_students)
                
                predicted_outcomes.append({
                    "day": day,
                    "active_students": len(active_students),
                    "completion_rates": completion_rates,
                    "avg_engagement": sum(s.engagement_factor for s in active_students) / len(active_students) if active_students else 0
                })
        
        # 计算最终指标
        final_active = [s for s in virtual_students if progress_tracker[s.learner_id]["active"]]
        final_dropout_rate = (len(virtual_students) - len(final_active)) / len(virtual_students)
        
        # 计算技能成就
        for concept in set(concept for module in course_modules for concept in module.concepts):
            final_mastery = [progress_tracker[s.learner_id]["skill_progress"][concept]["mastery"] 
                           for s in virtual_students if concept in progress_tracker[s.learner_id]["skill_progress"]]
            skill_achievements[concept] = sum(final_mastery) / len(final_mastery) if final_mastery else 0
        
        return SimulationResult(
            simulation_id=simulation_id,
            virtual_students=virtual_students,
            course_progress=progress_tracker,
            skill_achievements=skill_achievements,
            engagement_metrics={
                "avg_engagement": sum(s.engagement_factor for s in final_active) / len(final_active) if final_active else 0,
                "dropout_rate": final_dropout_rate
            },
            completion_rates={
                module.module_id: sum(1 for s in virtual_students 
                                    if module.module_id in progress_tracker[s.learner_id]["completed_modules"]) / len(virtual_students)
                for module in course_modules
            },
            predicted_outcomes=predicted_outcomes
        )

class TIRReflectionEngine:
    """TIR(可迁移迭代反思)引擎"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def generate_reflection_insights(self, simulation_result: SimulationResult, 
                                   original_strategies: List[Dict[str, Any]],
                                   expected_outcomes: Dict[str, float]) -> ReflectionInsights:
        """生成TIR反思洞察"""
        
        # 分析模拟结果与预期的差异
        identified_issues = []
        optimization_suggestions = []
        
        # 1. 技能达成度分析
        for skill, achieved_mastery in simulation_result.skill_achievements.items():
            expected_mastery = expected_outcomes.get(skill, 0.8)
            if achieved_mastery < expected_mastery * 0.8:  # 低于预期80%
                identified_issues.append(f"{skill}技能达成度不足: {achieved_mastery:.2f} vs 预期{expected_mastery}")
                optimization_suggestions.append({
                    "type": "increase_focus",
                    "target": skill,
                    "suggestion": f"增加{skill}的练习难度和时长",
                    "urgency": "high"
                })
        
        # 2. 参与度分析
        if simulation_result.engagement_metrics["dropout_rate"] > 0.2:
            identified_issues.append(f"辍学率过高: {simulation_result.engagement_metrics['dropout_rate']:.2%}")
            optimization_suggestions.append({
                "type": "reduce_difficulty",
                "target": "overall_course",
                "suggestion": "降低课程整体难度梯度，增加支持材料",
                "urgency": "high"
            })
        
        # 3. 完成率分析
        low_completion_modules = [module_id for module_id, rate in simulation_result.completion_rates.items() 
                                if rate < 0.7]
        if low_completion_modules:
            identified_issues.append(f"低完成率模块: {', '.join(low_completion_modules)}")
            for module in low_completion_modules:
                optimization_suggestions.append({
                    "type": "module_restructure",
                    "target": module,
                    "suggestion": f"重新设计{module}模块的内容和节奏",
                    "urgency": "medium"
                })
        
        # 4. 策略有效性评估
        strategy_effectiveness = 1.0 - (len(identified_issues) * 0.2)  # 简单评估
        strategy_effectiveness = max(0.1, min(1.0, strategy_effectiveness))
        
        # 5. 生成下一轮迭代建议
        next_iteration_recommendations = [
            "基于模拟结果调整课程难度",
            "增加个性化学习路径",
            "优化内容类型分布",
            "建立早期预警机制"
        ]
        
        return ReflectionInsights(
            insights_id=f"reflection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategy_effectiveness=strategy_effectiveness,
            identified_issues=identified_issues,
            optimization_suggestions=optimization_suggestions,
            confidence_level=0.8,
            next_iteration_recommendations=next_iteration_recommendations
        )

class SimulationReflectionAgent:
    """模拟与反思智能体主类"""
    
    def __init__(self, llm_client=None, storage_path: str = "simulation_results"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.student_generator = VirtualStudentGenerator()
        self.simulation_engine = LearningSimulationEngine(SimulationConfig())
        self.reflection_engine = TIRReflectionEngine(llm_client)
    
    def run_simulation_cycle(self, learner_profiles: List[Dict[str, Any]], 
                           course_modules: List[Dict[str, Any]],
                           adjustment_strategies: List[Dict[str, Any]],
                           expected_outcomes: Dict[str, float]) -> Dict[str, Any]:
        """运行完整的模拟反思周期"""
        logger.info("开始模拟反思周期")
        
        # 1. 生成虚拟学生
        virtual_students = self.student_generator.generate_virtual_students(
            learner_profiles, SimulationConfig()
        )
        
        # 2. 构建课程模块
        course_modules_list = [CourseModule(**module) for module in course_modules]
        
        # 3. 应用调整策略到课程模块
        adjusted_modules = self._apply_strategies(course_modules_list, adjustment_strategies)
        
        # 4. 运行模拟
        simulation_result = self.simulation_engine.run_simulation(
            virtual_students, adjusted_modules
        )
        
        # 5. 生成反思洞察
        reflection_insights = self.reflection_engine.generate_reflection_insights(
            simulation_result, adjustment_strategies, expected_outcomes
        )
        
        # 6. 构建完整结果
        cycle_result = {
            "cycle_id": f"cycle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "simulation_result": asdict(simulation_result),
            "reflection_insights": asdict(reflection_insights),
            "recommended_next_actions": reflection_insights.next_iteration_recommendations
        }
        
        # 7. 保存结果
        self._save_cycle_result(cycle_result)
        
        logger.info(f"模拟反思周期完成，策略有效性: {reflection_insights.strategy_effectiveness:.2f}")
        return cycle_result
    
    def _apply_strategies(self, course_modules: List[CourseModule], 
                         strategies: List[Dict[str, Any]]) -> List[CourseModule]:
        """将调整策略应用到课程模块"""
        adjusted_modules = [module for module in course_modules]
        
        for strategy in strategies:
            if strategy["strategy_type"] == "add_content":
                # 添加新模块
                for target_skill in strategy["target_skills"]:
                    new_module = CourseModule(
                        module_id=f"added_{target_skill}",
                        title=f"新增{target_skill}模块",
                        concepts=[target_skill],
                        difficulty=0.6,  # 中等难度
                        duration_hours=8.0,
                        content_type="exercise"
                    )
                    adjusted_modules.append(new_module)
            
            elif strategy["strategy_type"] == "modify_content":
                # 调整现有模块难度
                for module in adjusted_modules:
                    if any(skill in module.concepts for skill in strategy["target_skills"]):
                        module.difficulty = min(1.0, module.difficulty * 0.9)  # 降低难度
        
        return adjusted_modules
    
    def _save_cycle_result(self, result: Dict[str, Any]):
        """保存周期结果"""
        file_path = self.storage_path / f"{result['cycle_id']}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def generate_improvement_report(self, cycle_result: Dict[str, Any]) -> str:
        """生成改进报告"""
        insights = cycle_result["reflection_insights"]
        
        report = f"""
模拟反思报告
=============

🎯 策略有效性: {insights['strategy_effectiveness']:.1%}
📊 发现问题: {len(insights['identified_issues'])} 个

🔍 主要问题:
"""
        
        for issue in insights["identified_issues"]:
            report += f"   • {issue}\n"
        
        report += "\n💡 优化建议:\n"
        for suggestion in insights["optimization_suggestions"][:3]:
            report += f"   • [{suggestion['urgency']}] {suggestion['suggestion']}\n"
        
        report += f"\n🔄 下一轮迭代重点:\n"
        for rec in insights["next_iteration_recommendations"]:
            report += f"   • {rec}\n"
        
        return report

class MockLLMClient:
    """模拟LLM客户端，用于测试"""
    
    def generate(self, prompt: str) -> str:
        """模拟高级反思分析"""
        return """
基于模拟结果，建议以下优化策略：
1. 深度学习模块需要降低30%难度
2. 增加更多的实践项目
3. 建立个性化学习路径
4. 实施早期干预机制
"""

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    test_learner_profiles = [
        {
            "user_id": "learner_001",
            "knowledge_mastery": {"Python": 0.8, "统计学": 0.6, "机器学习": 0.3},
            "learning_style": {"processing": "active", "perception": "sensory"},
            "behavioral_patterns": {"engagement_score": 0.85}
        },
        {
            "user_id": "learner_002", 
            "knowledge_mastery": {"Python": 0.4, "统计学": 0.7, "机器学习": 0.1},
            "learning_style": {"processing": "reflective", "perception": "intuitive"},
            "behavioral_patterns": {"engagement_score": 0.7}
        }
    ]
    
    test_course_modules = [
        {
            "module_id": "python_basics",
            "title": "Python基础",
            "concepts": ["Python语法", "数据结构"],
            "difficulty": 0.3,
            "duration_hours": 20.0,
            "content_type": "lecture"
        },
        {
            "module_id": "ml_intro",
            "title": "机器学习入门",
            "concepts": ["监督学习", "模型评估"],
            "difficulty": 0.6,
            "duration_hours": 30.0,
            "content_type": "exercise"
        }
    ]
    
    test_strategies = [
        {
            "strategy_type": "add_content",
            "target_skills": ["深度学习", "A/B测试"],
            "action_items": ["添加深度学习模块", "增加A/B测试案例"]
        }
    ]
    
    test_expected_outcomes = {
        "Python": 0.9, "机器学习": 0.8, "深度学习": 0.7, "A/B测试": 0.75
    }
    
    # 运行模拟反思
    agent = SimulationReflectionAgent(MockLLMClient())
    result = agent.run_simulation_cycle(
        test_learner_profiles,
        test_course_modules,
        test_strategies,
        test_expected_outcomes
    )
    
    print("🎯 模拟与反思智能体演示")
    print("=" * 50)
    
    # 显示结果摘要
    print(agent.generate_improvement_report(result))
    
    # 保存结果
    with open("simulation_cycle_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 完整结果已保存: simulation_cycle_result.json")