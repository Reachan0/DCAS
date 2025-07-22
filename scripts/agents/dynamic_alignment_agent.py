#!/usr/bin/env python3
"""
动态对齐策略智能体 (Dynamic Alignment Strategy Agent)
系统的核心决策引擎，负责制定课程对齐策略
"""

import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketDemand:
    """市场需求数据结构"""
    job_title: str
    required_skills: List[str]
    skill_importance: Dict[str, float]  # skill -> importance score (0-1)
    market_trend: str  # growing, stable, declining
    salary_range: Dict[str, float]  # min, max, avg

@dataclass
class CourseStatus:
    """课程现状数据结构"""
    course_id: str
    course_name: str
    covered_concepts: List[str]
    skill_coverage: Dict[str, float]  # skill -> coverage score (0-1)
    difficulty_level: str  # beginner, intermediate, advanced
    student_enrollment: int
    completion_rate: float

@dataclass
class LearnerSnapshot:
    """学习者快照数据结构"""
    user_id: str
    knowledge_mastery: Dict[str, str]  # concept -> mastered/learning/not_started
    learning_style: Dict[str, str]
    behavioral_patterns: Dict[str, Any]
    skill_gaps: List[str]  # identified skill gaps

@dataclass
class AlignmentIssue:
    """对齐问题数据结构"""
    issue_type: str  # skill_gap, content_mismatch, difficulty_mismatch
    description: str
    severity: str  # high, medium, low
    affected_skills: List[str]
    evidence: Dict[str, Any]

@dataclass
class AdjustmentStrategy:
    """调整策略数据结构"""
    strategy_id: str
    strategy_type: str  # add_content, modify_content, remove_content, reorder_content
    target_skills: List[str]
    action_items: List[str]
    expected_outcome: str
    implementation_priority: str  # high, medium, low
    estimated_effort: str  # hours, days, weeks
    risk_level: str  # low, medium, high

@dataclass
class PTOTNode:
    """PTOT思维树节点"""
    node_id: str
    level: str  # problem, thinking, option, tactic
    content: str
    parent_id: Optional[str] = None
    children: List[str] = None
    confidence: float = 0.8
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class PTOTReasoningEngine:
    """PTOT思维树推理引擎"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.tree = {}
        self.root_id = "root"
    
    def _build_ptot_prompt(self, market_demand: MarketDemand, 
                          course_status: CourseStatus, 
                          learner_snapshot: LearnerSnapshot) -> str:
        """构建PTOT推理Prompt"""
        return f"""你是一个专业的课程设计专家，需要基于以下三方信息制定课程对齐策略。

【市场需求】:
职位: {market_demand.job_title}
核心技能需求: {', '.join(market_demand.required_skills[:5])}
技能重要性: {json.dumps({k:v for k,v in list(market_demand.skill_importance.items())[:5]}, ensure_ascii=False)}
市场趋势: {market_demand.market_trend}

【课程现状】:
课程名称: {course_status.course_name}
已覆盖概念: {', '.join(course_status.covered_concepts[:5])}
技能覆盖率: {json.dumps({k:v for k,v in list(course_status.skill_coverage.items())[:5]}, ensure_ascii=False)}
难度等级: {course_status.difficulty_level}
学生人数: {course_status.student_enrollment}
完成率: {course_status.completion_rate}

【学习者状态】:
用户ID: {learner_snapshot.user_id}
知识掌握: {json.dumps({k:v for k,v in list(learner_snapshot.knowledge_mastery.items())[:5]}, ensure_ascii=False)}
学习风格: {json.dumps(learner_snapshot.learning_style, ensure_ascii=False)}
已识别技能缺口: {', '.join(learner_snapshot.skill_gaps[:5])}

请使用PTOT思维树结构分析：
1. 问题层 (Problem): 识别主要对齐问题
2. 思考层 (Thinking): 分析问题原因和影响
3. 选项层 (Option): 提出可能的调整方案
4. 策略层 (Tactic): 制定具体执行策略

请严格按照以下JSON格式返回：
{{
  "tree": [
    {{
      "node_id": "problem_1",
      "level": "problem",
      "content": "主要问题描述",
      "children": ["thinking_1", "thinking_2"]
    }},
    {{
      "node_id": "thinking_1",
      "level": "thinking",
      "content": "问题原因分析",
      "parent_id": "problem_1",
      "children": ["option_1", "option_2"]
    }},
    {{
      "node_id": "option_1",
      "level": "option",
      "content": "调整方案描述",
      "parent_id": "thinking_1",
      "children": ["tactic_1"]
    }},
    {{
      "node_id": "tactic_1",
      "level": "tactic",
      "content": "具体执行策略",
      "parent_id": "option_1"
    }}
  ]
}}"""

    def _parse_ptot_response(self, response: str) -> List[PTOTNode]:
        """解析PTOT响应"""
        try:
            data = json.loads(response.strip())
            nodes = []
            for node_data in data.get("tree", []):
                nodes.append(PTOTNode(**node_data))
            return nodes
        except Exception as e:
            logger.error(f"PTOT解析失败: {e}")
            return []

class ThreeWayAnalyzer:
    """三方信息比较分析器"""
    
    def analyze_alignment(self, market_demand: MarketDemand, 
                         course_status: CourseStatus, 
                         learner_snapshot: LearnerSnapshot) -> List[AlignmentIssue]:
        """分析三方信息的对齐问题"""
        issues = []
        
        # 1. 技能缺口分析
        required_skills = set(market_demand.required_skills)
        covered_skills = set(course_status.skill_coverage.keys())
        learner_skills = set(learner_snapshot.knowledge_mastery.keys())
        
        missing_skills = required_skills - covered_skills
        if missing_skills:
            issues.append(AlignmentIssue(
                issue_type="skill_gap",
                description=f"课程缺少市场必需技能: {', '.join(list(missing_skills)[:3])}",
                severity="high" if len(missing_skills) > 3 else "medium",
                affected_skills=list(missing_skills),
                evidence={"market_required": list(required_skills), "course_covered": list(covered_skills)}
            ))
        
        # 2. 难度匹配分析
        market_level = self._get_market_level(market_demand.job_title)
        course_level = course_status.difficulty_level
        
        level_mapping = {"beginner": 1, "intermediate": 2, "advanced": 3}
        if abs(level_mapping.get(market_level, 2) - level_mapping.get(course_level, 2)) > 1:
            issues.append(AlignmentIssue(
                issue_type="difficulty_mismatch",
                description=f"课程难度({course_level})与市场需求({market_level})不匹配",
                severity="medium",
                affected_skills=market_demand.required_skills,
                evidence={"market_level": market_level, "course_level": course_level}
            ))
        
        # 3. 学习者个性化缺口
        for skill_gap in learner_snapshot.skill_gaps:
            if skill_gap in required_skills:
                issues.append(AlignmentIssue(
                    issue_type="personalized_gap",
                    description=f"学习者个人技能缺口: {skill_gap}",
                    severity="high",
                    affected_skills=[skill_gap],
                    evidence={"learner_gap": skill_gap, "market_required": True}
                ))
        
        return issues
    
    def _get_market_level(self, job_title: str) -> str:
        """根据职位判断市场级别"""
        junior_keywords = ["junior", "初级", "entry", "助理"]
        senior_keywords = ["senior", "高级", "资深", "lead", "专家", "资深"]
        
        job_lower = job_title.lower()
        if any(keyword in job_lower for keyword in junior_keywords):
            return "beginner"
        elif any(keyword in job_lower for keyword in senior_keywords):
            return "advanced"
        else:
            return "intermediate"

class StrategyGenerator:
    """课程调整策略生成器"""
    
    def generate_strategies(self, issues: List[AlignmentIssue], 
                          market_demand: MarketDemand, 
                          course_status: CourseStatus) -> List[AdjustmentStrategy]:
        """基于对齐问题生成调整策略"""
        strategies = []
        
        for issue in issues:
            if issue.issue_type == "skill_gap":
                strategies.extend(self._generate_skill_gap_strategies(issue, market_demand))
            elif issue.issue_type == "difficulty_mismatch":
                strategies.extend(self._generate_difficulty_strategies(issue))
            elif issue.issue_type == "personalized_gap":
                strategies.extend(self._generate_personalized_strategies(issue))
        
        return strategies
    
    def _generate_skill_gap_strategies(self, issue: AlignmentIssue, 
                                     market_demand: MarketDemand) -> List[AdjustmentStrategy]:
        """生成技能缺口解决策略"""
        strategies = []
        
        for skill in issue.affected_skills[:3]:  # 限制前3个技能
            importance = market_demand.skill_importance.get(skill, 0.5)
            
            strategies.append(AdjustmentStrategy(
                strategy_id=f"add_skill_{skill.replace(' ', '_')}",
                strategy_type="add_content",
                target_skills=[skill],
                action_items=[
                    f"添加{skill}基础概念讲解",
                    f"设计{skill}实践项目",
                    f"创建{skill}技能评估"
                ],
                expected_outcome=f"提升{skill}技能覆盖率至85%以上",
                implementation_priority="high" if importance > 0.8 else "medium",
                estimated_effort="2-3周",
                risk_level="low"
            ))
        
        return strategies
    
    def _generate_difficulty_strategies(self, issue: AlignmentIssue) -> List[AdjustmentStrategy]:
        """生成难度调整策略"""
        strategies = []
        
        if "初级" in issue.description:
            strategies.append(AdjustmentStrategy(
                strategy_id="increase_difficulty",
                strategy_type="modify_content",
                target_skills=issue.affected_skills,
                action_items=[
                    "增加高级案例和项目",
                    "引入复杂问题场景",
                    "添加性能优化内容"
                ],
                expected_outcome="将课程难度提升至中级水平",
                implementation_priority="medium",
                estimated_effort="3-4周",
                risk_level="medium"
            ))
        else:
            strategies.append(AdjustmentStrategy(
                strategy_id="decrease_difficulty",
                strategy_type="modify_content",
                target_skills=issue.affected_skills,
                action_items=[
                    "简化复杂概念",
                    "增加基础示例",
                    "提供更多练习题"
                ],
                expected_outcome="将课程难度调整至合适水平",
                implementation_priority="medium",
                estimated_effort="2-3周",
                risk_level="low"
            ))
        
        return strategies
    
    def _generate_personalized_strategies(self, issue: AlignmentIssue) -> List[AdjustmentStrategy]:
        """生成个性化策略"""
        strategies = []
        
        for skill in issue.affected_skills:
            strategies.append(AdjustmentStrategy(
                strategy_id=f"personalize_{skill.replace(' ', '_')}",
                strategy_type="add_content",
                target_skills=[skill],
                action_items=[
                    f"为{skill}创建个性化学习路径",
                    f"设计针对{skill}的补救练习",
                    f"提供{skill}的进阶挑战"
                ],
                expected_outcome=f"帮助学习者掌握{skill}",
                implementation_priority="high",
                estimated_effort="1-2周",
                risk_level="low"
            ))
        
        return strategies

class DynamicAlignmentAgent:
    """动态对齐策略智能体主类"""
    
    def __init__(self, llm_client=None, storage_path: str = "alignment_strategies"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.ptot_engine = PTOTReasoningEngine(llm_client)
        self.analyzer = ThreeWayAnalyzer()
        self.strategy_generator = StrategyGenerator()
    
    def analyze_and_plan(self, market_demand: MarketDemand, 
                        course_status: CourseStatus, 
                        learner_snapshot: LearnerSnapshot) -> Dict[str, Any]:
        """分析三方信息并制定对齐策略"""
        logger.info(f"开始分析课程对齐: {course_status.course_name} vs {market_demand.job_title}")
        
        # 1. 分析对齐问题
        issues = self.analyzer.analyze_alignment(market_demand, course_status, learner_snapshot)
        
        # 2. 生成调整策略
        strategies = self.strategy_generator.generate_strategies(issues, market_demand, course_status)
        
        # 3. PTOT推理（可选）
        ptot_nodes = []
        if self.ptot_engine.llm_client:
            try:
                prompt = self.ptot_engine._build_ptot_prompt(market_demand, course_status, learner_snapshot)
                response = self.ptot_engine.llm_client.generate(prompt)
                ptot_nodes = self.ptot_engine._parse_ptot_response(response)
            except Exception as e:
                logger.warning(f"PTOT推理失败: {e}")
        
        # 4. 构建完整报告
        report = {
            "analysis_id": f"alignment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "market_demand": asdict(market_demand),
            "course_status": asdict(course_status),
            "learner_snapshot": asdict(learner_snapshot),
            "alignment_issues": [asdict(issue) for issue in issues],
            "adjustment_strategies": [asdict(strategy) for strategy in strategies],
            "ptot_reasoning": [asdict(node) for node in ptot_nodes] if ptot_nodes else []
        }
        
        # 5. 保存报告
        self._save_report(report)
        
        logger.info(f"对齐分析完成，发现 {len(issues)} 个问题，生成 {len(strategies)} 个策略")
        return report
    
    def _save_report(self, report: Dict[str, Any]):
        """保存分析报告"""
        file_path = self.storage_path / f"{report['analysis_id']}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
    def get_strategy_summary(self, report: Dict[str, Any]) -> str:
        """生成策略摘要"""
        issues = report["alignment_issues"]
        strategies = report["adjustment_strategies"]
        
        summary = f"""课程对齐策略报告
日期: {report['timestamp'][:10]}
课程: {report['course_status']['course_name']}
目标岗位: {report['market_demand']['job_title']}

发现问题: {len(issues)} 个
调整策略: {len(strategies)} 个

优先级策略:"""
        
        high_priority = [s for s in strategies if s["implementation_priority"] == "high"]
        for strategy in high_priority[:3]:
            summary += f"\n• [{strategy['strategy_type']}] {strategy['expected_outcome']}"
        
        return summary

class MockLLMClient:
    """模拟LLM客户端，用于测试"""
    
    def generate(self, prompt: str) -> str:
        """模拟PTOT推理响应"""
        return json.dumps({
            "tree": [
                {
                    "node_id": "problem_1",
                    "level": "problem",
                    "content": "课程与市场需求存在技能缺口",
                    "children": ["thinking_1"]
                },
                {
                    "node_id": "thinking_1",
                    "level": "thinking",
                    "content": "缺少数据分析和机器学习相关技能",
                    "parent_id": "problem_1",
                    "children": ["option_1"]
                },
                {
                    "node_id": "option_1",
                    "level": "option",
                    "content": "添加相关技能模块",
                    "parent_id": "thinking_1",
                    "children": ["tactic_1"]
                },
                {
                    "node_id": "tactic_1",
                    "level": "tactic",
                    "content": "增加2周的数据分析实战项目",
                    "parent_id": "option_1"
                }
            ]
        })

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    market_demand = MarketDemand(
        job_title="数据科学家",
        required_skills=["Python", "机器学习", "数据分析", "SQL", "统计学"],
        skill_importance={
            "Python": 0.9,
            "机器学习": 0.95,
            "数据分析": 0.85,
            "SQL": 0.8,
            "统计学": 0.75
        },
        market_trend="growing",
        salary_range={"min": 15000, "max": 30000, "avg": 22000}
    )
    
    course_status = CourseStatus(
        course_id="CS101",
        course_name="Python编程基础",
        covered_concepts=["Python语法", "数据结构", "函数", "类与对象"],
        skill_coverage={
            "Python": 0.8,
            "机器学习": 0.1,
            "数据分析": 0.3,
            "SQL": 0.0,
            "统计学": 0.2
        },
        difficulty_level="beginner",
        student_enrollment=100,
        completion_rate=0.85
    )
    
    learner_snapshot = LearnerSnapshot(
        user_id="learner_001",
        knowledge_mastery={
            "Python语法": "mastered",
            "数据结构": "learning",
            "机器学习": "not_started"
        },
        learning_style={"processing": "active", "perception": "sensory"},
        behavioral_patterns={"activity_level": "high", "engagement_score": 0.8},
        skill_gaps=["机器学习", "高级数据分析", "SQL"]
    )
    
    # 创建智能体并分析
    llm_client = MockLLMClient()
    agent = DynamicAlignmentAgent(llm_client)
    
    report = agent.analyze_and_plan(market_demand, course_status, learner_snapshot)
    
    print("🎯 动态对齐策略分析报告")
    print("=" * 50)
    print(agent.get_strategy_summary(report))
    
    # 保存完整报告
    with open("alignment_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 完整报告已保存: alignment_report.json")