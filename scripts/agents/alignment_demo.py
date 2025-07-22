#!/usr/bin/env python3
"""
动态对齐策略智能体演示
展示完整的三方信息分析和策略制定流程
"""

import json
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.agents.dynamic_alignment_agent import (
    DynamicAlignmentAgent, MarketDemand, CourseStatus, LearnerSnapshot, MockLLMClient
)

def demo():
    """演示完整的三方对齐分析流程"""
    print("🎯 动态对齐策略智能体演示")
    print("=" * 60)
    
    # 创建智能体实例
    llm_client = MockLLMClient()
    agent = DynamicAlignmentAgent(llm_client, storage_path="demo_alignment")
    
    # 场景1: 数据科学课程对齐分析
    print("\n📊 场景1: 数据科学课程对齐分析")
    print("-" * 40)
    
    # 1. 市场需求
    market_demand = MarketDemand(
        job_title="高级数据科学家",
        required_skills=[
            "Python编程", "机器学习", "深度学习", "SQL数据库", 
            "统计学", "数据可视化", "A/B测试", "云计算", "大数据处理"
        ],
        skill_importance={
            "Python编程": 0.95, "机器学习": 0.9, "深度学习": 0.85,
            "SQL数据库": 0.8, "统计学": 0.85, "数据可视化": 0.75,
            "A/B测试": 0.7, "云计算": 0.65, "大数据处理": 0.8
        },
        market_trend="growing",
        salary_range={"min": 20000, "max": 40000, "avg": 28000}
    )
    
    # 2. 课程现状
    course_status = CourseStatus(
        course_id="DS101",
        course_name="数据科学基础课程",
        covered_concepts=[
            "Python基础语法", "Pandas数据处理", "Numpy数值计算",
            "Matplotlib可视化", "基础统计学", "线性回归"
        ],
        skill_coverage={
            "Python编程": 0.8, "机器学习": 0.3, "深度学习": 0.1,
            "SQL数据库": 0.4, "统计学": 0.7, "数据可视化": 0.6,
            "A/B测试": 0.2, "云计算": 0.1, "大数据处理": 0.2
        },
        difficulty_level="intermediate",
        student_enrollment=150,
        completion_rate=0.78
    )
    
    # 3. 学习者状态
    learner_snapshot = LearnerSnapshot(
        user_id="learner_cohort_2024",
        knowledge_mastery={
            "Python编程": "mastered", "统计学": "learning", 
            "机器学习": "learning", "SQL数据库": "not_started",
            "深度学习": "not_started", "数据可视化": "learning"
        },
        learning_style={
            "processing": "active", 
            "perception": "sensory", 
            "understanding": "sequential"
        },
        behavioral_patterns={
            "activity_level": "high",
            "engagement_score": 0.85,
            "preferred_content_type": ["案例研究", "实践项目", "互动练习"]
        },
        skill_gaps=["深度学习", "A/B测试", "云计算", "大数据处理"]
    )
    
    # 4. 执行分析
    print("\n🔍 正在分析三方信息对齐情况...")
    report = agent.analyze_and_plan(market_demand, course_status, learner_snapshot)
    
    # 5. 展示分析结果
    print("\n📋 分析结果:")
    print("-" * 40)
    
    # 对齐问题
    issues = report["alignment_issues"]
    print(f"发现 {len(issues)} 个对齐问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. [{issue['severity'].upper()}] {issue['description']}")
    
    # 调整策略
    strategies = report["adjustment_strategies"]
    print(f"\n生成 {len(strategies)} 个调整策略:")
    
    # 按优先级分组
    high_priority = [s for s in strategies if s["implementation_priority"] == "high"]
    medium_priority = [s for s in strategies if s["implementation_priority"] == "medium"]
    
    if high_priority:
        print("\n🚨 高优先级策略:")
        for strategy in high_priority:
            print(f"   [{strategy['strategy_type']}] {strategy['expected_outcome']}")
            print(f"   📋 行动项: {', '.join(strategy['action_items'][:2])}...")
            print(f"   ⏱️ 预计时间: {strategy['estimated_effort']}")
            print()
    
    if medium_priority:
        print("⚠️ 中等优先级策略:")
        for strategy in medium_priority[:3]:  # 显示前3个
            print(f"   [{strategy['strategy_type']}] {strategy['expected_outcome']}")
    
    # 6. 生成实施建议
    print("\n🔧 实施建议:")
    print("-" * 40)
    generate_implementation_plan(strategies)
    
    # 7. 保存完整报告
    with open("alignment_analysis_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 完整分析报告已保存: alignment_analysis_report.json")

def generate_implementation_plan(strategies: list):
    """生成实施计划"""
    
    # 按时间排序
    time_mapping = {
        "1-2周": 1, "2-3周": 2, "3-4周": 3, "4-5周": 4
    }
    
    sorted_strategies = sorted(strategies, 
                              key=lambda x: time_mapping.get(x["estimated_effort"], 5))
    
    print("📅 分阶段实施计划:")
    
    phase_1 = [s for s in sorted_strategies if s["implementation_priority"] == "high"]
    phase_2 = [s for s in sorted_strategies if s["implementation_priority"] == "medium"]
    
    if phase_1:
        print("\n🎯 第一阶段 (高优先级):")
        for strategy in phase_1:
            print(f"   • {strategy['expected_outcome']} ({strategy['estimated_effort']})")
    
    if phase_2:
        print("\n⚙️ 第二阶段 (中等优先级):")
        for strategy in phase_2[:3]:  # 显示前3个
            print(f"   • {strategy['expected_outcome']} ({strategy['estimated_effort']})")
    
    # 风险评估
    high_risk = [s for s in strategies if s["risk_level"] == "high"]
    if high_risk:
        print(f"\n⚠️ 高风险策略 ({len(high_risk)}个):")
        for strategy in high_risk:
            print(f"   • {strategy['strategy_type']} - {strategy['expected_outcome']}")

def demo_different_scenarios():
    """演示不同场景"""
    print("\n🎭 不同场景对比分析")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "初学者课程",
            "market": "初级数据分析师",
            "course_level": "beginner",
            "learner_gaps": ["统计学基础", "Python语法"]
        },
        {
            "name": "进阶课程", 
            "market": "高级机器学习工程师",
            "course_level": "intermediate",
            "learner_gaps": ["深度学习", "模型部署", "MLOps"]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}分析:")
        
        # 创建简化的分析结果
        market_demand = MarketDemand(
            job_title=scenario["market"],
            required_skills=["Python", "SQL", "机器学习", "统计学"],
            skill_importance={"Python": 0.9, "SQL": 0.8, "机器学习": 0.85, "统计学": 0.75},
            market_trend="growing",
            salary_range={"min": 15000, "max": 35000, "avg": 22000}
        )
        
        course_status = CourseStatus(
            course_id=f"{scenario['name']}_001",
            course_name=scenario["name"],
            covered_concepts=["基础概念"],
            skill_coverage={"Python": 0.6, "SQL": 0.4, "机器学习": 0.2, "统计学": 0.5},
            difficulty_level=scenario["course_level"],
            student_enrollment=100,
            completion_rate=0.8
        )
        
        learner_snapshot = LearnerSnapshot(
            user_id="demo_learner",
            knowledge_mastery={},
            learning_style={"processing": "active"},
            behavioral_patterns={"engagement_score": 0.8},
            skill_gaps=scenario["learner_gaps"]
        )
        
        # 快速分析
        from scripts.agents.dynamic_alignment_agent import ThreeWayAnalyzer
        analyzer = ThreeWayAnalyzer()
        issues = analyzer.analyze_alignment(market_demand, course_status, learner_snapshot)
        
        print(f"   发现问题: {len(issues)} 个")
        for issue in issues:
            print(f"   • {issue.description[:50]}...")

if __name__ == "__main__":
    demo()
    demo_different_scenarios()