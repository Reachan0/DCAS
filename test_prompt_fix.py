#!/usr/bin/env python3
"""
测试Prompt修复是否解决了1 token问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.agents.job_market_analyst_agent import JobMarketAnalystAgent, MockModelClient

def test_prompt_fix():
    """测试Prompt修复"""
    print("🧪 测试Prompt修复效果...")
    
    # 创建Mock客户端和Agent
    mock_client = MockModelClient()
    agent = JobMarketAnalystAgent(mock_client)
    
    # 测试职位
    job_title = "Python数据科学家"
    job_description = """
    我们正在寻找一位经验丰富的Python数据科学家加入我们的团队。
    主要职责包括：
    - 使用Python、Pandas、NumPy进行数据分析
    - 构建机器学习模型，使用Scikit-learn、TensorFlow
    - 数据可视化，使用Matplotlib、Seaborn
    - 与团队协作，进行需求分析和项目管理
    - SQL数据库查询和数据处理
    """
    
    print(f"📋 测试职位: {job_title}")
    print(f"📋 职位描述长度: {len(job_description)} 字符")
    
    # 执行分析
    result = agent.analyze(job_title, job_description)
    
    print("\n" + "="*60)
    print("📊 分析结果:")
    print("="*60)
    print(f"✅ 成功: {result['success']}")
    print(f"🎯 最终技能: {result['final_skills']}")
    print(f"📝 CoT响应长度: {len(result['cot_response'])} 字符")
    print("\n🧠 完整CoT响应:")
    print("-" * 50)
    print(result['cot_response'])
    print("-" * 50)
    
    # 检查是否解决了1 token问题
    if len(result['cot_response']) > 100:
        print("\n✅ Prompt修复成功！生成了完整的思维链响应")
    else:
        print("\n❌ 问题仍存在，响应过短")
    
    return result

if __name__ == "__main__":
    test_prompt_fix()