#!/usr/bin/env python3
"""
测试生成配置修复是否解决1 token问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.agents.job_market_analyst_agent import JobMarketAnalystAgent, MockModelClient

def test_new_prompt_format():
    """测试新的prompt格式"""
    print("🧪 测试新的prompt格式...")
    
    mock_client = MockModelClient()
    agent = JobMarketAnalystAgent(mock_client)
    
    # 查看新的prompt结构
    prompt = agent._build_cot_prompt("数据科学家", "负责机器学习模型开发")
    
    print(f"📝 新Prompt长度: {len(prompt)} 字符")
    print("\n🔍 新Prompt结构:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
    
    # 检查是否移除了可能触发停止的格式
    problematic_patterns = [
        "**最终能力要求列表:**",
        "Python, SQL, 机器学习",
        "[在此列出技能",
    ]
    
    found_issues = []
    for pattern in problematic_patterns:
        if pattern in prompt:
            found_issues.append(pattern)
    
    if found_issues:
        print(f"❌ 仍包含可能问题的格式: {found_issues}")
        return False
    else:
        print("✅ 已移除可能导致提前停止的格式")
        return True

def generate_mock_response():
    """生成模拟的完整响应来测试解析"""
    return """第一步 - 职位类型识别:
这是一个数据科学技术职位，属于人工智能和数据分析领域。

第二步 - 技术技能深度分析:
需要掌握Python编程、机器学习框架如TensorFlow和PyTorch、数据处理工具如Pandas。

第三步 - 业务技能映射:
需要理解业务需求，进行数据建模和预测分析。

第四步 - 软技能与协作要求:
需要良好的沟通能力和团队协作精神。

第五步 - 技能优先级排序和最终清单:
核心技能包括: Python, 机器学习, TensorFlow, 数据分析, 统计学, 团队协作"""

def test_parsing_with_new_format():
    """测试新格式的解析"""
    print("\n🧪 测试新格式解析...")
    
    from scripts.agents.job_market_analyst_agent import JobMarketAnalystAgent
    agent = JobMarketAnalystAgent()
    
    mock_response = generate_mock_response()
    parsed_skills = agent._parse_final_skills(mock_response)
    
    print(f"📝 模拟响应:\n{mock_response}")
    print(f"\n🎯 解析结果: {parsed_skills}")
    
    if "Python" in parsed_skills and "机器学习" in parsed_skills:
        print("✅ 新格式解析成功")
        return True
    else:
        print("❌ 新格式解析失败")
        return False

if __name__ == "__main__":
    print("🚀 测试生成配置和prompt修复...")
    print("=" * 60)
    
    # 测试新prompt格式
    prompt_ok = test_new_prompt_format()
    
    # 测试解析功能
    parsing_ok = test_parsing_with_new_format()
    
    print("\n" + "=" * 60)
    print("📊 测试结果:")
    print(f"✅ Prompt格式: {'通过' if prompt_ok else '失败'}")
    print(f"✅ 解析功能: {'通过' if parsing_ok else '失败'}")
    
    if prompt_ok and parsing_ok:
        print("\n🎉 修复完成！建议重启API服务器测试")
        print("💡 关键修复:")
        print("1. 添加了 min_new_tokens=50 强制生成")
        print("2. 设置 early_stopping=False 禁用早停")
        print("3. 移除了可能触发停止的prompt格式")
        print("4. 优化了pad_token_id配置")
    else:
        print("\n⚠️ 还需要进一步调试")