#!/usr/bin/env python3
"""
测试JSON格式解析是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.agents.job_market_analyst_agent import JobMarketAnalystAgent

def test_json_parsing():
    """测试JSON格式解析"""
    print("🧪 测试JSON格式解析...")
    
    # 模拟当前模型输出的JSON格式
    json_response = '''```json
{
  "skills": [
    {
      "name": "Python",
      "type": "Programming Language"
    },
    {
      "name": "SQL",
      "type": "Database Query Language"
    },
    {
      "name": "机器学习",
      "type": "Technical Skill"
    }
  ]
}
```'''
    
    agent = JobMarketAnalystAgent()
    parsed_skills = agent._parse_final_skills(json_response)
    
    print(f"📝 输入JSON:\n{json_response}")
    print(f"\n🎯 解析结果: {parsed_skills}")
    
    expected_skills = ["Python", "SQL", "机器学习"]
    if all(skill in parsed_skills for skill in expected_skills):
        print("✅ JSON解析成功")
        return True
    else:
        print("❌ JSON解析失败")
        return False

def test_natural_language_parsing():
    """测试自然语言格式解析"""
    print("\n🧪 测试自然语言格式解析...")
    
    # 模拟期望的自然语言输出
    natural_response = """第一步 - 职位类型识别:
这是一个数据科学技术职位。

第二步 - 技术技能深度分析:
需要Python编程和SQL数据库技能。

第三步 - 业务技能映射:
需要数据分析能力。

第四步 - 软技能与协作要求:
需要团队协作能力。

第五步 - 技能优先级排序和最终清单:
核心技能: Python, SQL, 数据分析, 机器学习, 团队协作"""
    
    agent = JobMarketAnalystAgent()
    parsed_skills = agent._parse_final_skills(natural_response)
    
    print(f"📝 输入自然语言:\n{natural_response}")
    print(f"\n🎯 解析结果: {parsed_skills}")
    
    expected_skills = ["Python", "SQL", "数据分析"]
    if any(skill in parsed_skills for skill in expected_skills):
        print("✅ 自然语言解析成功")
        return True
    else:
        print("❌ 自然语言解析失败")
        return False

if __name__ == "__main__":
    print("🚀 测试解析功能...")
    print("=" * 60)
    
    # 测试两种格式的解析
    json_ok = test_json_parsing()
    natural_ok = test_natural_language_parsing()
    
    print("\n" + "=" * 60)
    print("📊 解析测试结果:")
    print(f"✅ JSON格式: {'通过' if json_ok else '失败'}")
    print(f"✅ 自然语言: {'通过' if natural_ok else '失败'}")
    
    if json_ok:
        print("\n🎉 当前JSON输出可以正确解析！")
        print("💡 建议:")
        print("1. 重启API服务器测试新的配置")
        print("2. 观察模型是否输出更多的思维链内容")
        print("3. 如果仍输出JSON，解析功能已就绪")
    else:
        print("\n⚠️ 需要进一步调试解析逻辑")