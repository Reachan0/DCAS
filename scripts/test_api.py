#!/usr/bin/env python3
"""
Qwen3就业市场分析API测试脚本
"""

import requests
import json
import sys
import os

# API配置
API_URL = "http://localhost:8000"

def test_single_analysis():
    """测试单条职位分析"""
    print("=== 单条职位分析测试 ===")
    
    # 测试数据
    test_jobs = [
        {
            "job_title": "数据科学家",
            "job_description": "负责构建机器学习模型，分析大数据，与产品团队合作，需要Python、SQL、机器学习经验，具备统计分析和数据可视化能力"
        },
        {
            "job_title": "前端开发工程师", 
            "job_description": "开发响应式Web应用，使用React/Vue框架，与UI/UX设计师协作，需要HTML/CSS/JavaScript经验，具备良好的代码规范和团队协作能力"
        },
        {
            "job_title": "产品经理",
            "job_description": "负责产品规划和需求分析，协调开发团队，制定产品路线图，需要市场调研和项目管理经验，具备用户研究和商业分析能力"
        }
    ]
    
    for job in test_jobs:
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                json=job,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {job['job_title']}")
                print(f"   技能: {result['final_skills']}")
                print(f"   成功: {result['success']}")
                print("-" * 50)
            else:
                print(f"❌ 错误: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 网络错误: {e}")
        except Exception as e:
            print(f"❌ 其他错误: {e}")

def test_batch_analysis():
    """测试批量职位分析"""
    print("\n=== 批量职位分析测试 ===")
    
    # 测试数据
    batch_jobs = [
        {
            "job_title": "后端开发工程师",
            "job_description": "负责API开发，使用Java/Spring框架，设计数据库架构，需要Redis、MySQL经验，具备高并发处理和系统优化能力"
        },
        {
            "job_title": "UI/UX设计师",
            "job_description": "负责用户界面和体验设计，使用Figma/Adobe工具，进行用户研究，需要设计思维和用户体验优化能力"
        },
        {
            "job_title": "DevOps工程师",
            "job_description": "负责CI/CD流程，使用Docker/Kubernetes，配置云服务，需要Linux、云平台和自动化部署经验"
        }
    ]
    
    try:
        response = requests.post(
            f"{API_URL}/analyze/batch",
            json={"jobs": batch_jobs},
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 批量处理完成")
            print(f"   总数: {result['count']}")
            print(f"   耗时: {result['processing_time']:.2f}秒")
            print()
            
            for i, r in enumerate(result['results'], 1):
                print(f"{i}. {r['job_title']}")
                print(f"   技能: {r['final_skills']}")
                print()
        else:
            print(f"❌ 批量错误: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 网络错误: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {e}")

def test_health_check():
    """测试健康检查"""
    print("\n=== 健康检查测试 ===")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 服务状态: {result['status']}")
            print(f"   模型已加载: {result['model_loaded']}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 健康检查错误: {e}")

def test_info():
    """测试系统信息"""
    print("\n=== 系统信息测试 ===")
    
    try:
        response = requests.get(f"{API_URL}/info", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 服务信息: {result['service']}")
            print(f"   版本: {result['version']}")
            print(f"   可用端点: {', '.join(result['endpoints'].keys())}")
        else:
            print(f"❌ 信息获取失败: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 信息获取错误: {e}")

def save_results(results, filename):
    """保存结果到文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✅ 结果已保存到: {filename}")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")

def main():
    """主函数"""
    print("🚀 Qwen3就业市场分析API测试工具")
    print(f"API地址: {API_URL}")
    print("-" * 60)
    
    # 检查服务是否可用
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code != 200:
            print("❌ API服务未启动或不可访问")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 无法连接API服务: {e}")
        print("请确保服务已启动: python qwen3_api_server.py")
        sys.exit(1)
    
    # 运行所有测试
    test_health_check()
    test_info()
    test_single_analysis()
    test_batch_analysis()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！")
    print("📖 详细文档: http://localhost:8000/docs")

if __name__ == "__main__":
    # 设置模型路径（可选）
    os.environ["QWEN3_MODEL_PATH"] = "/Users/xuanchong/projects/models/Qwen3-14B-SFT"
    main()