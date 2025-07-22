#!/usr/bin/env python3
"""
Qwen3就业市场分析API测试脚本 - 优化版
"""

import requests
import json
import sys
import os
import time

# API配置
API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 300  # 5分钟超时

def test_health_check():
    """测试健康检查"""
    print("\n=== 健康检查测试 ===")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 服务状态: {result['status']}")
            print(f"   模型已加载: {result['model_loaded']}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 健康检查错误: {e}")
        return False

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
            return True
        else:
            print(f"❌ 信息获取失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 信息获取错误: {e}")
        return False

def test_single_analysis():
    """测试单条职位分析"""
    print("\n=== 单条职位分析测试 ===")
    
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
    
    results = []
    
    for i, job in enumerate(test_jobs, 1):
        print(f"\n测试{i}: {job['job_title']}")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                json=job,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 分析成功 (耗时: {elapsed:.2f}s)")
                print(f"   最终技能: {result.get('final_skills', 'N/A')}")
                print(f"   成功状态: {result.get('success', False)}")
                results.append({
                    "job": job,
                    "result": result,
                    "elapsed": elapsed,
                    "success": True
                })
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ 分析失败: {error_msg}")
                results.append({
                    "job": job,
                    "error": error_msg,
                    "success": False
                })
                
        except requests.exceptions.Timeout:
            print(f"❌ 请求超时 ({REQUEST_TIMEOUT}s)")
            results.append({
                "job": job,
                "error": "timeout",
                "success": False
            })
        except Exception as e:
            print(f"❌ 其他错误: {e}")
            results.append({
                "job": job,
                "error": str(e),
                "success": False
            })
        
        # 避免请求过快
        time.sleep(1)
    
    return results

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
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_URL}/analyze/batch",
            json={"jobs": batch_jobs},
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 批量分析成功 (总耗时: {elapsed:.2f}s)")
            print(f"   处理数量: {result.get('count', 0)}")
            print(f"   处理时间: {result.get('processing_time', 0):.2f}s")
            
            for i, r in enumerate(result.get('results', []), 1):
                print(f"   {i}. {r.get('job_title', 'N/A')}")
                print(f"      技能: {r.get('final_skills', 'N/A')}")
                print(f"      耗时: {r.get('processing_time', 0):.2f}s")
            
            return {"success": True, "data": result, "elapsed": elapsed}
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"❌ 批量分析失败: {error_msg}")
            return {"success": False, "error": error_msg}
            
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时 ({REQUEST_TIMEOUT}s)")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"❌ 批量分析错误: {e}")
        return {"success": False, "error": str(e)}

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
    print("🚀 Qwen3就业市场分析API测试工具 (优化版)")
    print(f"API地址: {API_URL}")
    print(f"超时设置: {REQUEST_TIMEOUT}s")
    print("-" * 60)
    
    # 检查服务是否可用
    print("检查服务状态...")
    if not test_health_check():
        print("❌ 服务未启动或不可访问")
        print("请确保服务已启动: python qwen3_api_server.py")
        sys.exit(1)
    
    # 运行所有测试
    test_info()
    
    single_results = test_single_analysis()
    batch_result = test_batch_analysis()
    
    # 保存测试结果
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "single_analysis": single_results,
        "batch_analysis": batch_result,
        "api_url": API_URL
    }
    
    save_results(all_results, "test_results.json")
    
    # 统计结果
    successful_single = sum(1 for r in single_results if r.get("success"))
    total_single = len(single_results)
    
    print("\n" + "=" * 60)
    print("📊 测试统计")
    print(f"单条测试: {successful_single}/{total_single} 成功")
    print(f"批量测试: {'✅' if batch_result.get('success') else '❌'} 成功")
    print("✅ 所有测试完成！")
    print("📖 详细文档: http://localhost:8000/docs")
    print("📁 详细结果已保存到: test_results.json")

if __name__ == "__main__":
    # 设置模型路径（可选）
    if len(sys.argv) > 1:
        os.environ["QWEN3_MODEL_PATH"] = sys.argv[1]
    main()