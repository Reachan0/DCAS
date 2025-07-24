#!/usr/bin/env python3
"""
优化版API测试脚本 - 支持重试、超时控制和性能监控
"""

import requests
import json
import sys
import os
import time
import threading
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# API配置
API_URL = "http://localhost:8001"  # 使用优化版端口
DEFAULT_TIMEOUT = 15  # 降低默认超时时间
MAX_RETRIES = 3
RETRY_DELAY = 2

class OptimizedAPITester:
    """优化版API测试器"""
    
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.session = requests.Session()
        # 设置会话级别的超时和连接池
        self.session.headers.update({"Content-Type": "application/json"})
        
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_service(self, max_wait: int = 60) -> bool:
        """等待服务可用"""
        print(f"⏳ 等待服务启动 (最多{max_wait}秒)...")
        
        for i in range(max_wait):
            if self.test_connection():
                print("✅ 服务已就绪")
                return True
            time.sleep(1)
            if i % 10 == 0:
                print(f"   等待中... {i}s")
        
        print("❌ 服务启动超时")
        return False
    
    def analyze_with_retry(self, job_title: str, job_description: str, 
                          timeout: int = DEFAULT_TIMEOUT, use_quick: bool = False) -> Optional[Dict]:
        """带重试的分析请求"""
        endpoint = "/analyze/quick" if use_quick else "/analyze"
        
        request_data = {
            "job_title": job_title,
            "job_description": job_description,
            "timeout": timeout
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                
                response = self.session.post(
                    f"{self.api_url}{endpoint}",
                    json=request_data,
                    timeout=timeout + 5  # 给HTTP请求额外时间
                )
                
                request_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    result['http_request_time'] = request_time
                    return result
                
                elif response.status_code == 408:  # 超时
                    print(f"⏰ 请求超时 (尝试 {attempt + 1}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                
                elif response.status_code == 429:  # 服务器忙
                    print(f"🔄 服务器忙 (尝试 {attempt + 1}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * 2)
                        continue
                
                else:
                    print(f"❌ HTTP错误 {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"⏰ 网络超时 (尝试 {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ 网络错误: {e}")
                return None
                
        print(f"❌ 所有重试失败")
        return None
    
    def get_system_stats(self) -> Optional[Dict]:
        """获取系统统计"""
        try:
            response = self.session.get(f"{self.api_url}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def test_performance_comparison(self):
        """性能对比测试"""
        print("\n=== 性能对比测试 ===")
        
        test_job = {
            "job_title": "数据科学家",
            "job_description": "负责机器学习模型开发，需要Python、SQL、数据分析经验"
        }
        
        # 测试快速模式
        print("🚀 测试快速模式...")
        quick_result = self.analyze_with_retry(
            test_job["job_title"], 
            test_job["job_description"], 
            timeout=5, 
            use_quick=True
        )
        
        if quick_result:
            print(f"   ✅ 快速模式: {quick_result['processing_time']:.2f}s")
            print(f"   技能: {quick_result['final_skills']}")
        
        # 测试标准模式
        print("🎯 测试标准模式...")
        standard_result = self.analyze_with_retry(
            test_job["job_title"], 
            test_job["job_description"], 
            timeout=30, 
            use_quick=False
        )
        
        if standard_result:
            print(f"   ✅ 标准模式: {standard_result['processing_time']:.2f}s")
            print(f"   技能: {standard_result['final_skills']}")
        
        # 性能对比
        if quick_result and standard_result:
            speedup = standard_result['processing_time'] / quick_result['processing_time']
            print(f"\n📈 性能对比:")
            print(f"   快速模式比标准模式快 {speedup:.1f}x")

def test_single_analysis_optimized(tester: OptimizedAPITester):
    """优化版单条分析测试"""
    print("=== 优化版单条职位分析测试 ===")
    
    test_jobs = [
        {
            "job_title": "Python开发工程师",
            "job_description": "负责后端开发，使用Django框架，需要Python、MySQL、Redis经验"
        },
        {
            "job_title": "数据分析师", 
            "job_description": "分析业务数据，制作报表，需要SQL、Python、Excel技能"
        },
        {
            "job_title": "产品经理",
            "job_description": "负责产品规划，需求分析，项目管理和用户研究经验"
        }
    ]
    
    for i, job in enumerate(test_jobs, 1):
        print(f"\n📋 测试 {i}/3: {job['job_title']}")
        
        # 先尝试快速模式
        result = tester.analyze_with_retry(
            job['job_title'], 
            job['job_description'], 
            timeout=10,
            use_quick=True
        )
        
        if result:
            print(f"   ✅ 成功 ({result['processing_time']:.2f}s)")
            print(f"   技能: {result['final_skills']}")
        else:
            print(f"   ❌ 失败")

def test_concurrent_requests(tester: OptimizedAPITester):
    """并发请求测试"""
    print("\n=== 并发请求测试 ===")
    
    test_job = {
        "job_title": "软件工程师",
        "job_description": "开发Web应用，需要Java、Spring、MySQL经验"
    }
    
    def single_request(request_id: int):
        """单个请求"""
        start_time = time.time()
        result = tester.analyze_with_retry(
            f"{test_job['job_title']} #{request_id}",
            test_job['job_description'],
            timeout=15,
            use_quick=True  # 使用快速模式减少超时
        )
        
        total_time = time.time() - start_time
        return {
            'id': request_id,
            'success': result is not None,
            'total_time': total_time,
            'processing_time': result['processing_time'] if result else 0
        }
    
    # 并发执行
    num_requests = 3  # 降低并发数避免超载
    print(f"🔄 发送 {num_requests} 个并发请求...")
    
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(single_request, i) for i in range(1, num_requests + 1)]
        results = []
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                results.append(result)
                status = "✅" if result['success'] else "❌"
                print(f"   {status} 请求#{result['id']}: {result['total_time']:.2f}s")
            except Exception as e:
                print(f"   ❌ 请求失败: {e}")
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    if results:
        avg_time = sum(r['total_time'] for r in results) / len(results)
        print(f"\n📊 并发测试结果:")
        print(f"   成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        print(f"   平均耗时: {avg_time:.2f}s")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化版API测试工具")
    parser.add_argument("--api-url", default=API_URL, help="API服务地址")
    parser.add_argument("--quick-only", action="store_true", help="仅使用快速模式")
    parser.add_argument("--no-wait", action="store_true", help="不等待服务启动")
    args = parser.parse_args()
    
    print("🚀 优化版Qwen3 API测试工具")
    print(f"API地址: {args.api_url}")
    print("-" * 60)
    
    tester = OptimizedAPITester(args.api_url)
    
    # 检查服务状态
    if not args.no_wait and not tester.wait_for_service():
        print("💡 启动建议:")
        print("   python scripts/qwen3_api_server_optimized.py")
        sys.exit(1)
    
    # 显示系统状态
    stats = tester.get_system_stats()
    if stats:
        print(f"\n📊 系统状态:")
        system = stats.get('system', {})
        print(f"   CPU使用率: {system.get('cpu_percent', 0):.1f}%")
        print(f"   内存使用率: {system.get('memory_percent', 0):.1f}%")
        print(f"   可用内存: {system.get('memory_available_gb', 0):.1f}GB")
        print(f"   模型已加载: {'✅' if system.get('model_loaded') else '❌'}")
    
    # 运行测试
    test_single_analysis_optimized(tester)
    
    if not args.quick_only:
        tester.test_performance_comparison()
        test_concurrent_requests(tester)
    
    print("\n" + "=" * 60)
    print("✅ 优化版测试完成！")
    print(f"📖 API文档: {args.api_url}/docs")

if __name__ == "__main__":
    main()