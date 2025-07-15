import os
import glob
import tiktoken

# --- 配置 ---

# GraphRAG 处理乘数
# GraphRAG 的索引过程会多次调用大模型，总处理 token 量通常是原始文本量的数倍。
# 这里我们使用一个保守的估计值 3。
GRAPHRAG_PROCESSING_MULTIPLIER = 3

# 假设输入输出token的比例
# 在GraphRAG构建过程中，大部分是输入（原始文本），输出（提取的知识）相对较少。
# 我们假设 80% 是输入，20% 是输出。
INPUT_TOKEN_RATIO = 0.8
OUTPUT_TOKEN_RATIO = 0.2

# --- 模型定价信息 ---
# 基于用户提供的数据
MODELS = [
    {
        "name": "doubao-seed-1.6",
        "notes": "假设: 输入 < 32K, 输出 < 0.2K tokens",
        "pricing": {
            "input_miss": 0.80, "input_hit": 0.16, "output": 2.00, "cache_storage_per_hour": 0.017,
        }
    },
    {
        "name": "doubao-seed-1.6 (大输出)",
        "notes": "假设: 输入 < 32K, 输出 > 0.2K tokens (对比)",
        "pricing": {
            "input_miss": 0.80, "input_hit": 0.16, "output": 8.00, "cache_storage_per_hour": 0.017,
        }
    },
    {
        "name": "doubao-seed-1.6-thinking",
        "notes": "假设: 输入 < 32K tokens",
        "pricing": {
            "input_miss": 0.80, "input_hit": 0.16, "output": 8.00, "cache_storage_per_hour": 0.017,
        }
    },
    {
        "name": "doubao-seed-1.6-flash",
        "notes": "假设: 输入 < 32K tokens",
        "pricing": {
            "input_miss": 0.15, "input_hit": 0.03, "output": 1.50, "cache_storage_per_hour": 0.017,
        }
    },
    {
        "name": "doubao-1.5-thinking-pro",
        "notes": "不支持缓存",
        "pricing": {
            "input_miss": 4.00, "input_hit": None, "output": 16.00, "cache_storage_per_hour": None,
        }
    },
    {
        "name": "deepseek-r1",
        "notes": "",
        "pricing": {
            "input_miss": 4.00, "input_hit": 0.80, "output": 16.00, "cache_storage_per_hour": 0.017,
        }
    },
    {
        "name": "deepseek-r1-distill-qwen-32b",
        "notes": "",
        "pricing": {
            "input_miss": 1.50, "input_hit": 0.30, "output": 6.00, "cache_storage_per_hour": 0.017,
        }
    },
    {
        "name": "deepseek-r1-distill-qwen-7b",
        "notes": "不支持缓存",
        "pricing": {
            "input_miss": 0.60, "input_hit": None, "output": 2.40, "cache_storage_per_hour": None,
        }
    },
]


# --- 函数 ---

def get_tokenizer():
    """获取一个通用的分词器，这里使用 tiktoken 的 cl100k_base"""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        print("tiktoken 初始化失败，将使用基于字符数的估算方法。")
        return None

def count_tokens_from_file(filepath, tokenizer):
    """计算单个文件的 token 数量"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            return len(text) / 4
    except Exception as e:
        print(f"无法读取或处理文件 {filepath}: {e}")
        return 0

def estimate_cost(directory_path):
    """
    估算给定目录下所有 .json 文件的 GraphRAG 构建成本。
    """
    print(f"开始扫描目录: {directory_path}")
    
    file_paths = glob.glob(os.path.join(directory_path, '**', '*.json'), recursive=True)
    
    if not file_paths:
        print(f"错误：在指定目录下未找到任何 .json 文件。请检查路径是否正确。")
        return

    print(f"找到 {len(file_paths)} 个 .json 文件。开始计算 Token 总数...")

    tokenizer = get_tokenizer()
    total_tokens = sum(count_tokens_from_file(fp, tokenizer) for fp in file_paths)

    if total_tokens == 0:
        print("错误：无法计算总 Token 数，所有文件都无法处理。")
        return

    print("\n--- 基础信息 ---")
    print(f"原始文本总 Token 数 (估算): {total_tokens:,.0f}")

    graphrag_total_tokens = total_tokens * GRAPHRAG_PROCESSING_MULTIPLIER
    input_tokens = graphrag_total_tokens * INPUT_TOKEN_RATIO
    output_tokens = graphrag_total_tokens * OUTPUT_TOKEN_RATIO
    
    print(f"GraphRAG 处理总 Token 数 (估算, x{GRAPHRAG_PROCESSING_MULTIPLIER}): {graphrag_total_tokens:,.0f}")
    print(f"  - 其中输入 Tokens (估算, {INPUT_TOKEN_RATIO*100:.0f}%): {input_tokens:,.0f}")
    print(f"  - 其中输出 Tokens (估算, {OUTPUT_TOKEN_RATIO*100:.0f}%): {output_tokens:,.0f}")
    
    print("\n" + "="*50)
    print("      多模型 GraphRAG 构建成本横向对比")
    print("="*50)

    for model in MODELS:
        print(f"\n--- 模型: {model['name']} ---")
        if model['notes']:
            print(f"({model['notes']})")

        p = model['pricing']
        
        # 计算费用
        cost_in_miss = (input_tokens / 1_000_000) * p['input_miss']
        cost_out = (output_tokens / 1_000_000) * p['output']
        total_cost_miss = cost_in_miss + cost_out
        
        print(f"  [缓存未命中] 总费用: {total_cost_miss:,.2f} 元")
        print(f"    - 输入: {cost_in_miss:,.2f} 元 (@{p['input_miss']}/M)")
        print(f"    - 输出: {cost_out:,.2f} 元 (@{p['output']}/M)")

        if p['input_hit'] is not None:
            cost_in_hit = (input_tokens / 1_000_000) * p['input_hit']
            total_cost_hit = cost_in_hit + cost_out
            print(f"  [缓存命中]   总费用: {total_cost_hit:,.2f} 元")
            print(f"    - 输入: {cost_in_hit:,.2f} 元 (@{p['input_hit']}/M)")
        
        if p['cache_storage_per_hour'] is not None:
            cache_cost = (input_tokens / 1_000_000) * p['cache_storage_per_hour']
            print(f"  [缓存存储]   费用: {cache_cost:,.4f} 元 / 小时")

    print("\n" + "="*50)
    print("\n--- 免责声明 ---")
    print("1. Token 数量基于通用分词器估算，可能与各模型实际计算方式有差异。")
    print(f"2. GraphRAG 的 {GRAPHRAG_PROCESSING_MULTIPLIER} 倍处理量是一个行业经验估算值，实际消耗可能因数据和配置而异。")
    print("3. 对于分级定价模型，估算是基于对 GraphRAG 典型输入输出长度的假设。")
    print("4. '缓存命中' 概率未知，对于首次处理，'缓存未命中' 的价格更具参考价值。")
    print("5. 此结果仅供参考，实际费用以平台最终账单为准。")


if __name__ == '__main__':
    target_directory = 'datasets/Course Details/General'
    if not os.path.isdir(target_directory):
        print(f"错误：目录 '{target_directory}' 不存在。")
    else:
        estimate_cost(target_directory)