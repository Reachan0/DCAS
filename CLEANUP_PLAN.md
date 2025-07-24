# DCAS项目文件整理方案

## 当前问题分析
1. **重复文件**: 多个版本的API服务器和测试文件
2. **混乱的界面文件**: 5个不同版本的Streamlit应用
3. **临时文件堆积**: 多个output、temp、debug目录
4. **文档分散**: docs文件夹内容少，说明文档分散
5. **配置文件重复**: 多个requirements文件

## 整理方案

### 1. 保留的核心文件
- `dcas_expert_committee_clean.py` (主要演示界面)
- `dcas_core/` (核心功能模块)
- `knowledge_graph_output_production/` (生产版知识图谱)
- `datasets/` (原始数据)
- `scripts/` (经过清理的工具脚本)

### 2. 需要删除的文件
- 重复的Streamlit界面文件
- 临时和调试文件
- 旧版本的API服务器
- 重复的配置文件

### 3. 需要重新组织的目录
- 合并多个output目录
- 整理scripts子目录
- 统一配置文件

### 4. 新的目录结构
```
DCAS/
├── README.md                           # 主说明文档
├── requirements.txt                    # 统一依赖
├── dcas_config.json                   # 主配置文件
├── main.py                            # 主入口(专家委员会系统)
├── dcas_core/                         # 核心功能模块
├── datasets/                          # 原始数据(保持不变)
├── knowledge_graph/                   # 知识图谱(重命名)
├── scripts/                           # 工具脚本(清理后)
├── docs/                              # 文档(整理后)
├── outputs/                           # 所有输出(合并)
└── archive/                           # 已有的归档
```

## 执行步骤
1. 备份当前重要文件
2. 删除冗余文件
3. 重新组织目录
4. 更新配置和文档
5. 测试核心功能