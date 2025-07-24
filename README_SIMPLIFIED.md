# 🎓 DCAS - 动态课程对齐系统 (简化版)

**Dynamic Curriculum Alignment System - Simplified Edition**

基于多智能体的智能化教育课程与就业需求对齐系统，帮助学习者获得与市场需求高度匹配的个性化学习体验。

## 🌟 系统特色

### 🤖 五大智能体协同工作
1. **📊 就业市场分析智能体** - 使用CoT思维链分析职位需求
2. **👤 学习者画像智能体** - 基于Felder-Silverman模型的个性化画像
3. **⚖️ 动态对齐策略智能体** - PTOT思维树推理，制定精准对齐策略
4. **📚 课程内容生成智能体** - 个性化学习内容和路径生成
5. **🎮 模拟与反思智能体** - TIR数字孪生模拟，预测学习效果

### 🔄 端到端智能工作流
```
职位分析 → 学习者画像 → 对齐分析 → 内容生成 → 效果模拟 → 优化建议
```

### 🎯 核心功能
- **智能技能缺口识别** - 精准识别学习者与目标职位的技能差距
- **个性化学习路径** - 基于学习风格的定制化内容推荐
- **学习效果预测** - 虚拟学生模拟，预测学习成功概率
- **动态课程调整** - 实时优化学习内容和难度

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 推荐使用虚拟环境

### 一键启动
```bash
# 运行启动脚本
./start_dcas.sh

# 或手动启动Web界面
pip install streamlit pandas numpy scikit-learn networkx matplotlib seaborn tqdm
streamlit run dcas_web_app.py
```

### 命令行演示
```bash
# 运行完整演示
python3 demo.py

# 快速系统测试
python3 -c "from dcas_core.orchestrator import DCASOrchestrator; print('✅ 系统正常')"
```

## 🏗️ 系统架构

```
📦 DCAS/
├── 🧠 dcas_core/                 # 核心系统模块
│   ├── data_models.py            # 统一数据模型
│   ├── config.py                 # 配置管理系统
│   ├── content_generator.py      # 内容生成模块
│   └── orchestrator.py           # 系统协调器
├── 🤖 scripts/agents/            # 智能体实现
│   ├── job_market_analyst_agent.py
│   ├── learner_profiling_agent.py
│   ├── dynamic_alignment_agent.py
│   └── simulation_reflection_agent.py
├── 🌐 dcas_web_app.py            # Web演示界面
├── 💻 demo.py                    # 命令行演示
├── 🚀 start_dcas.sh              # 一键启动脚本
└── 📊 datasets/                  # 数据集目录
    ├── Course Details/           # MIT OCW课程数据
    └── Job Descptions/           # 职位描述数据
```

## 🎮 使用演示

### Web界面演示
1. 启动：`streamlit run dcas_web_app.py`
2. 访问：http://localhost:8501
3. 功能：
   - 🔧 系统状态监控
   - 📊 职位需求分析
   - 👤 学习者画像分析  
   - 🚀 完整工作流演示

### 命令行演示
```bash
python3 demo.py
```
支持两种模式：
- **自动演示** - 运行预设的数据科学转岗和前端升级场景
- **交互演示** - 自定义职位和学习者信息

## 📋 演示场景

### 🎯 场景1：数据科学转岗
- **目标职位**：高级数据科学家
- **学习者背景**：4年Python开发经验，希望转入数据科学
- **系统输出**：个性化学习路径 + 学习时间预估 + 成功概率预测

### 💻 场景2：前端架构师升级  
- **目标职位**：高级前端架构师
- **学习者背景**：3年React开发经验，希望提升到架构师级别
- **系统输出**：技能提升建议 + 架构能力培养 + 风险因素预警

## ⚙️ 配置说明

系统使用 `dcas_config.json` 进行配置管理：

```json
{
  "system": {
    "version": "1.0.0-simplified",
    "debug": true,
    "data_dir": "dcas_data"
  },
  "agents": {
    "job_market_analyst": {"enabled": true},
    "learner_profiling": {"enabled": true},
    "dynamic_alignment": {"enabled": true},
    "content_generator": {"enabled": true},
    "simulation_reflection": {"enabled": true}
  }
}
```

## 🔧 核心技术

### 智能体技术栈
- **CoT思维链推理** - 结构化分析职位需求
- **PTOT思维树** - 多层次策略制定
- **TIR迭代反思** - 持续优化学习效果
- **数字孪生模拟** - 虚拟学生学习预测

### 机器学习组件
- **Sentence Transformers** - 课程内容嵌入
- **余弦相似度** - 技能匹配计算  
- **K-means聚类** - 课程主题分析
- **知识图谱** - 课程关系建模

## 📊 系统输出

### 分析报告
- **技能匹配度分析** - 当前vs目标技能对比
- **个性化学习建议** - 基于学习风格的推荐
- **学习时间预估** - 精确到小时的时间规划
- **成功概率预测** - 基于历史数据的成功率

### 个性化内容
- **讲座模块** - 理论知识讲解
- **练习模块** - 动手实践训练
- **项目模块** - 综合能力培养
- **案例研究** - 实际场景分析

## 🚧 已知限制

### 简化版限制
- 使用Mock LLM客户端（生产环境需要真实模型）
- 课程数据基于MIT OCW样本
- 模拟学生数量限制为50（可配置）
- 单机部署，暂不支持分布式

### 扩展方向
- 集成真实LLM API（OpenAI、Qwen等）
- 增加更多课程数据源
- 实现分布式部署
- 添加用户认证和数据持久化

## 🤝 开发说明

### 快速测试
```bash
# 测试系统协调器
python3 dcas_core/orchestrator.py

# 测试内容生成器
python3 dcas_core/content_generator.py

# 测试配置系统
python3 dcas_core/config.py
```

---

**🎓 DCAS - 让学习与就业无缝对接！**

*基于多智能体的动态课程对齐系统，为每一位学习者量身定制最优学习路径。*