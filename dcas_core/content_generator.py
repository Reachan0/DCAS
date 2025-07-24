#!/usr/bin/env python3
"""
课程内容生成模块
实现课程相关智能体中缺失的内容生成功能
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import re

from .data_models import LearnerData, JobMarketData, ContentRecommendation
from .config import get_config

logger = logging.getLogger(__name__)

class ContentTemplate:
    """内容模板类"""
    
    def __init__(self, template_type: str, template_content: str, difficulty_level: str):
        self.template_type = template_type  # lecture, exercise, project, case_study
        self.template_content = template_content
        self.difficulty_level = difficulty_level
    
    def generate_content(self, skill: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """基于模板生成具体内容"""
        content = self.template_content.format(
            skill=skill,
            difficulty=self.difficulty_level,
            **context
        )
        
        return {
            "type": self.template_type,
            "content": content,
            "skill": skill,
            "difficulty": self.difficulty_level,
            "estimated_minutes": self._estimate_time(content)
        }
    
    def _estimate_time(self, content: str) -> int:
        """估算学习时间（分钟）"""
        word_count = len(content.split())
        if self.template_type == "lecture":
            return max(15, word_count // 3)  # 3词/分钟阅读速度
        elif self.template_type == "exercise":
            return max(30, word_count // 2)  # 需要更多思考时间
        elif self.template_type == "project":
            return max(120, word_count * 2)  # 项目需要大量实践时间
        else:  # case_study
            return max(45, word_count // 2)

class ContentTemplateManager:
    """内容模板管理器"""
    
    def __init__(self):
        self.templates = self._load_default_templates()
        config = get_config()
        template_dir = Path(config.get("agents.content_generator.template_dir", "content_templates"))
        template_dir.mkdir(exist_ok=True)
        
        # 尝试加载自定义模板
        self._load_custom_templates(template_dir)
    
    def _load_default_templates(self) -> Dict[str, Dict[str, ContentTemplate]]:
        """加载默认内容模板"""
        templates = {
            "lecture": {
                "beginner": ContentTemplate(
                    "lecture",
                    """
# {skill} 基础入门

## 学习目标
通过本节课程，你将掌握{skill}的基本概念和核心原理。

## 核心概念
{skill}是一个重要的技能，在现代职场中有着广泛的应用。

### 基本定义
{skill}指的是...（这里会根据具体技能进行个性化生成）

### 为什么重要
1. 提升工作效率
2. 增强解决问题的能力
3. 符合市场需求趋势

## 实际应用场景
- 场景1：日常工作中的应用
- 场景2：项目实施中的作用
- 场景3：团队协作中的价值

## 小结
本节课我们学习了{skill}的基础知识，为后续深入学习打下基础。

## 思考题
1. {skill}在你的工作中可能有哪些应用？
2. 如何开始实践{skill}？
                    """,
                    "beginner"
                ),
                "intermediate": ContentTemplate(
                    "lecture", 
                    """
# {skill} 进阶应用

## 学习目标  
深入理解{skill}的高级概念，掌握实际应用中的最佳实践。

## 进阶概念
在基础知识的基础上，我们将探讨{skill}的更深层次应用。

### 高级特性
- 特性1：复杂场景下的应用
- 特性2：性能优化技巧
- 特性3：与其他技能的整合

### 最佳实践
1. 经验分享：业界标准做法
2. 常见陷阱及避免方法
3. 效率提升技巧

## 案例分析
### 案例1：真实项目中的{skill}应用
详细分析一个实际项目中如何运用{skill}解决复杂问题。

### 案例2：错误示例分析
通过分析常见错误，加深对{skill}正确使用的理解。

## 总结
{skill}的进阶应用需要在实践中不断积累经验。
                    """,
                    "intermediate"
                )
            },
            "exercise": {
                "beginner": ContentTemplate(
                    "exercise",
                    """
# {skill} 基础练习

## 练习目标
通过实际操作练习，巩固{skill}的基本技能。

## 练习1：基础操作
**任务描述：**
完成一个简单的{skill}相关任务，熟悉基本操作流程。

**步骤指导：**
1. 准备工作环境
2. 按照示例完成基础操作
3. 验证结果是否正确

**预期输出：**
能够独立完成基本的{skill}操作。

## 练习2：问题解决
**场景描述：**
给定一个实际问题，运用{skill}来解决。

**解题思路：**
1. 分析问题需求
2. 选择合适的{skill}方法
3. 实施解决方案
4. 检验解决效果

## 自我检查
- [ ] 是否理解了{skill}的基本概念？
- [ ] 能否独立完成基础操作？
- [ ] 遇到问题时知道如何寻求帮助？

## 扩展思考
尝试将{skill}应用到你熟悉的其他领域中。
                    """,
                    "beginner"
                )
            },
            "project": {
                "intermediate": ContentTemplate(
                    "project",
                    """
# {skill} 综合项目

## 项目概述
通过一个完整的项目实践，综合运用{skill}解决实际问题。

## 项目背景
在现代工作环境中，{skill}的应用场景越来越广泛。本项目将模拟真实工作场景，让你获得宝贵的实践经验。

## 项目目标
1. 深度掌握{skill}的实际应用
2. 培养项目管理和问题解决能力
3. 建立可展示的作品集

## 项目阶段

### 第一阶段：需求分析与规划
- 理解项目需求
- 制定实施计划
- 选择合适的{skill}工具和方法

### 第二阶段：设计与开发
- 设计解决方案
- 实施核心功能
- 进行初步测试

### 第三阶段：优化与完善
- 性能优化
- 功能完善
- 文档编写

### 第四阶段：展示与反思
- 成果展示
- 项目总结
- 经验分享

## 评估标准
1. 技术实现的正确性（40%）
2. 方案设计的合理性（30%）
3. 项目文档的完整性（20%）
4. 创新性和实用性（10%）

## 资源支持
- 技术文档和参考资料
- 在线答疑和指导
- 同伴互助和经验分享

## 提交要求
1. 完整的项目源码
2. 详细的项目文档
3. 演示视频或截图
4. 项目总结报告
                    """,
                    "intermediate"
                )
            },
            "case_study": {
                "beginner": ContentTemplate(
                    "case_study",
                    """
# {skill} 案例研究

## 案例背景
本案例来源于真实的工作场景，展示了{skill}在实际应用中的重要作用。

## 情境描述
某公司面临了一个需要运用{skill}来解决的挑战...

### 初始状况
- 问题现象：描述遇到的具体问题
- 影响范围：问题对业务的影响
- 资源限制：可用的资源和约束条件

### 解决过程
1. **问题分析阶段**
   - 深入调研问题根源
   - 收集相关数据和信息
   - 确定解决方案的方向

2. **方案设计阶段**
   - 运用{skill}设计解决方案
   - 考虑各种可能的实施路径
   - 评估方案的可行性

3. **实施执行阶段**
   - 按计划实施解决方案
   - 监控实施过程和效果
   - 及时调整和优化

### 结果与反思
- **取得的成果**：具体的改善效果
- **经验教训**：过程中的重要收获
- **未来改进**：进一步优化的方向

## 关键学习点
1. {skill}在复杂问题中的应用策略
2. 解决问题的系统性思维方法
3. 团队协作和沟通的重要性

## 讨论题
1. 如果你遇到类似问题，会如何运用{skill}？
2. 该案例中的解决方案还有哪些改进空间？
3. 如何将这个案例的经验应用到其他场景？

## 拓展阅读
- 相关理论资料
- 类似案例分析
- 行业最佳实践
                    """,
                    "beginner"
                )
            }
        }
        return templates
    
    def _load_custom_templates(self, template_dir: Path):
        """加载自定义模板"""
        try:
            for template_file in template_dir.glob("*.json"):
                with open(template_file, 'r', encoding='utf-8') as f:
                    custom_templates = json.load(f)
                    # 合并自定义模板
                    logger.info(f"加载自定义模板: {template_file}")
        except Exception as e:
            logger.warning(f"加载自定义模板失败: {e}")
    
    def get_template(self, content_type: str, difficulty: str) -> Optional[ContentTemplate]:
        """获取指定类型和难度的模板"""
        return self.templates.get(content_type, {}).get(difficulty)
    
    def get_available_types(self) -> List[str]:
        """获取可用的内容类型"""
        return list(self.templates.keys())

class LearningPathGenerator:
    """学习路径生成器"""
    
    def __init__(self, content_manager: ContentTemplateManager):
        self.content_manager = content_manager
    
    def generate_learning_path(self, target_skills: List[str], 
                             learner_data: LearnerData,
                             difficulty_preference: str = "progressive") -> List[Dict[str, Any]]:
        """
        生成个性化学习路径
        
        Args:
            target_skills: 目标技能列表
            learner_data: 学习者数据
            difficulty_preference: 难度偏好 (progressive, uniform, adaptive)
        """
        learning_path = []
        
        for skill in target_skills:
            # 根据学习者当前掌握程度确定起始难度
            current_mastery = learner_data.knowledge_state.get(skill, 0.0)
            
            if current_mastery < 0.3:
                start_difficulty = "beginner"
            elif current_mastery < 0.7:
                start_difficulty = "intermediate"
            else:
                start_difficulty = "advanced"
            
            # 生成该技能的学习序列
            skill_modules = self._generate_skill_modules(
                skill, start_difficulty, learner_data.learning_style
            )
            
            learning_path.extend(skill_modules)
        
        return learning_path
    
    def _generate_skill_modules(self, skill: str, difficulty: str, 
                               learning_style: Dict[str, str]) -> List[Dict[str, Any]]:
        """为单个技能生成学习模块序列"""
        modules = []
        
        # 根据学习风格调整内容类型顺序
        if learning_style.get("processing") == "active":
            # 主动型学习者：更多练习和项目
            content_sequence = ["lecture", "exercise", "case_study", "project"]
        else:
            # 反思型学习者：更多理论和案例
            content_sequence = ["lecture", "case_study", "exercise", "project"]
        
        for i, content_type in enumerate(content_sequence):
            template = self.content_manager.get_template(content_type, difficulty)
            if template:
                module = {
                    "module_id": f"{skill}_{content_type}_{difficulty}",
                    "title": f"{skill} - {content_type.title()}",
                    "skill": skill,
                    "content_type": content_type,
                    "difficulty": difficulty,
                    "sequence_order": i + 1,
                    "template": template
                }
                modules.append(module)
        
        return modules

class PersonalizedContentGenerator:
    """个性化内容生成器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.template_manager = ContentTemplateManager()
        self.path_generator = LearningPathGenerator(self.template_manager)
    
    def generate_personalized_content(self, job_market_data: JobMarketData,
                                    learner_data: LearnerData) -> ContentRecommendation:
        """生成个性化课程内容推荐"""
        
        # 1. 识别技能缺口
        required_skills = set(job_market_data.required_skills)
        current_skills = set(learner_data.knowledge_state.keys())
        skill_gaps = list(required_skills - current_skills)
        
        # 添加需要提升的技能（掌握度低于市场需求的）
        for skill in required_skills.intersection(current_skills):
            current_level = learner_data.knowledge_state.get(skill, 0)
            required_level = job_market_data.skill_importance.get(skill, 0.8)
            if current_level < required_level * 0.8:  # 低于要求的80%
                skill_gaps.append(skill)
        
        # 2. 生成学习路径
        learning_path = self.path_generator.generate_learning_path(
            skill_gaps, learner_data
        )
        
        # 3. 生成具体内容
        content_modules = []
        total_time = 0
        
        for module_info in learning_path:
            try:
                # 使用模板生成内容
                template = module_info["template"]
                context = {
                    "learner_style": learner_data.learning_style,
                    "job_context": job_market_data.job_title,
                    "skill_importance": job_market_data.skill_importance.get(module_info["skill"], 0.8)
                }
                
                content = template.generate_content(module_info["skill"], context)
                
                # 移除不可序列化的template对象，只保留必要信息
                module_info_clean = {k: v for k, v in module_info.items() if k != "template"}
                content.update(module_info_clean)  # 合并模块信息（不包括template）
                content_modules.append(content)
                total_time += content["estimated_minutes"]
                
            except Exception as e:
                logger.error(f"生成内容失败 {module_info['skill']}: {e}")
                continue
        
        # 4. 生成个性化说明
        personalization_notes = self._generate_personalization_notes(
            learner_data, job_market_data, skill_gaps
        )
        
        # 5. 构建推荐结果
        recommendation = ContentRecommendation(
            recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            target_skills=skill_gaps,
            content_modules=content_modules,
            learning_path=[module["module_id"] for module in content_modules],
            personalization_notes=personalization_notes,
            estimated_completion_time=total_time / 60.0  # 转换为小时
        )
        
        return recommendation
    
    def _generate_personalization_notes(self, learner_data: LearnerData,
                                      job_market_data: JobMarketData,
                                      skill_gaps: List[str]) -> List[str]:
        """生成个性化说明"""
        notes = []
        
        # 基于学习风格的建议
        processing_style = learner_data.learning_style.get("processing", "active")
        if processing_style == "active":
            notes.append("💡 根据你的主动型学习风格，课程安排了更多实践练习和项目。")
        else:
            notes.append("💡 根据你的反思型学习风格，课程包含更多理论讲解和案例分析。")
        
        # 基于技能缺口的说明
        high_priority_skills = [
            skill for skill in skill_gaps 
            if job_market_data.skill_importance.get(skill, 0) > 0.8
        ]
        if high_priority_skills:
            notes.append(f"🎯 重点关注高优先级技能: {', '.join(high_priority_skills[:3])}")
        
        # 基于市场趋势的建议
        if job_market_data.market_trend == "growing":
            notes.append("📈 该领域正在快速发展，建议加快学习进度以抓住机会。")
        
        return notes
    
    def save_content_recommendation(self, recommendation: ContentRecommendation):
        """保存内容推荐到文件"""
        config = get_config()
        output_dir = Path(config.get("agents.content_generator.output_dir"))
        output_dir.mkdir(exist_ok=True)
        
        file_path = output_dir / f"{recommendation.recommendation_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(recommendation.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"内容推荐已保存: {file_path}")

class MockLLMClient:
    """模拟LLM客户端用于测试"""
    
    def generate(self, prompt: str) -> str:
        return "这是模拟生成的内容。在实际应用中，这里会调用真实的LLM来生成个性化内容。"

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    job_data = JobMarketData(
        job_title="数据科学家",
        required_skills=["Python", "机器学习", "SQL", "数据可视化"],
        skill_importance={"Python": 0.9, "机器学习": 0.8, "SQL": 0.7, "数据可视化": 0.6},
        market_trend="growing",
        salary_info={"min": 15000, "max": 30000, "avg": 22000}
    )
    
    learner_data = LearnerData(
        user_id="test_learner",
        knowledge_state={"Python": 0.6, "SQL": 0.3},
        learning_style={"processing": "active", "perception": "sensory"},
        preferences={"engagement_score": 0.8},
        skill_gaps=["机器学习", "数据可视化"],
        engagement_score=0.8
    )
    
    # 测试内容生成
    generator = PersonalizedContentGenerator(MockLLMClient())
    recommendation = generator.generate_personalized_content(job_data, learner_data)
    
    print("🎓 个性化内容生成测试")
    print(f"目标技能: {recommendation.target_skills}")
    print(f"生成模块数: {len(recommendation.content_modules)}")
    print(f"预计完成时间: {recommendation.estimated_completion_time:.1f} 小时")
    print(f"个性化建议: {recommendation.personalization_notes}")
    
    print("✅ 内容生成模块测试通过")