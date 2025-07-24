#!/usr/bin/env python3
"""
知识图谱集成模块
将已构建的课程知识图谱集成到DCAS系统中，提供智能化的课程推荐和内容生成
"""

import json
import pickle
import logging
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

from .data_models import JobMarketData, LearnerData, ContentRecommendation
from .config import get_config

logger = logging.getLogger(__name__)

@dataclass
class CourseNode:
    """课程节点数据结构"""
    course_id: str
    course_name: str
    description: str
    topics: List[str]
    difficulty_level: str
    prerequisites: List[str] = None
    learning_objectives: List[str] = None
    skills_covered: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.learning_objectives is None:
            self.learning_objectives = []
        if self.skills_covered is None:
            self.skills_covered = []

@dataclass
class SkillPath:
    """技能学习路径"""
    skill_name: str
    required_courses: List[str]
    recommended_sequence: List[str]
    estimated_time: float
    difficulty_progression: List[str]

class KnowledgeGraphIntegrator:
    """知识图谱集成器"""
    
    def __init__(self, kg_output_dir: str = None):
        """
        初始化知识图谱集成器
        
        Args:
            kg_output_dir: 知识图谱输出目录
        """
        # 自动检测知识图谱目录
        if kg_output_dir is None:
            possible_dirs = [
                "knowledge_graph",  # 新的目录名
                "knowledge_graph_output_production",
                "knowledge_graph_output"
            ]
            
            for dir_name in possible_dirs:
                if Path(dir_name).exists():
                    kg_output_dir = dir_name
                    break
            
            if kg_output_dir is None:
                logger.warning("未找到知识图谱输出目录，将使用模拟数据")
                self.use_mock_data = True
                return
        
        self.kg_dir = Path(kg_output_dir)
        self.use_mock_data = False
        self.graph = None
        self.embeddings = None
        self.courses = []
        self.course_nodes = {}
        self.skill_course_mapping = {}
        
        logger.info(f"🔗 初始化知识图谱集成器，目录: {self.kg_dir}")
        self._load_knowledge_graph()
        self._build_skill_mappings()
    
    def _load_knowledge_graph(self):
        """加载知识图谱数据"""
        try:
            # 加载图结构
            graph_files = list((self.kg_dir / "graphs").glob("*.pkl"))
            if graph_files:
                latest_graph_file = max(graph_files, key=lambda x: x.stat().st_mtime)
                with open(latest_graph_file, 'rb') as f:
                    self.graph = pickle.load(f)
                logger.info(f"✅ 加载知识图谱: {latest_graph_file.name}")
            
            # 加载embeddings
            embedding_files = list((self.kg_dir / "embeddings").glob("*.pkl"))
            if embedding_files:
                latest_embedding_file = max(embedding_files, key=lambda x: x.stat().st_mtime)
                with open(latest_embedding_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    self.courses = data['courses']
                logger.info(f"✅ 加载embeddings: {latest_embedding_file.name}")
                logger.info(f"   - 课程数量: {len(self.courses)}")
                logger.info(f"   - 向量维度: {self.embeddings.shape[1]}")
            
            # 构建课程节点对象
            self._build_course_nodes()
            
        except Exception as e:
            logger.error(f"知识图谱加载失败: {e}")
            self.use_mock_data = True
    
    def _build_course_nodes(self):
        """构建课程节点对象"""
        for i, course_data in enumerate(self.courses):
            course_node = CourseNode(
                course_id=f"course_{i}",
                course_name=course_data['course_name'],
                description=course_data['course_description'],
                topics=course_data['topics'],
                difficulty_level=self._infer_difficulty(course_data),
                skills_covered=self._extract_skills_from_topics(course_data['topics'])
            )
            self.course_nodes[course_node.course_id] = course_node
    
    def _infer_difficulty(self, course_data: Dict) -> str:
        """根据课程数据推断难度等级"""
        course_name = course_data['course_name'].lower()
        description = course_data['course_description'].lower()
        
        # 简单规则推断
        if any(word in course_name or word in description for word in 
               ['introduction', 'basic', 'fundamentals', '101', 'beginner']):
            return 'beginner'
        elif any(word in course_name or word in description for word in 
                 ['advanced', 'expert', 'phd', 'doctoral', 'graduate']):
            return 'advanced'
        else:
            return 'intermediate'
    
    def _extract_skills_from_topics(self, topics: List[str]) -> List[str]:
        """从主题中提取技能"""
        # 技能关键词映射
        skill_mappings = {
            'Computer Science': ['编程', 'Programming', '软件开发'],
            'Engineering': ['工程设计', '系统设计', '技术分析'],
            'Mathematics': ['数学建模', '统计分析', '算法设计'],
            'Data Mining': ['数据挖掘', '数据分析', 'Python', 'R'],
            'Machine Learning': ['机器学习', '深度学习', 'AI'],
            'Electronics': ['电子技术', '硬件设计', '电路分析'],
            'Graphics and Visualization': ['可视化', '图形设计', 'UI设计'],
            'Software Design and Engineering': ['软件工程', '系统架构', '项目管理']
        }
        
        skills = []
        for topic in topics:
            if topic in skill_mappings:
                skills.extend(skill_mappings[topic])
        
        return list(set(skills))  # 去重
    
    def _build_skill_mappings(self):
        """构建技能-课程映射"""
        for course_id, course_node in self.course_nodes.items():
            for skill in course_node.skills_covered:
                if skill not in self.skill_course_mapping:
                    self.skill_course_mapping[skill] = []
                self.skill_course_mapping[skill].append(course_id)
    
    def find_courses_by_skills(self, required_skills: List[str], 
                             max_courses: int = 10) -> List[Dict[str, Any]]:
        """
        根据技能需求查找相关课程
        
        Args:
            required_skills: 需要的技能列表
            max_courses: 最大返回课程数
            
        Returns:
            匹配的课程列表，按相关度排序
        """
        if self.use_mock_data:
            return self._get_mock_courses(required_skills, max_courses)
        
        course_scores = {}
        
        # 计算每个课程的匹配分数
        for skill in required_skills:
            if skill in self.skill_course_mapping:
                for course_id in self.skill_course_mapping[skill]:
                    course_scores[course_id] = course_scores.get(course_id, 0) + 1
        
        # 排序并获取最相关的课程
        sorted_courses = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for course_id, score in sorted_courses[:max_courses]:
            course_node = self.course_nodes[course_id]
            results.append({
                'course_id': course_id,
                'course_name': course_node.course_name,
                'description': course_node.description[:200] + "..." if len(course_node.description) > 200 else course_node.description,
                'topics': course_node.topics,
                'skills_covered': course_node.skills_covered,
                'difficulty_level': course_node.difficulty_level,
                'relevance_score': score,
                'matched_skills': [skill for skill in required_skills if skill in course_node.skills_covered]
            })
        
        return results
    
    def recommend_learning_path(self, target_skills: List[str], 
                              learner_data: LearnerData) -> Dict[str, Any]:
        """
        基于知识图谱推荐学习路径
        
        Args:
            target_skills: 目标技能
            learner_data: 学习者数据
            
        Returns:
            推荐的学习路径
        """
        if self.use_mock_data:
            return self._get_mock_learning_path(target_skills, learner_data)
        
        # 获取相关课程
        relevant_courses = self.find_courses_by_skills(target_skills, max_courses=20)
        
        # 根据学习者当前水平筛选课程
        filtered_courses = self._filter_courses_by_level(relevant_courses, learner_data)
        
        # 构建学习序列
        learning_sequence = self._build_learning_sequence(filtered_courses, learner_data)
        
        # 估算学习时间
        estimated_time = self._estimate_learning_time(learning_sequence)
        
        return {
            'target_skills': target_skills,
            'recommended_courses': learning_sequence,
            'total_courses': len(learning_sequence),
            'estimated_time_hours': estimated_time,
            'personalization_notes': self._generate_path_notes(learner_data, learning_sequence)
        }
    
    def _filter_courses_by_level(self, courses: List[Dict], 
                               learner_data: LearnerData) -> List[Dict]:
        """根据学习者水平筛选课程"""
        # 简单的水平评估
        avg_mastery = np.mean(list(learner_data.knowledge_state.values())) if learner_data.knowledge_state else 0.3
        
        if avg_mastery < 0.3:
            preferred_levels = ['beginner', 'intermediate']
        elif avg_mastery < 0.7:
            preferred_levels = ['intermediate', 'advanced']
        else:
            preferred_levels = ['advanced', 'intermediate']
        
        filtered = []
        for course in courses:
            if course['difficulty_level'] in preferred_levels:
                filtered.append(course)
        
        return filtered[:10]  # 限制数量
    
    def _build_learning_sequence(self, courses: List[Dict], 
                               learner_data: LearnerData) -> List[Dict]:
        """构建学习序列"""
        # 按难度排序：初级 -> 中级 -> 高级
        difficulty_order = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        
        # 根据学习风格调整顺序
        if learner_data.learning_style.get('processing') == 'active':
            # 主动型学习者：理论与实践交替
            theory_courses = [c for c in courses if any(topic in ['Mathematics', 'Theory'] for topic in c['topics'])]
            practical_courses = [c for c in courses if any(topic in ['Engineering', 'Programming'] for topic in c['topics'])]
            
            sequence = []
            for i in range(max(len(theory_courses), len(practical_courses))):
                if i < len(theory_courses):
                    sequence.append(theory_courses[i])
                if i < len(practical_courses):
                    sequence.append(practical_courses[i])
        else:
            # 反思型学习者：按难度递进
            sequence = sorted(courses, key=lambda x: difficulty_order.get(x['difficulty_level'], 2))
        
        return sequence
    
    def _estimate_learning_time(self, courses: List[Dict]) -> float:
        """估算学习时间（小时）"""
        base_hours = {'beginner': 20, 'intermediate': 30, 'advanced': 40}
        total_hours = 0
        
        for course in courses:
            difficulty = course['difficulty_level']
            total_hours += base_hours.get(difficulty, 25)
        
        return total_hours
    
    def _generate_path_notes(self, learner_data: LearnerData, 
                           courses: List[Dict]) -> List[str]:
        """生成个性化学习路径说明"""
        notes = []
        
        # 基于学习风格
        processing_style = learner_data.learning_style.get('processing', 'active')
        if processing_style == 'active':
            notes.append("🎯 基于你的主动型学习风格，路径包含更多实践项目和动手实验")
        else:
            notes.append("📚 基于你的反思型学习风格，路径强调理论基础和深度思考")
        
        # 基于课程数量
        if len(courses) > 8:
            notes.append("⏰ 学习路径较长，建议分阶段完成，每完成2-3门课程休息调整")
        
        # 基于难度分布
        difficulties = [c['difficulty_level'] for c in courses]
        if difficulties.count('advanced') > len(courses) // 2:
            notes.append("🚀 路径包含较多高级课程，建议确保前置基础扎实")
        
        return notes
    
    def get_course_prerequisites(self, course_id: str) -> List[str]:
        """获取课程先修要求"""
        if self.use_mock_data or course_id not in self.course_nodes:
            return []
        
        # 基于知识图谱分析先修关系
        if self.graph is None:
            return []
        
        # 找到相关的较低难度课程作为先修课程
        target_course = self.course_nodes[course_id]
        prerequisites = []
        
        for other_id, other_course in self.course_nodes.items():
            if (other_id != course_id and 
                other_course.difficulty_level == 'beginner' and 
                target_course.difficulty_level in ['intermediate', 'advanced'] and
                len(set(other_course.skills_covered) & set(target_course.skills_covered)) > 0):
                prerequisites.append(other_id)
        
        return prerequisites[:3]  # 最多3门先修课程
    
    def _get_mock_courses(self, required_skills: List[str], max_courses: int) -> List[Dict]:
        """获取模拟课程数据"""
        mock_courses = [
            {
                'course_id': 'mock_001',
                'course_name': f'{skill}基础教程',
                'description': f'这是一门关于{skill}的基础课程，涵盖核心概念和实践应用。',
                'topics': ['Computer Science', 'Engineering'],
                'skills_covered': [skill],
                'difficulty_level': 'beginner',
                'relevance_score': 1,
                'matched_skills': [skill]
            }
            for skill in required_skills[:max_courses]
        ]
        return mock_courses
    
    def _get_mock_learning_path(self, target_skills: List[str], 
                              learner_data: LearnerData) -> Dict[str, Any]:
        """获取模拟学习路径"""
        mock_courses = self._get_mock_courses(target_skills, len(target_skills))
        
        return {
            'target_skills': target_skills,
            'recommended_courses': mock_courses,
            'total_courses': len(mock_courses),
            'estimated_time_hours': len(mock_courses) * 25,
            'personalization_notes': ['📝 当前使用模拟数据，实际部署时将基于真实知识图谱生成路径']
        }

class EnhancedContentGenerator:
    """增强的内容生成器，集成知识图谱"""
    
    def __init__(self, kg_integrator: KnowledgeGraphIntegrator):
        self.kg_integrator = kg_integrator
    
    def generate_kg_enhanced_content(self, job_market_data: JobMarketData,
                                   learner_data: LearnerData) -> Dict[str, Any]:
        """
        基于知识图谱生成增强的课程内容推荐
        
        Args:
            job_market_data: 就业市场数据
            learner_data: 学习者数据
            
        Returns:
            增强的内容推荐
        """
        # 获取技能相关课程
        relevant_courses = self.kg_integrator.find_courses_by_skills(
            job_market_data.required_skills, max_courses=15
        )
        
        # 生成学习路径
        learning_path = self.kg_integrator.recommend_learning_path(
            job_market_data.required_skills, learner_data
        )
        
        # 为每个技能生成详细的学习模块
        skill_modules = []
        for skill in job_market_data.required_skills:
            skill_courses = [c for c in relevant_courses if skill in c.get('matched_skills', [])]
            if skill_courses:
                best_course = skill_courses[0]  # 取最相关的课程
                
                module = {
                    'skill': skill,
                    'source_course': best_course['course_name'],
                    'difficulty': best_course['difficulty_level'],
                    'content_type': 'kg_enhanced',
                    'learning_objectives': [
                        f'理解{skill}的核心概念',
                        f'掌握{skill}的实际应用',
                        f'能够运用{skill}解决实际问题'
                    ],
                    'recommended_resources': [
                        f"参考课程: {best_course['course_name']}",
                        f"相关主题: {', '.join(best_course['topics'][:3])}"
                    ],
                    'estimated_hours': 20 if best_course['difficulty_level'] == 'beginner' else 30
                }
                skill_modules.append(module)
        
        return {
            'enhanced_recommendation': {
                'kg_relevant_courses': relevant_courses,
                'personalized_learning_path': learning_path,
                'skill_modules': skill_modules,
                'total_estimated_hours': sum(m['estimated_hours'] for m in skill_modules),
                'kg_integration_notes': [
                    '🔗 基于课程知识图谱生成的个性化推荐',
                    f'📊 分析了{len(relevant_courses)}门相关课程',
                    f'🎯 为{len(job_market_data.required_skills)}个技能定制学习路径'
                ]
            }
        }

# 测试代码
if __name__ == "__main__":
    # 测试知识图谱集成
    integrator = KnowledgeGraphIntegrator()
    
    # 模拟数据
    test_skills = ['Python', '机器学习', '数据分析']
    
    if not integrator.use_mock_data:
        courses = integrator.find_courses_by_skills(test_skills)
        print("🔍 相关课程:")
        for course in courses[:3]:
            print(f"  - {course['course_name']} (相关度: {course['relevance_score']})")
    
    print("✅ 知识图谱集成模块测试完成")