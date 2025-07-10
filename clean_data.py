#!/usr/bin/env python3
"""
DCAS - 数据清洗脚本

清理temp_results.csv中的数据质量问题：
1. 删除包含"无"的记录
2. 统一语言为英文
3. 清理无效的技能描述
4. 标准化技能格式
"""

import pandas as pd
import re
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        # 中英文技能映射词典
        self.skill_translation = {
            # 软技能
            "沟通能力": "Communication Skills",
            "团队合作": "Teamwork",
            "领导力": "Leadership",
            "组织能力": "Organization Skills",
            "创造力": "Creativity",
            "主动性": "Initiative",
            "积极性": "Proactive",
            "责任心": "Responsibility",
            "抗压能力": "Stress Management",
            "问题解决": "Problem Solving",
            "时间管理": "Time Management",
            "注重细节": "Attention to Detail",
            "客户服务": "Customer Service",
            "项目管理": "Project Management",
            "分析能力": "Analytical Skills",
            "学习能力": "Learning Ability",
            "适应能力": "Adaptability",
            "同理心": "Empathy",
            "多任务处理": "Multitasking",
            "决策能力": "Decision Making",
            
            # 技术技能
            "数据分析": "Data Analysis",
            "数据库管理": "Database Management",
            "网页设计": "Web Design",
            "软件开发": "Software Development",
            "系统管理": "System Administration",
            "网络安全": "Network Security",
            "云计算": "Cloud Computing",
            "人工智能": "Artificial Intelligence",
            "机器学习": "Machine Learning",
            "区块链": "Blockchain",
            
            # 专业技能
            "财务分析": "Financial Analysis",
            "市场营销": "Marketing",
            "品牌策略": "Brand Strategy",
            "社交媒体管理": "Social Media Management",
            "内容创作": "Content Creation",
            "搜索引擎优化": "SEO",
            "数字营销": "Digital Marketing",
            "销售技巧": "Sales Skills",
            "谈判技巧": "Negotiation Skills",
            "商业分析": "Business Analysis",
            "战略规划": "Strategic Planning",
            "风险管理": "Risk Management",
            "质量控制": "Quality Control",
            "供应链管理": "Supply Chain Management",
            "库存管理": "Inventory Management",
            "采购管理": "Procurement",
            "运营管理": "Operations Management",
            "人力资源": "Human Resources",
            "招聘": "Recruitment",
            "培训": "Training",
            "绩效管理": "Performance Management",
            
            # 医疗相关
            "心理治疗技术": "Psychotherapy Techniques",
            "心理评估": "Psychological Assessment",
            "治疗计划制定": "Treatment Planning",
            "危机干预": "Crisis Intervention",
            "病例管理": "Case Management",
            "心理创伤治疗": "Trauma Treatment",
            "临床文档管理": "Clinical Documentation",
            "电子健康记录": "Electronic Health Records",
            "心理诊断": "Psychological Diagnosis",
            
            # 工程技术
            "机械设计": "Mechanical Design",
            "电气工程": "Electrical Engineering",
            "土木工程": "Civil Engineering",
            "建筑设计": "Architectural Design",
            "工程制图": "Engineering Drawing",
            "质量保证": "Quality Assurance",
            "测试": "Testing",
            "维护": "Maintenance",
            "故障排除": "Troubleshooting",
            
            # 其他
            "法律知识": "Legal Knowledge",
            "合规管理": "Compliance Management",
            "审计": "Auditing",
            "税务": "Tax",
            "会计": "Accounting",
            "簿记": "Bookkeeping",
            "预算管理": "Budget Management",
            "成本控制": "Cost Control"
        }
        
        # 无效内容模式
        self.invalid_patterns = [
            r'^无$',
            r'^None$',
            r'^N/A$',
            r'^null$',
            r'^空$',
            r'^.*Job Details.*$',
            r'^.*About The Role.*$',
            r'^.*Overview.*$',
            r'^.*Responsibilities.*$',
            r'^.*Position Summary.*$',
            r'^.*Description And Requirements.*$',
            r'^.*Job Description.*$',
            r'^.*Who We Are.*$',
            r'.*including recycling.*',
            r'.*you will report to.*',
            r'.*feel welcome to be.*',
            r'.*brighter future for pets.*',
            r'.*we believe in how positively.*',
            r'.*materials testing and.*'
        ]
    
    def is_valid_skill_text(self, text: str) -> bool:
        """检查技能文本是否有效"""
        if pd.isna(text) or not isinstance(text, str):
            return False
        
        text = text.strip()
        
        # 检查是否为空或太短
        if len(text) < 3:
            return False
        
        # 检查是否匹配无效模式
        for pattern in self.invalid_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        
        # 检查是否包含过多非技能相关内容
        job_desc_indicators = [
            'you will', 'we are', 'position', 'responsibilities', 
            'requirements', 'job details', 'about the role', 'overview',
            'description', 'summary', 'who we are'
        ]
        
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in job_desc_indicators):
            return False
        
        return True
    
    def translate_skills(self, skills_text: str) -> str:
        """将中文技能翻译为英文"""
        if not isinstance(skills_text, str):
            return skills_text
        
        # 分割技能
        skills = [skill.strip() for skill in skills_text.split(',')]
        translated_skills = []
        
        for skill in skills:
            if not skill:
                continue
            
            # 查找精确匹配
            if skill in self.skill_translation:
                translated_skills.append(self.skill_translation[skill])
            else:
                # 查找部分匹配
                translated = False
                for chinese, english in self.skill_translation.items():
                    if chinese in skill:
                        # 替换中文部分
                        new_skill = skill.replace(chinese, english)
                        translated_skills.append(new_skill)
                        translated = True
                        break
                
                if not translated:
                    # 保留原文（可能已经是英文）
                    translated_skills.append(skill)
        
        return ', '.join(translated_skills)
    
    def standardize_skills(self, skills_text: str) -> str:
        """标准化技能格式"""
        if not isinstance(skills_text, str):
            return skills_text
        
        # 统一分隔符
        skills_text = re.sub(r'[，、；;]', ',', skills_text)
        
        # 分割并清理
        skills = [skill.strip() for skill in skills_text.split(',')]
        clean_skills = []
        
        for skill in skills:
            if not skill:
                continue
            
            # 移除引号
            skill = re.sub(r'^["\']|["\']$', '', skill)
            
            # 标准化大小写（首字母大写）
            if skill.islower():
                skill = skill.title()
            
            # 移除重复
            if skill not in clean_skills:
                clean_skills.append(skill)
        
        return ', '.join(clean_skills)
    
    def clean_dataset(self, input_file: str, output_file: str) -> int:
        """清洗数据集"""
        logger.info(f"开始清洗数据: {input_file}")
        
        # 读取数据
        df = pd.read_csv(input_file)
        initial_count = len(df)
        logger.info(f"原始数据: {initial_count} 条记录")
        
        # 1. 删除技能要求为空或无效的记录
        df = df[df['skill_requirements'].apply(self.is_valid_skill_text)]
        after_filter = len(df)
        logger.info(f"过滤无效记录后: {after_filter} 条记录 (删除 {initial_count - after_filter} 条)")
        
        # 2. 翻译中文技能
        logger.info("翻译中文技能...")
        df['skill_requirements'] = df['skill_requirements'].apply(self.translate_skills)
        
        # 3. 标准化技能格式
        logger.info("标准化技能格式...")
        df['skill_requirements'] = df['skill_requirements'].apply(self.standardize_skills)
        
        # 4. 再次过滤（确保清洗后仍然有效）
        df = df[df['skill_requirements'].apply(lambda x: len(x.strip()) > 5)]
        final_count = len(df)
        logger.info(f"最终清洗后: {final_count} 条记录")
        
        # 保存清洗后的数据
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"清洗完成，保存到: {output_file}")
        
        return final_count
    
    def analyze_data_quality(self, file_path: str):
        """分析数据质量"""
        df = pd.read_csv(file_path)
        
        print(f"\n=== 数据质量分析 ===")
        print(f"总记录数: {len(df)}")
        
        # 检查空值
        empty_skills = df['skill_requirements'].isna().sum()
        print(f"空技能记录: {empty_skills}")
        
        # 检查包含"无"的记录
        wu_count = df['skill_requirements'].str.contains('无', na=False).sum()
        print(f"包含'无'的记录: {wu_count}")
        
        # 技能长度分布
        skill_lengths = df['skill_requirements'].str.len()
        print(f"技能描述长度: 最小 {skill_lengths.min()}, 最大 {skill_lengths.max()}, 平均 {skill_lengths.mean():.1f}")
        
        # 中文字符比例
        chinese_count = df['skill_requirements'].str.contains(r'[\u4e00-\u9fff]', na=False).sum()
        print(f"包含中文的记录: {chinese_count} ({chinese_count/len(df)*100:.1f}%)")
        
        print(f"\n=== 示例技能 ===")
        sample_skills = df['skill_requirements'].dropna().head(10)
        for i, skill in enumerate(sample_skills, 1):
            print(f"{i}. {skill}")

def main():
    """主函数"""
    input_file = "datasets/temp_results.csv"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"datasets/cleaned_job_skills_{timestamp}.csv"
    
    # 创建清洗器
    cleaner = DataCleaner()
    
    # 分析原始数据质量
    print("=== 原始数据分析 ===")
    cleaner.analyze_data_quality(input_file)
    
    # 清洗数据
    final_count = cleaner.clean_dataset(input_file, output_file)
    
    # 分析清洗后数据质量
    print("\n=== 清洗后数据分析 ===")
    cleaner.analyze_data_quality(output_file)
    
    print(f"""
=== 数据清洗完成 ===

清洗结果:
- 输出文件: {output_file}
- 最终记录数: {final_count}

主要清洗操作:
1. ✅ 删除包含"无"的记录
2. ✅ 删除无效的职位描述文本
3. ✅ 中文技能翻译为英文
4. ✅ 统一技能格式和大小写
5. ✅ 去重和标准化

下一步: 使用清洗后的数据重新生成LlamaFactory训练数据
""")

if __name__ == "__main__":
    main()