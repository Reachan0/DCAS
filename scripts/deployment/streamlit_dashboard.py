#!/usr/bin/env python3
"""
DCAS 课程知识图谱可视化界面

基于Streamlit的交互式Web界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="DCAS 课程知识图谱",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CourseKnowledgeGraphDashboard:
    """课程知识图谱可视化仪表板"""
    
    def __init__(self):
        self.load_data()
        
    @st.cache_data
    def load_data(_self):
        """加载知识图谱数据（使用缓存）"""
        # 自动检测数据目录
        possible_dirs = [
            "knowledge_graph_output_production",
            "knowledge_graph_output"
        ]
        
        data_dir = None
        for dir_name in possible_dirs:
            if Path(dir_name).exists():
                data_dir = Path(dir_name)
                break
        
        if data_dir is None:
            st.error("❌ 找不到知识图谱数据目录")
            return None, None, None, None
        
        try:
            # 加载知识图谱
            graph_files = list((data_dir / "graphs").glob("*.pkl"))
            if graph_files:
                latest_graph_file = max(graph_files, key=lambda x: x.stat().st_mtime)
                with open(latest_graph_file, 'rb') as f:
                    graph = pickle.load(f)
            else:
                graph = None
            
            # 加载embeddings
            embedding_files = list((data_dir / "embeddings").glob("*.pkl"))
            if embedding_files:
                latest_embedding_file = max(embedding_files, key=lambda x: x.stat().st_mtime)
                with open(latest_embedding_file, 'rb') as f:
                    embedding_data = pickle.load(f)
                    embeddings = embedding_data['embeddings']
                    courses = embedding_data['courses']
                    metadata = {k: v for k, v in embedding_data.items() 
                              if k not in ['embeddings', 'courses']}
            else:
                embeddings = None
                courses = []
                metadata = {}
            
            # 加载分析结果
            analysis_files = list((data_dir / "analysis").glob("*.json"))
            if analysis_files:
                latest_analysis_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
                with open(latest_analysis_file, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
            else:
                analysis_data = {}
            
            return graph, embeddings, courses, analysis_data
            
        except Exception as e:
            st.error(f"❌ 数据加载失败: {e}")
            return None, None, None, None
    
    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("🎓 DCAS 课程知识图谱")
        st.sidebar.markdown("---")
        
        # 数据概览
        if self.courses:
            st.sidebar.markdown("### 📊 数据概览")
            st.sidebar.metric("课程总数", len(self.courses))
            if self.embeddings is not None:
                st.sidebar.metric("向量维度", self.embeddings.shape[1])
            if self.graph is not None:
                st.sidebar.metric("图谱连接", len(self.graph.edges))
        
        st.sidebar.markdown("---")
        
        # 功能选择
        st.sidebar.markdown("### 🔍 功能导航")
        page = st.sidebar.selectbox(
            "选择功能",
            ["🏠 首页概览", "🔍 课程搜索", "🕸️ 知识图谱", "📊 主题分析", "🎯 相似度分析", "📈 统计报告"],
            index=0
        )
        
        return page
    
    def render_home(self):
        """渲染首页"""
        st.title("🎓 DCAS 课程知识图谱可视化平台")
        st.markdown("---")
        
        if not self.courses:
            st.error("❌ 没有找到数据，请先运行知识图谱构建脚本")
            return
        
        # 主要指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📚 课程总数",
                value=len(self.courses),
                delta=f"已处理"
            )
        
        with col2:
            if self.embeddings is not None:
                st.metric(
                    label="🔢 向量维度",
                    value=self.embeddings.shape[1],
                    delta="Qwen3-Embedding"
                )
        
        with col3:
            if self.graph is not None:
                st.metric(
                    label="🕸️ 图谱连接",
                    value=len(self.graph.edges),
                    delta=f"密度: {nx.density(self.graph):.3f}"
                )
        
        with col4:
            if self.analysis_data:
                unique_topics = len(self.analysis_data.get('topic_distribution', {}))
                st.metric(
                    label="🏷️ 主题数量",
                    value=unique_topics,
                    delta="已识别"
                )
        
        # 主题分布图表
        if self.analysis_data and 'topic_distribution' in self.analysis_data:
            st.markdown("### 📊 热门主题分布")
            
            topic_dist = self.analysis_data['topic_distribution']
            top_topics = dict(list(topic_dist.items())[:15])
            
            fig = px.bar(
                x=list(top_topics.values()),
                y=list(top_topics.keys()),
                orientation='h',
                title="前15个热门主题",
                labels={'x': '课程数量', 'y': '主题'},
                color=list(top_topics.values()),
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # 快速搜索
        st.markdown("### 🔍 快速搜索")
        search_term = st.text_input("搜索课程关键词", placeholder="输入课程名称、主题或描述关键词")
        
        if search_term:
            results = self.search_courses(search_term, top_k=5)
            if results:
                st.markdown(f"找到 {len(results)} 个相关课程：")
                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result['course_name']} (相关度: {result['score']})"):
                        st.write(f"**主题**: {', '.join(result['topics'][:5])}")
                        st.write(f"**描述**: {result['description']}")
            else:
                st.info("没有找到相关课程")
    
    def render_search(self):
        """渲染搜索页面"""
        st.title("🔍 课程搜索")
        st.markdown("---")
        
        if not self.courses:
            st.error("❌ 没有数据可供搜索")
            return
        
        # 搜索选项
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "搜索关键词",
                placeholder="输入课程名称、主题、描述关键词",
                help="支持模糊匹配，不区分大小写"
            )
        
        with col2:
            max_results = st.selectbox("最大结果数", [5, 10, 20, 50], index=1)
        
        # 高级搜索选项
        with st.expander("🔧 高级搜索选项"):
            col1, col2 = st.columns(2)
            
            with col1:
                # 主题过滤
                all_topics = set()
                for course in self.courses:
                    all_topics.update(course['topics'])
                selected_topics = st.multiselect(
                    "按主题过滤",
                    sorted(list(all_topics)),
                    help="选择一个或多个主题进行过滤"
                )
            
            with col2:
                # 搜索权重设置
                st.markdown("**搜索权重设置**")
                name_weight = st.slider("课程名称权重", 1, 20, 10)
                desc_weight = st.slider("描述权重", 1, 20, 5)
                topic_weight = st.slider("主题权重", 1, 20, 3)
        
        # 执行搜索
        if search_query:
            # 自定义搜索函数
            results = self.advanced_search(
                search_query, selected_topics, max_results,
                name_weight, desc_weight, topic_weight
            )
            
            if results:
                st.success(f"🎯 找到 {len(results)} 个相关课程")
                
                # 结果展示
                for i, result in enumerate(results, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"### {i}. {result['course_name']}")
                            st.write(f"**相关度评分**: {result['score']}")
                            st.write(f"**主题**: {', '.join(result['topics'][:5])}")
                            
                            # 描述（可展开）
                            with st.expander("查看课程描述"):
                                st.write(result['description'])
                        
                        with col2:
                            # 相似课程按钮
                            if st.button(f"查找相似课程", key=f"similar_{i}"):
                                similar_courses = self.find_similar_courses_by_index(result['course_index'], 5)
                                st.write("**相似课程:**")
                                for sim_course in similar_courses:
                                    st.write(f"• {sim_course['course_name']} ({sim_course['similarity']:.3f})")
                        
                        st.markdown("---")
            else:
                st.info("🤷‍♂️ 没有找到匹配的课程，请尝试其他关键词")
    
    def render_knowledge_graph(self):
        """渲染知识图谱页面"""
        st.title("🕸️ 知识图谱可视化")
        st.markdown("---")
        
        if self.graph is None:
            st.error("❌ 没有知识图谱数据")
            return
        
        # 图谱控制选项
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_nodes = st.slider("显示节点数", 10, min(100, len(self.graph.nodes)), 50)
        
        with col2:
            layout_type = st.selectbox(
                "布局算法",
                ["spring", "kamada_kawai", "circular", "random"],
                help="不同的布局算法会产生不同的可视化效果"
            )
        
        with col3:
            node_size_factor = st.slider("节点大小", 0.5, 3.0, 1.0, 0.1)
        
        # 主题筛选
        if self.courses:
            all_topics = set()
            for course in self.courses:
                all_topics.update(course['topics'])
            
            selected_topics_filter = st.multiselect(
                "按主题筛选节点",
                sorted(list(all_topics)),
                help="选择主题来高亮相关课程"
            )
        
        # 生成图谱可视化
        fig = self.create_interactive_graph(
            max_nodes, layout_type, node_size_factor, selected_topics_filter
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # 图谱统计信息
        st.markdown("### 📊 图谱统计")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("节点数", len(self.graph.nodes))
        with col2:
            st.metric("边数", len(self.graph.edges))
        with col3:
            st.metric("密度", f"{nx.density(self.graph):.4f}")
        with col4:
            st.metric("连通分量", nx.number_connected_components(self.graph))
        
        # 中心性分析
        st.markdown("### 🎯 中心性分析")
        
        # 计算中心性指标
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # 获取top课程
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**度中心性最高的课程**")
            for i, (node_id, centrality) in enumerate(top_degree, 1):
                course_name = self.graph.nodes[node_id]['name']
                st.write(f"{i}. {course_name[:50]}... ({centrality:.3f})")
        
        with col2:
            st.markdown("**介数中心性最高的课程**")
            for i, (node_id, centrality) in enumerate(top_betweenness, 1):
                course_name = self.graph.nodes[node_id]['name']
                st.write(f"{i}. {course_name[:50]}... ({centrality:.3f})")
    
    def render_topic_analysis(self):
        """渲染主题分析页面"""
        st.title("📊 主题分析")
        st.markdown("---")
        
        if not self.analysis_data:
            st.error("❌ 没有主题分析数据")
            return
        
        # 主题分布分析
        if 'topic_distribution' in self.analysis_data:
            topic_dist = self.analysis_data['topic_distribution']
            
            # 总体统计
            st.markdown("### 📈 主题分布概览")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总主题数", len(topic_dist))
            with col2:
                total_courses = sum(topic_dist.values())
                st.metric("课程-主题关联", total_courses)
            with col3:
                avg_per_topic = total_courses / len(topic_dist)
                st.metric("平均每主题课程数", f"{avg_per_topic:.1f}")
            with col4:
                max_count = max(topic_dist.values())
                st.metric("最热门主题课程数", max_count)
            
            # 主题分布图表
            st.markdown("### 📊 主题热度分布")
            
            # 选择显示的主题数量
            top_n = st.slider("显示前N个主题", 10, min(50, len(topic_dist)), 20)
            
            top_topics = dict(list(topic_dist.items())[:top_n])
            
            # 创建交互式柱状图
            fig = px.bar(
                x=list(top_topics.values()),
                y=list(top_topics.keys()),
                orientation='h',
                title=f"前{top_n}个热门主题分布",
                labels={'x': '课程数量', 'y': '主题'},
                color=list(top_topics.values()),
                color_continuous_scale='plasma'
            )
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # 主题词云（使用条形图模拟）
            st.markdown("### ☁️ 主题热度云图")
            
            # 创建饼图
            fig_pie = px.pie(
                values=list(top_topics.values()),
                names=list(top_topics.keys()),
                title=f"前{top_n}个主题占比分布"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # 聚类分析
        if 'cluster_analysis' in self.analysis_data and self.analysis_data['cluster_analysis']:
            st.markdown("### 🎯 聚类分析")
            
            cluster_data = self.analysis_data['cluster_analysis']
            num_clusters = self.analysis_data.get('num_clusters', len(cluster_data))
            
            st.write(f"系统识别出 **{num_clusters}** 个课程聚类")
            
            # 聚类详情
            for cluster_name, topics in cluster_data.items():
                with st.expander(f"📁 {cluster_name} - 主要主题"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**主题词频:**")
                        for topic, count in topics.items():
                            st.write(f"• {topic}: {count}")
                    
                    with col2:
                        # 该聚类的饼图
                        if topics:
                            fig_cluster = px.pie(
                                values=list(topics.values()),
                                names=list(topics.keys()),
                                title=f"{cluster_name} 主题分布"
                            )
                            st.plotly_chart(fig_cluster, use_container_width=True)
        
        # 主题相关性分析
        if self.embeddings is not None and self.courses:
            st.markdown("### 🔗 主题相关性矩阵")
            
            # 计算主题-主题相关性
            topic_correlation = self.calculate_topic_correlation()
            
            if topic_correlation is not None:
                fig_heatmap = px.imshow(
                    topic_correlation,
                    title="主题相关性热力图",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def render_similarity_analysis(self):
        """渲染相似度分析页面"""
        st.title("🎯 相似度分析")
        st.markdown("---")
        
        if self.embeddings is None or not self.courses:
            st.error("❌ 没有embedding数据")
            return
        
        # 课程选择
        st.markdown("### 🎯 选择目标课程")
        
        # 搜索课程
        search_course = st.text_input("搜索课程", placeholder="输入课程名称进行搜索")
        
        # 课程列表
        course_options = []
        if search_course:
            # 过滤课程
            for i, course in enumerate(self.courses):
                if search_course.lower() in course['course_name'].lower():
                    course_options.append((i, course['course_name']))
        else:
            # 显示所有课程（限制数量）
            course_options = [(i, course['course_name']) for i, course in enumerate(self.courses[:50])]
        
        if course_options:
            selected_course_idx = st.selectbox(
                "选择课程",
                options=[idx for idx, _ in course_options],
                format_func=lambda x: next(name for idx, name in course_options if idx == x),
                help="选择一个课程来分析其相似课程"
            )
            
            # 相似度设置
            col1, col2 = st.columns(2)
            with col1:
                similarity_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.5, 0.05)
            with col2:
                max_similar = st.slider("最大相似课程数", 5, 50, 20)
            
            # 计算相似度
            similar_courses = self.find_similar_courses_by_index(selected_course_idx, max_similar)
            
            # 过滤低于阈值的课程
            filtered_similar = [
                course for course in similar_courses 
                if course['similarity'] >= similarity_threshold
            ]
            
            # 显示目标课程信息
            st.markdown("### 📋 目标课程信息")
            target_course = self.courses[selected_course_idx]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**课程名称**: {target_course['course_name']}")
                st.write(f"**主题**: {', '.join(target_course['topics'])}")
                with st.expander("查看课程描述"):
                    st.write(target_course['course_description'])
            
            with col2:
                # 目标课程的主题分布
                if target_course['topics']:
                    fig_target = px.pie(
                        values=[1] * len(target_course['topics']),
                        names=target_course['topics'],
                        title="目标课程主题"
                    )
                    fig_target.update_traces(showlegend=False)
                    st.plotly_chart(fig_target, use_container_width=True)
            
            # 显示相似课程
            if filtered_similar:
                st.markdown(f"### 🔍 相似课程 (阈值 ≥ {similarity_threshold})")
                st.write(f"找到 {len(filtered_similar)} 个相似课程")
                
                # 相似度分布图
                similarities = [course['similarity'] for course in filtered_similar]
                fig_dist = px.histogram(
                    x=similarities,
                    bins=20,
                    title="相似度分布",
                    labels={'x': '相似度', 'y': '课程数量'}
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # 相似课程列表
                for i, similar_course in enumerate(filtered_similar, 1):
                    with st.expander(f"{i}. {similar_course['course_name']} (相似度: {similar_course['similarity']:.3f})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**主题**: {', '.join(similar_course['topics'][:5])}")
                            st.write(f"**描述**: {similar_course['description']}")
                        
                        with col2:
                            # 主题重叠分析
                            target_topics = set(target_course['topics'])
                            similar_topics = set(similar_course['topics'])
                            overlap = target_topics.intersection(similar_topics)
                            
                            if overlap:
                                st.write("**重叠主题**:")
                                for topic in overlap:
                                    st.write(f"• {topic}")
            else:
                st.info(f"没有找到相似度 ≥ {similarity_threshold} 的课程")
            
            # 降维可视化
            st.markdown("### 📈 相似度可视化")
            
            if len(filtered_similar) > 0:
                # 准备数据
                target_embedding = self.embeddings[selected_course_idx:selected_course_idx+1]
                similar_embeddings = np.array([
                    self.embeddings[course['course_index']] 
                    for course in filtered_similar
                ])
                all_embeddings = np.vstack([target_embedding, similar_embeddings])
                
                # 降维
                reduction_method = st.selectbox("降维方法", ["PCA", "t-SNE"])
                
                if reduction_method == "PCA":
                    reducer = PCA(n_components=2)
                else:
                    perplexity = min(30, len(all_embeddings) - 1)
                    reducer = TSNE(n_components=2, perplexity=max(2, perplexity), random_state=42)
                
                try:
                    reduced_embeddings = reducer.fit_transform(all_embeddings)
                    
                    # 创建散点图
                    labels = ['目标课程'] + [f"相似课程 {i}" for i in range(1, len(filtered_similar) + 1)]
                    colors = ['red'] + ['blue'] * len(filtered_similar)
                    
                    fig_scatter = go.Figure()
                    
                    # 添加目标课程
                    fig_scatter.add_trace(go.Scatter(
                        x=[reduced_embeddings[0, 0]],
                        y=[reduced_embeddings[0, 1]],
                        mode='markers',
                        marker=dict(size=15, color='red'),
                        name='目标课程',
                        text=[target_course['course_name']],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                    
                    # 添加相似课程
                    fig_scatter.add_trace(go.Scatter(
                        x=reduced_embeddings[1:, 0],
                        y=reduced_embeddings[1:, 1],
                        mode='markers',
                        marker=dict(size=10, color='blue', opacity=0.7),
                        name='相似课程',
                        text=[course['course_name'] for course in filtered_similar],
                        hovertemplate='%{text}<br>相似度: ' + 
                                    '<br>'.join([f"{course['similarity']:.3f}" for course in filtered_similar]) +
                                    '<extra></extra>'
                    ))
                    
                    fig_scatter.update_layout(
                        title=f"{reduction_method} 降维可视化",
                        xaxis_title=f"{reduction_method} 1",
                        yaxis_title=f"{reduction_method} 2"
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"降维可视化失败: {e}")
        else:
            st.info("请搜索并选择一个课程进行相似度分析")
    
    def render_statistics(self):
        """渲染统计报告页面"""
        st.title("📈 统计报告")
        st.markdown("---")
        
        if not self.courses:
            st.error("❌ 没有数据")
            return
        
        # 总体统计
        st.markdown("### 📊 数据集概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("课程总数", len(self.courses))
        
        with col2:
            if self.embeddings is not None:
                st.metric("向量维度", self.embeddings.shape[1])
        
        with col3:
            if self.graph is not None:
                density = nx.density(self.graph)
                st.metric("图谱密度", f"{density:.4f}")
        
        with col4:
            if self.analysis_data:
                total_topics = len(self.analysis_data.get('topic_distribution', {}))
                st.metric("识别主题数", total_topics)
        
        # 课程长度分布
        st.markdown("### 📏 课程描述长度分布")
        
        desc_lengths = [len(course['course_description']) for course in self.courses]
        
        fig_length = px.histogram(
            x=desc_lengths,
            bins=30,
            title="课程描述长度分布",
            labels={'x': '描述长度(字符)', 'y': '课程数量'}
        )
        st.plotly_chart(fig_length, use_container_width=True)
        
        # 主题数量分布
        st.markdown("### 🏷️ 每课程主题数量分布")
        
        topic_counts = [len(course['topics']) for course in self.courses]
        
        fig_topics = px.histogram(
            x=topic_counts,
            bins=max(topic_counts),
            title="每课程主题数量分布",
            labels={'x': '主题数量', 'y': '课程数量'}
        )
        st.plotly_chart(fig_topics, use_container_width=True)
        
        # 图谱分析
        if self.graph is not None:
            st.markdown("### 🕸️ 图谱拓扑分析")
            
            # 度分布
            degrees = [self.graph.degree(n) for n in self.graph.nodes()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_degree = px.histogram(
                    x=degrees,
                    bins=20,
                    title="节点度分布",
                    labels={'x': '度', 'y': '节点数量'}
                )
                st.plotly_chart(fig_degree, use_container_width=True)
            
            with col2:
                # 连通分量
                components = list(nx.connected_components(self.graph))
                component_sizes = [len(comp) for comp in components]
                
                fig_components = px.bar(
                    x=range(1, len(component_sizes) + 1),
                    y=component_sizes,
                    title="连通分量大小",
                    labels={'x': '分量ID', 'y': '节点数量'}
                )
                st.plotly_chart(fig_components, use_container_width=True)
        
        # 数据质量报告
        st.markdown("### ✅ 数据质量报告")
        
        # 检查数据完整性
        missing_names = sum(1 for course in self.courses if not course['course_name'])
        missing_descriptions = sum(1 for course in self.courses if not course['course_description'])
        missing_topics = sum(1 for course in self.courses if not course['topics'])
        
        quality_data = {
            "指标": ["课程名称完整性", "课程描述完整性", "主题标签完整性"],
            "完整数量": [
                len(self.courses) - missing_names,
                len(self.courses) - missing_descriptions,
                len(self.courses) - missing_topics
            ],
            "缺失数量": [missing_names, missing_descriptions, missing_topics],
            "完整率": [
                f"{(1 - missing_names/len(self.courses))*100:.1f}%",
                f"{(1 - missing_descriptions/len(self.courses))*100:.1f}%",
                f"{(1 - missing_topics/len(self.courses))*100:.1f}%"
            ]
        }
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True)
        
        # 导出功能
        st.markdown("### 💾 数据导出")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("导出课程列表"):
                courses_df = pd.DataFrame([
                    {
                        'course_name': course['course_name'],
                        'topics': ', '.join(course['topics']),
                        'description_length': len(course['course_description'])
                    }
                    for course in self.courses
                ])
                st.download_button(
                    label="下载CSV",
                    data=courses_df.to_csv(index=False),
                    file_name="courses_list.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("导出主题统计"):
                if self.analysis_data and 'topic_distribution' in self.analysis_data:
                    topic_df = pd.DataFrame([
                        {'topic': topic, 'count': count}
                        for topic, count in self.analysis_data['topic_distribution'].items()
                    ])
                    st.download_button(
                        label="下载CSV",
                        data=topic_df.to_csv(index=False),
                        file_name="topic_statistics.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("导出图谱摘要"):
                if self.graph is not None:
                    summary = {
                        'total_nodes': len(self.graph.nodes),
                        'total_edges': len(self.graph.edges),
                        'density': nx.density(self.graph),
                        'components': nx.number_connected_components(self.graph)
                    }
                    st.download_button(
                        label="下载JSON",
                        data=json.dumps(summary, indent=2),
                        file_name="graph_summary.json",
                        mime="application/json"
                    )
    
    # 辅助方法
    def search_courses(self, keyword, top_k=10):
        """搜索课程"""
        results = []
        keyword_lower = keyword.lower()
        
        for i, course in enumerate(self.courses):
            score = 0
            
            if keyword_lower in course['course_name'].lower():
                score += 10
            if keyword_lower in course['course_description'].lower():
                score += 5
            for topic in course['topics']:
                if keyword_lower in topic.lower():
                    score += 3
            
            if score > 0:
                results.append({
                    'course_index': i,
                    'course_name': course['course_name'],
                    'score': score,
                    'topics': course['topics'],
                    'description': course['course_description'][:200] + "..." if len(course['course_description']) > 200 else course['course_description']
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def advanced_search(self, query, topics_filter, max_results, name_weight, desc_weight, topic_weight):
        """高级搜索"""
        results = []
        query_lower = query.lower()
        
        for i, course in enumerate(self.courses):
            # 主题过滤
            if topics_filter:
                if not any(topic in course['topics'] for topic in topics_filter):
                    continue
            
            score = 0
            
            # 加权搜索
            if query_lower in course['course_name'].lower():
                score += name_weight
            if query_lower in course['course_description'].lower():
                score += desc_weight
            for topic in course['topics']:
                if query_lower in topic.lower():
                    score += topic_weight
            
            if score > 0:
                results.append({
                    'course_index': i,
                    'course_name': course['course_name'],
                    'score': score,
                    'topics': course['topics'],
                    'description': course['course_description'][:200] + "..." if len(course['course_description']) > 200 else course['course_description']
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]
    
    def find_similar_courses_by_index(self, target_idx, top_k=5):
        """根据索引查找相似课程"""
        if self.embeddings is None:
            return []
        
        similarities = cosine_similarity([self.embeddings[target_idx]], self.embeddings)[0]
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # 排除自己
        
        results = []
        for idx in similar_indices:
            results.append({
                'course_index': idx,
                'course_name': self.courses[idx]['course_name'],
                'similarity': float(similarities[idx]),
                'topics': self.courses[idx]['topics'],
                'description': self.courses[idx]['course_description'][:200] + "..." if len(self.courses[idx]['course_description']) > 200 else self.courses[idx]['course_description']
            })
        
        return results
    
    def create_interactive_graph(self, max_nodes, layout_type, node_size_factor, topic_filter):
        """创建交互式图谱"""
        if self.graph is None:
            return None
        
        # 节点选择逻辑
        if topic_filter:
            # 根据主题筛选节点
            filtered_nodes = []
            for node, data in self.graph.nodes(data=True):
                node_topics = data.get('topics', [])
                if any(topic in node_topics for topic in topic_filter):
                    filtered_nodes.append(node)
            
            if len(filtered_nodes) > max_nodes:
                # 如果筛选后还是太多，按度数选择
                degrees = dict(self.graph.degree())
                filtered_nodes = sorted(filtered_nodes, key=lambda x: degrees[x], reverse=True)[:max_nodes]
            
            subgraph = self.graph.subgraph(filtered_nodes)
        else:
            # 按度数选择top节点
            degrees = dict(self.graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph = self.graph.subgraph([node for node, _ in top_nodes])
        
        if len(subgraph.nodes) == 0:
            st.warning("没有找到匹配的节点")
            return None
        
        # 布局计算
        if layout_type == "spring":
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
        elif layout_type == "kamada_kawai":
            pos = nx.kamada_kawai_layout(subgraph)
        elif layout_type == "circular":
            pos = nx.circular_layout(subgraph)
        else:  # random
            pos = nx.random_layout(subgraph)
        
        # 创建边的traces
        edge_x = []
        edge_y = []
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # 创建节点的traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_sizes = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # 节点信息
            node_data = subgraph.nodes[node]
            node_name = node_data['name']
            node_topics = node_data.get('topics', [])
            
            node_text.append(node_name[:30] + "..." if len(node_name) > 30 else node_name)
            node_info.append(f"课程: {node_name}<br>主题: {', '.join(node_topics[:3])}")
            
            # 节点大小基于度数
            degree = subgraph.degree(node)
            node_sizes.append(10 + degree * node_size_factor * 2)
        
        # 创建图形
        fig = go.Figure()
        
        # 添加边
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='连接'
        ))
        
        # 添加节点
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_info,
            text=node_text,
            textposition="middle center",
            textfont=dict(size=8),
            marker=dict(
                size=node_sizes,
                color='lightblue',
                line=dict(width=2, color='blue')
            ),
            name='课程'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"课程知识图谱 ({len(subgraph.nodes)} 个节点)",
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="课程关联图谱 - 线条表示相似关系",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="grey", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def calculate_topic_correlation(self):
        """计算主题相关性矩阵"""
        try:
            # 获取所有主题
            all_topics = set()
            for course in self.courses:
                all_topics.update(course['topics'])
            
            topic_list = sorted(list(all_topics))
            
            # 限制主题数量以提高性能
            if len(topic_list) > 20:
                # 选择最流行的20个主题
                topic_counts = {}
                for course in self.courses:
                    for topic in course['topics']:
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                topic_list = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]
                topic_list = [topic for topic, _ in topic_list]
            
            # 创建主题向量
            topic_vectors = []
            for topic in topic_list:
                topic_courses = []
                for i, course in enumerate(self.courses):
                    if topic in course['topics']:
                        topic_courses.append(i)
                
                if topic_courses and self.embeddings is not None:
                    # 计算该主题的平均embedding
                    topic_embeddings = self.embeddings[topic_courses]
                    topic_vector = np.mean(topic_embeddings, axis=0)
                    topic_vectors.append(topic_vector)
                else:
                    topic_vectors.append(np.zeros(self.embeddings.shape[1]))
            
            if topic_vectors:
                topic_vectors = np.array(topic_vectors)
                correlation_matrix = cosine_similarity(topic_vectors)
                return correlation_matrix
            
        except Exception as e:
            st.error(f"主题相关性计算失败: {e}")
        
        return None
    
    def run(self):
        """运行仪表板"""
        # 加载数据
        self.graph, self.embeddings, self.courses, self.analysis_data = self.load_data()
        
        # 渲染侧边栏
        page = self.render_sidebar()
        
        # 根据选择的页面渲染内容
        if page == "🏠 首页概览":
            self.render_home()
        elif page == "🔍 课程搜索":
            self.render_search()
        elif page == "🕸️ 知识图谱":
            self.render_knowledge_graph()
        elif page == "📊 主题分析":
            self.render_topic_analysis()
        elif page == "🎯 相似度分析":
            self.render_similarity_analysis()
        elif page == "📈 统计报告":
            self.render_statistics()

def main():
    """主函数"""
    dashboard = CourseKnowledgeGraphDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()