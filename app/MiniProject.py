import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO

# -----------------------------
# 🎯 APP CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Workforce Intelligence Platform", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .skill-tag {
        background: #00b894;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏢 Workforce Intelligence Platform</div>', unsafe_allow_html=True)
st.write("Comprehensive employee analytics with **AI-powered insights**")

# -----------------------------
# 📂 LOAD DATA
# -----------------------------
st.sidebar.header("📁 Load Your Work Logs")

uploaded_file = st.sidebar.file_uploader("Upload your csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # -----------------------------
    # 🤖 NER MODEL SETUP
    # -----------------------------
    @st.cache_resource
    def load_ner_model():
        return pipeline("ner", 
                       model="dslim/bert-base-NER", 
                       aggregation_strategy="simple")
    
    def extract_skills_with_ner(text):
        """
        Use Hugging Face NER model to extract skills and technologies from text
        """
        try:
            ner_pipe = load_ner_model()
            entities = ner_pipe(text)
            
            skills_detected = []
            tech_keywords = [
                'python', 'java', 'javascript', 'css', 'html', 'react', 'angular', 'vue',
                'node', 'sql', 'database', 'api', 'docker', 'kubernetes', 'aws', 'cloud',
                'azure', 'gcp', 'testing', 'qa', 'design', 'development', 'frontend', 
                'backend', 'mobile', 'ios', 'android', 'machine learning', 'ai',
                'analysis', 'research', 'documentation', 'debugging', 'deployment'
            ]
            
            for entity in entities:
                entity_text = entity['word'].lower().strip()
                entity_type = entity['entity_group']
                entity_score = entity['score']
                
                # Only consider high-confidence entities and relevant types
                if entity_score > 0.85:
                    # Check for technology/skill keywords in any entity
                    for keyword in tech_keywords:
                        if keyword in entity_text:
                            skills_detected.append(keyword)
                    
                    # Also capture ORG, MISC, PRODUCT entities as they often contain tech names
                    if entity_type in ['ORG', 'MISC', 'PRODUCT'] and len(entity_text) > 2:
                        skills_detected.append(entity_text)
            
            return list(set(skills_detected))
        
        except Exception as e:
            st.error(f"NER processing error: {e}")
            return []

    def calculate_employee_scores(employee_data):
        score = 0
        # Completion rate (40% weight)
        completion_rate = (employee_data['Status'] == 'Done').mean()
        score += completion_rate * 40
        
        # Task diversity (20% weight)
        task_diversity = employee_data['TaskCategory'].nunique() / df['TaskCategory'].nunique()
        score += min(task_diversity * 20, 20)
        
        # Project diversity (20% weight)
        project_diversity = employee_data['Project'].nunique() / df['Project'].nunique()
        score += min(project_diversity * 20, 20)
        
        # Work volume (20% weight)
        volume_score = min(len(employee_data) / (len(df) / df['Employee'].nunique()) * 20, 20)
        score += volume_score
        
        return round(score, 1)

    leaderboard_data = []
    for employee in df['Employee'].unique():
        emp_data = df[df['Employee'] == employee]
        score = calculate_employee_scores(emp_data)
        
        leaderboard_data.append({
            'Employee': employee,
            'Score': score,
            'Tasks': len(emp_data),
            'Projects': emp_data['Project'].nunique(),
            'Completion Rate': f"{(emp_data['Status'] == 'Done').mean() * 100:.1f}%",
            'Task Categories': emp_data['TaskCategory'].nunique()
        })
    
    leaderboard_df = pd.DataFrame(leaderboard_data).sort_values('Score', ascending=False)

    def custom_metric(title, value):
        st.markdown(f"""
            <div class="metric-card">
                <p style="font-weight: bold; font-size: 1.1em; margin: 0; opacity: 0.9;">{title}</p>
                <p style="font-size: 2.2em; font-weight: 700; margin: 5px 0 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">{value}</p>
            </div>
        """, unsafe_allow_html=True)

    with st.spinner("🔄 Processing data and generating AI insights..."):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🏆 Leaderboard", "🔧 Skills", "📊 Reports", "👥 Team Optimizer", "🤖 NER Analysis"])

        with tab1:
            st.markdown('<div class="feature-card"><h3>📊 Organizational Overview</h3></div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                custom_metric("Total Employees", df['Employee'].nunique())
            with col2:
                custom_metric("Total Tasks", len(df))
            with col3:
                custom_metric("Active Projects", df['Project'].nunique())
            with col4:
                completion_rate = (df['Status'] == 'Done').mean() * 100
                custom_metric("Overall Completion", f"{completion_rate:.1f}%")
            
            st.markdown("---")
            st.subheader("🎯 Skill Distribution Across Organization")
            
            # Analyze skills across organization using NER
            all_skills = []
            skill_matrix = {}

            for employee in df['Employee'].unique():
                emp_data = df[df['Employee'] == employee]
                emp_skills = []
                
                for desc in emp_data['Description'].dropna():
                    # Use REAL NER instead of keyword matching
                    emp_skills.extend(extract_skills_with_ner(desc))
                
                unique_skills = list(set(emp_skills))
                skill_matrix[employee] = {
                    'skills': unique_skills,
                    'skill_count': len(unique_skills),
                    'tasks_analyzed': len(emp_data)
                }
                all_skills.extend(unique_skills)

            if all_skills:
                skill_counts = pd.Series(all_skills).value_counts()
                fig_skills = px.bar(skill_counts, 
                                  title="🚀 Most Common Skills Detected by AI",
                                  color=skill_counts.values,
                                  color_continuous_scale='viridis')
                fig_skills.update_layout(showlegend=False)
                st.plotly_chart(fig_skills, use_container_width=True)
            else:
                st.info("🤖 No skills detected yet. The AI is analyzing your data...")

        with tab2:
            st.markdown('<div class="feature-card"><h3>🏆 Worker Leaderboard</h3></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Performance Ranking")
                # Style the dataframe
                styled_df = leaderboard_df.reset_index(drop=True).style.background_gradient(
                    subset=['Score'], cmap='YlOrBr'
                )
                st.dataframe(styled_df, use_container_width=True)
            
            with col2:
                st.subheader("🎖️ Top Performers")
                top_3 = leaderboard_df.head(3)
                for i, (_, employee) in enumerate(top_3.iterrows()):
                    medal = ["🥇", "🥈", "🥉"][i]
                    st.markdown(f"""
                    <div style="padding: 15px; background: linear-gradient(135deg, #ffeaa7, #fab1a0); 
                                border-radius: 10px; margin: 10px 0; text-align: center;">
                        <h4 style="margin: 0; color: #2d3436;">{medal} {employee['Employee']}</h4>
                        <h3 style="margin: 5px 0; color: #e17055;">{employee['Score']} pts</h3>
                    </div>
                    """, unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="feature-card"><h3>🔧 Skills Dashboard</h3></div>', unsafe_allow_html=True)
            
            st.subheader("👨‍💼 Employee Skill Matrix")
            skill_matrix_df = pd.DataFrame([
                {'Employee': emp, 'Skill Count': data['skill_count'], 'Skills': ', '.join(data['skills'])}
                for emp, data in skill_matrix.items()
            ]).sort_values('Skill Count', ascending=False)
            
            # Display skills as tags
            st.subheader("🎯 Skills Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                if all_skills:
                    st.write("**Detected Skills:**")
                    for skill in list(set(all_skills))[:15]:  # Show first 15 unique skills
                        st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
            
            with col2:
                st.dataframe(skill_matrix_df, use_container_width=True)

        with tab4:
            st.markdown('<div class="feature-card"><h3>📊 Report Generator</h3></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_employee_report = st.selectbox("Select Employee for Report", df['Employee'].unique())
            
            with col2:
                report_type = st.selectbox("Report Type", ["Performance Summary", "Skills Analysis", "Full Evaluation"])
            
            if st.button("📄 Generate Report", type="primary"):
                emp_data = df[df['Employee'] == selected_employee_report]
                skills_data = skill_matrix[selected_employee_report]
                
                st.subheader(f"📋 Employee Report: {selected_employee_report}")
                
                # Performance Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    custom_metric("Performance Score", f"{calculate_employee_scores(emp_data)}/100")
                with col2:
                    custom_metric("Tasks Completed", f"{(emp_data['Status'] == 'Done').sum()}/{len(emp_data)}")
                with col3:
                    custom_metric("Skills Detected", skills_data['skill_count'])
                with col4:
                    custom_metric("Projects Involved", emp_data['Project'].nunique())
                
                # Detailed Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🛠️ AI-Detected Skills")
                    if skills_data['skills']:
                        for skill in skills_data['skills']:
                            st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                    else:
                        st.info("No specific skills detected by AI")
                
                with col2:
                    st.subheader("📈 Performance Insights")
                    completion_rate = (emp_data['Status'] == 'Done').mean() * 100
                    if completion_rate > 80:
                        st.success("**🌟 High Performer**: Excellent completion rate!")
                    elif completion_rate > 60:
                        st.info("**💪 Solid Contributor**: Good completion rate")
                    else:
                        st.warning("**📚 Needs Support**: Lower completion rate detected")
                    
                    if skills_data['skill_count'] >= 4:
                        st.success("**🎯 Versatile**: Diverse skill set")
                    else:
                        st.info("**🎯 Specialized**: Focused skill set")
                
                # Task Breakdown
                st.subheader("📝 Recent Tasks")
                st.dataframe(emp_data[['TicketID', 'Project', 'TaskCategory', 'Description', 'Status']].head(10))

                # Generate report content for download
                completion_rate_for_report = (emp_data['Status'] == 'Done').mean() * 100
                report_content = f"""
Employee Report: {selected_employee_report}

Performance Summary:
  Performance Score: {calculate_employee_scores(emp_data)}/100
  Tasks Completed: {(emp_data['Status'] == 'Done').sum()}/{len(emp_data)}
  Skills Detected: {skills_data['skill_count']}
  Projects Involved: {emp_data['Project'].nunique()}

AI-Detected Skills:
{'  - ' + '\\n  - '.join([skill.capitalize() for skill in skills_data['skills']]) if skills_data['skills'] else '  No specific skills detected by AI'}

Performance Insights:
  Completion Rate: {completion_rate_for_report:.1f}%
  {'High Performer: Excellent completion rate!' if completion_rate_for_report > 80 else 'Solid Contributor: Good completion rate' if completion_rate_for_report > 60 else 'Needs Support: Lower completion rate detected'}
  {'Versatile: Diverse skill set' if skills_data['skill_count'] >= 4 else 'Specialized: Focused skill set'}

Recent Tasks (Top 10):
{emp_data[['TicketID', 'Project', 'TaskCategory', 'Description', 'Status']].head(10).to_string()}
                """
                st.download_button(
                    label="💾 Download Report",
                    data=report_content,
                    file_name=f"{selected_employee_report}_report.txt",
                    mime="text/plain"
                )

        with tab5:
            st.markdown('<div class="feature-card"><h3>👥 Team Optimizer</h3></div>', unsafe_allow_html=True)
            
            st.subheader("📋 Project Team Analysis")
            
            project_teams = {}
            for project in df['Project'].unique():
                project_data = df[df['Project'] == project]
                team_members = project_data['Employee'].unique()
                
                # Analyze team composition
                team_skills = []
                for member in team_members:
                    if member in skill_matrix:
                        team_skills.extend(skill_matrix[member]['skills'])
                
                unique_skills = list(set(team_skills))
                
                project_teams[project] = {
                    'team_size': len(team_members),
                    'team_members': list(team_members),
                    'skills_covered': unique_skills,
                    'skill_gaps': [skill for skill in (set(all_skills) if all_skills else []) if skill not in unique_skills][:5]
                }
            
            # Display project teams
            for project, data in project_teams.items():
                with st.expander(f"🚀 {project} (Team: {data['team_size']} members)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**👥 Team Members:**")
                        for member in data['team_members']:
                            st.write(f"• {member}")
                        
                        st.write("**✅ Skills Covered:**")
                        for skill in data['skills_covered'][:8]:
                            st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                    
                    with col2:
                        st.write("**💡 Recommendations:**")
                        if len(data['skill_gaps']) > 0:
                            st.write("**Consider adding:**")
                            for gap in data['skill_gaps'][:3]:
                                st.markdown(f"<div style='color: #e17055; font-weight: bold;'>⚡ {gap}</div>", unsafe_allow_html=True)
                        else:
                            st.success("✅ Well-rounded team composition!")
                        
                        # Team balance score
                        avg_tasks = len(df[df['Project'] == project]) / data['team_size']
                        if avg_tasks > 10:
                            st.warning("📊 Team may be understaffed")
                        elif avg_tasks < 3:
                            st.info("📊 Team may be overstaffed")
                        else:
                            st.success("📊 Good team size for workload")
            
            # Team optimization suggestions
            st.subheader("🔧 Optimization Suggestions")
            
            # Find employees with unique skills
            unique_skill_holders = {}
            if all_skills:
                for skill in set(all_skills):
                    holders = [emp for emp, data in skill_matrix.items() if skill in data['skills']]
                    if len(holders) == 1:  # Only one person has this skill
                        unique_skill_holders[skill] = holders[0]
            
            if unique_skill_holders:
                st.error("🚨 Critical Skills Dependency:")
                for skill, holder in list(unique_skill_holders.items())[:3]:
                    st.write(f"• **{skill}** only known by **{holder}**")
            
            # Cross-training opportunities
            st.info("💡 Cross-Training Opportunities:")
            busy_employees = leaderboard_df.head(3)['Employee'].tolist()
            for emp in busy_employees[:2]:
                st.write(f"• Consider training backup for **{emp}**'s responsibilities")

        with tab6:
            st.markdown('<div class="feature-card"><h3>🤖 NER Analysis</h3></div>', unsafe_allow_html=True)
            
            st.write("See the actual Named Entity Recognition in action")
            
            sample_text = st.selectbox(
                "Select a task description to analyze:",
                df['Description'].head(10).tolist()
            )
            
            if st.button("🔍 Run NER Analysis", type="primary"):
                with st.spinner("Analyzing text with AI NER model..."):
                    ner_results = extract_skills_with_ner(sample_text)
                    
                    st.subheader("🎯 AI-Detected Skills & Technologies")
                    if ner_results:
                        for skill in ner_results:
                            st.markdown(f'<span class="skill-tag" style="background: #6c5ce7;">{skill}</span>', unsafe_allow_html=True)
                    else:
                        st.info("No specific skills detected by AI NER model")
                    
                    # Show raw NER results
                    st.subheader("📋 Raw NER Output")
                    try:
                        ner_pipe = load_ner_model()
                        raw_entities = ner_pipe(sample_text)
                        
                        if raw_entities:
                            entities_df = pd.DataFrame(raw_entities)
                            st.dataframe(entities_df)
                        else:
                            st.info("No entities detected in this text")
                    except Exception as e:
                        st.error(f"Error in NER analysis: {e}")

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1>🚀 Welcome to the Workforce Intelligence Platform!</h1>
        <h3>Unlock insights from your employee work logs with AI-powered analytics</h3>
        <br>
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; color: white; margin: 20px 0;">
            <h4>🎯 Features Include:</h4>
            <p>• 🤖 AI-Powered Skill Detection</p>
            <p>• 🏆 Employee Performance Ranking</p>
            <p>• 🔧 Team Optimization Insights</p>
            <p>• 📊 Automated Report Generation</p>
        </div>
        <p>To get started, please upload your work log data in CSV format using the sidebar on the left.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# 📈 FOOTER
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <h3>Workforce Intelligence Platform</h3>
    <p>Built By Team C • Powered by AI • NLP Mini Project</p>
</div>
""", unsafe_allow_html=True)
