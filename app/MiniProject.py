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

st.title("🏢 Workforce Intelligence Platform")
st.write("Comprehensive employee analytics with **AI-powered insights**")

# -----------------------------
# 📂 LOAD DATA
# -----------------------------
st.sidebar.header("📁 Load Your Work Logs")

uploaded_file = st.sidebar.file_uploader("Upload your csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Display dataset info
    st.subheader("📊 Organizational Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", df['Employee'].nunique())
    with col2:
        st.metric("Total Tasks", len(df))
    with col3:
        st.metric("Active Projects", df['Project'].nunique())
    with col4:
        completion_rate = (df['Status'] == 'Done').mean() * 100
        st.metric("Overall Completion", f"{completion_rate:.1f}%")
    
    # -----------------------------
    # 🏆 1. WORKER LEADERBOARD
    # -----------------------------
    st.header("🏆 Worker Leaderboard")
    
    # Calculate performance scores for each employee
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
    
    # Create leaderboard
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
    
    # Display leaderboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Performance Ranking")
        st.dataframe(leaderboard_df.reset_index(drop=True), use_container_width=True)
    
    with col2:
        st.subheader("Top Performers")
        top_3 = leaderboard_df.head(3)
        for i, (_, employee) in enumerate(top_3.iterrows()):
            medal = ["🥇", "🥈", "🥉"][i]
            st.write(f"{medal} **{employee['Employee']}** - {employee['Score']} pts")
    
    # -----------------------------
    # 🔧 2. SKILLS DASHBOARD
    # -----------------------------
    st.header("🔧 Skills Dashboard")
    
    # Skill extraction function
    def extract_skills_from_text(text):
        skills_keywords = {
            'testing': ['test', 'testing', 'qa', 'quality', 'comprehensive tests'],
            'design': ['design', 'mockup', 'ui', 'ux', 'visual'],
            'frontend': ['frontend', 'css', 'javascript', 'browser', 'ui'],
            'backend': ['backend', 'api', 'server', 'database'],
            'support': ['support', 'troubleshoot', 'escalate', 'help'],
            'management': ['manage', 'coordinate', 'lead', 'organize'],
            'documentation': ['document', 'write', 'create docs'],
            'analysis': ['analyze', 'research', 'investigate'],
            'development': ['develop', 'build', 'create', 'implement']
        }
        
        detected_skills = []
        text_lower = text.lower()
        
        for skill, keywords in skills_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_skills.append(skill)
        
        return list(set(detected_skills))
    
    # Analyze skills across organization
    all_skills = []
    skill_matrix = {}
    
    for employee in df['Employee'].unique():
        emp_data = df[df['Employee'] == employee]
        emp_skills = []
        
        for desc in emp_data['Description'].dropna():
            emp_skills.extend(extract_skills_from_text(desc))
        
        unique_skills = list(set(emp_skills))
        skill_matrix[employee] = {
            'skills': unique_skills,
            'skill_count': len(unique_skills),
            'tasks_analyzed': len(emp_data)
        }
        all_skills.extend(unique_skills)
    
    # Skills visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Skill Distribution")
        skill_counts = pd.Series(all_skills).value_counts()
        fig_skills = px.bar(skill_counts, title="Most Common Skills Across Organization")
        st.plotly_chart(fig_skills, use_container_width=True)
    
    with col2:
        st.subheader("Employee Skill Matrix")
        skill_matrix_df = pd.DataFrame([
            {'Employee': emp, 'Skill Count': data['skill_count'], 'Skills': ', '.join(data['skills'])}
            for emp, data in skill_matrix.items()
        ]).sort_values('Skill Count', ascending=False)
        
        st.dataframe(skill_matrix_df, use_container_width=True)
    
    # -----------------------------
    # 📊 3. REPORT GENERATOR
    # -----------------------------
    st.header("📊 Report Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_employee_report = st.selectbox("Select Employee for Report", df['Employee'].unique())
    
    with col2:
        report_type = st.selectbox("Report Type", ["Performance Summary", "Skills Analysis", "Full Evaluation"])
    
    if st.button("📄 Generate Report"):
        emp_data = df[df['Employee'] == selected_employee_report]
        skills_data = skill_matrix[selected_employee_report]
        
        st.subheader(f"📋 Employee Report: {selected_employee_report}")
        
        # Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Performance Score", f"{calculate_employee_scores(emp_data)}/100")
        with col2:
            st.metric("Tasks Completed", f"{(emp_data['Status'] == 'Done').sum()}/{len(emp_data)}")
        with col3:
            st.metric("Skills Detected", skills_data['skill_count'])
        with col4:
            st.metric("Projects Involved", emp_data['Project'].nunique())
        
        # Detailed Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛠️ Detected Skills")
            if skills_data['skills']:
                for skill in skills_data['skills']:
                    st.write(f"• {skill.capitalize()}")
            else:
                st.write("No specific skills detected in task descriptions")
        
        with col2:
            st.subheader("📈 Performance Insights")
            completion_rate = (emp_data['Status'] == 'Done').mean() * 100
            if completion_rate > 80:
                st.success("**High Performer**: Excellent completion rate!")
            elif completion_rate > 60:
                st.info("**Solid Contributor**: Good completion rate")
            else:
                st.warning("**Needs Support**: Lower completion rate detected")
            
            if skills_data['skill_count'] >= 4:
                st.success("**Versatile**: Diverse skill set")
            else:
                st.info("**Specialized**: Focused skill set")
        
        # Task Breakdown
        st.subheader("📝 Recent Tasks")
        st.dataframe(emp_data[['TicketID', 'Project', 'TaskCategory', 'Description', 'Status']].head(10))
    
    # -----------------------------
    # 👥 4. TEAM OPTIMIZER
    # -----------------------------
    st.header("👥 Team Optimizer")
    
    # Project analysis
    st.subheader("Project Team Analysis")
    
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
            'skill_gaps': [skill for skill in skill_counts.index if skill not in unique_skills][:5]
        }
    
    # Display project teams
    for project, data in project_teams.items():
        with st.expander(f"📋 {project} (Team: {data['team_size']} members)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Team Members:**")
                for member in data['team_members']:
                    st.write(f"• {member}")
                
                st.write("**Skills Covered:**")
                for skill in data['skills_covered'][:8]:
                    st.write(f"✓ {skill}")
            
            with col2:
                st.write("**Recommendations:**")
                if len(data['skill_gaps']) > 0:
                    st.write("**Consider adding:**")
                    for gap in data['skill_gaps'][:3]:
                        st.write(f"⚡ {gap}")
                else:
                    st.success("✅ Well-rounded team composition!")
                
                # Team balance score
                avg_tasks = len(df[df['Project'] == project]) / data['team_size']
                if avg_tasks > 10:
                    st.info("📊 Team may be understaffed")
                elif avg_tasks < 3:
                    st.info("📊 Team may be overstaffed")
                else:
                    st.success("📊 Good team size for workload")
    
    # Team optimization suggestions
    st.subheader("🔧 Optimization Suggestions")
    
    # Find employees with unique skills
    unique_skill_holders = {}
    for skill in set(all_skills):
        holders = [emp for emp, data in skill_matrix.items() if skill in data['skills']]
        if len(holders) == 1:  # Only one person has this skill
            unique_skill_holders[skill] = holders[0]
    
    if unique_skill_holders:
        st.warning("🚨 Critical Skills Dependency:")
        for skill, holder in list(unique_skill_holders.items())[:3]:
            st.write(f"• **{skill}** only known by **{holder}**")
    
    # Cross-training opportunities
    st.info("💡 Cross-Training Opportunities:")
    busy_employees = leaderboard_df.head(3)['Employee'].tolist()
    for emp in busy_employees[:2]:
        st.write(f"• Consider training backup for **{emp}**'s responsibilities")

else:
    st.info("👆 Please upload your 'tickets_final.csv' file to begin analysis")

# -----------------------------
# 📈 FOOTER
# -----------------------------
st.markdown("---")
st.caption("Workforce Intelligence Platform • Built By Team C • NLP Mini Project")

# Add some styling
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.3);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
        backdrop-filter: blur(10px);
    }
    
    /* Dark mode specific adjustments */
    [data-theme="dark"] div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.15);
        border: 1px solid rgba(28, 131, 225, 0.5);
    }
    
    /* Ensure text is readable in both themes */
    div[data-testid="stMetric"] > div > div {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)