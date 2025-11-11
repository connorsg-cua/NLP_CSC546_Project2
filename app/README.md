# How to Run the App Locally

1. Download MiniProject.py from this folder
2. Create a virtual environment in your IDE (PyCharm, VS Code, etc.)
3. Install dependencies:
   pip install streamlit pandas transformers torch plotly numpy
4. Run the app:
   streamlit run MiniProject.py
5. Use the synthetic dataset (synthetic_work_logs.csv) when prompted

The app will open in your browser at http://localhost:8501


🚀 App Features Explained
What This App Does:
This app analyzes work logs to help understand employee performance and skills using real AI-powered analysis.

Key Features:
📊 Organizational Overview
Dashboard view of key metrics

Skill distribution across the organization

Completion rates and project statistics

🏆 Employee Leaderboard
Ranks employees by performance score

Shows top performers with medals

Tracks project involvement and task completion

🔧 Skills Dashboard
AI-powered skill detection using Named Entity Recognition (NER)

Visual skill tags and employee skill matrix

Real-time skill analysis from task descriptions

📊 Report Generator
Generates individual employee reports

Shows completion rates and task counts

Highlights strengths and areas for improvement

Downloadable reports in text format

👥 Team Optimizer
Checks if teams have the right skills

Finds missing skills in projects

Suggests team improvements and cross-training

🤖 NER Analysis Tab (NEW!)
See the AI in action with real-time entity detection

View raw NER output from Hugging Face models

Test different task descriptions to see what the AI detects

Understand how skills are automatically extracted

🧠 How the AI Works:
Uses Hugging Face NER Model (dslim/bert-base-NER)

Automatically extracts technologies, skills, and entities

Processes natural language in task descriptions

No manual keyword lists - learns from context!

How to Use:
Upload your CSV file (use the synthetic data first)

The AI will automatically analyze everything using NER

Explore the different tabs to see insights

Check the NER Analysis tab to see the AI in action
