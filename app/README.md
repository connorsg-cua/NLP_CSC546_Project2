# How to Run the App Locally

1. Download MiniProject.py from this folder
2. Create a virtual environment in your IDE (PyCharm, VS Code, etc.)
3. Install dependencies:
   pip install streamlit pandas transformers torch plotly numpy
4. Run the app:
   streamlit run MiniProject.py
5. Use the synthetic dataset (synthetic_work_logs.csv) when prompted

The app will open in your browser at http://localhost:8501


## 🚀 App Features Explained

### What This App Does:
This app analyzes work logs to help understand employee performance and skills using AI.

### Key Features:

1. **📊 Employee Leaderboard**
   - Ranks employees by performance score
   - Shows who completes the most tasks
   - Tracks project involvement

2. **🔧 Skills Detector** 
   - Automatically finds skills from task descriptions
   - Shows what technologies employees are using
   - Creates a skills map for your team

3. **📈 Performance Reports**
   - Generates individual employee reports
   - Shows completion rates and task counts
   - Highlights strengths and areas for improvement

4. **👥 Team Analyzer**
   - Checks if teams have the right skills
   - Finds missing skills in projects
   - Suggests team improvements

5. **🤖 Smart Text Analysis (NER)**
   - The app uses AI to read task descriptions
   - It automatically finds important information like:
     - Technologies used (Python, CSS, testing, etc.)
     - Project components mentioned
     - Skills demonstrated
   - This happens automatically when you upload your data!

### How to Use:
1. Upload your CSV file (use the synthetic data first)
2. The app will automatically analyze everything
3. Explore the different tabs to see insights
