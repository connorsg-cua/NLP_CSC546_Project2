# Synthetic dataset generator for "worker evaluation and profiling" project
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)
np.random.seed(42)

N = 300  # number of synthetic records

employees = [
    "Ana Morales","Luis Pérez","Carla Soto","Diego Ruiz","Elena García",
    "Mateo Torres","Sofía Rojas","Pedro Núñez","María López","Jorge Castillo",
    "Lucía Fernández","Andrés Gómez","Valeria Díaz","Tomás Silva","Isabella Cruz"
]

task_categories = [
    "security","testing","documentation","frontend","backend",
    "data","infrastructure","devops","recruiting","design","support"
]

projects = [
    "Project Atlas","Project Boreal","Project Cobalt","Project Delta","Project Ember",
    "Project Flux","Project Gaia","Project Helix"
]

statuses = ["Planned","In Progress","Blocked","Done"]

def random_week(start_week=1, end_week=4):
    return f"Week {random.randint(start_week, end_week)}"

def random_dates(start_date=datetime(2025, 1, 6), span_days=60):
    # pick a random start date within the range; duration 1–5 days
    s = start_date + timedelta(days=random.randint(0, span_days))
    d_days = random.randint(1, 5)
    e = s + timedelta(days=d_days)
    return s.date(), e.date()

def gen_description(emp, cat):
    verbs = {
    "security": [
        "thoroughly fixed the vulnerabilities found in the system",
        "investigated potential security breaches and established preventative measures",
        "patched security flaws to ensure compliance with best practices",
        "reviewed security protocols to identify areas for improvement"
    ],
    "testing": [
        "wrote comprehensive tests for new features to ensure functionality",
        "executed regression tests to verify that existing features remained unaffected",
        "automated test cases for critical functionalities to streamline the testing process",
        "validated application behavior under various conditions to enhance reliability"
    ],
    "documentation": [
        "updated the technical documentation for improved clarity and accessibility",
        "wrote detailed guidelines for best practices and coding standards",
        "created a thorough README for easy onboarding of new developers",
        "documented key processes and workflows to maintain knowledge continuity"
    ],
    "frontend": [
        "implemented a responsive UI for enhanced user experience across devices",
        "refactored existing components of the application for better performance",
        "fixed CSS issues in various browsers to ensure consistent appearance",
        "built a landing page for a new product launch that captured user engagement"
    ],
    "backend": [
        "implemented a robust API for seamless communication between services",
        "optimized database queries in complex transactions to reduce latency",
        "refactored the service architecture to improve scalability and maintainability",
        "added a new endpoint to support advanced data analytics features"
    ],
    "data": [
        "cleaned the dataset for improved accuracy in reporting and analysis",
        "built a data pipeline for efficient data ingestion from multiple sources",
        "analyzed key metrics on user behavior to inform product decisions",
        "created a real-time dashboard for monitoring system performance indicators"
    ],
    "infrastructure": [
        "configured a server for optimal performance under load conditions",
        "provisioned cloud resources for scalability in peak usage periods",
        "hardened the configuration of the firewall to strengthen security posture",
        "migrated the service to a more robust infrastructure platform"
    ],
    "devops": [
        "set up continuous integration pipelines for automated testing and deployment",
        "fixed build failures across multiple environments to ensure stability",
        "added comprehensive monitoring to the deployment process for better visibility",
        "containerized applications to enhance portability and scalability"
    ],
    "recruiting": [
        "interviewed candidates for key positions to assess their suitability",
        "screened resumes for potential hires that matched our company culture",
        "scheduled panel interviews to ensure a thorough evaluation process",
        "coordinated the offer process with the hiring team to finalize selections"
    ],
    "design": [
        "created high-fidelity mockups for enhanced visual storytelling",
        "updated the style guide for consistency in branding across platforms",
        "ran usability tests on new features to gather user feedback for improvements",
        "designed icons for key functionalities to enhance user interface clarity"
    ],
    "support": [
        "resolved complex tickets for users experiencing critical issues",
        "triaged incidents on the platform to prioritize urgent responses",
        "answered inquiries about product features promptly and accurately",
        "escalated cases for advanced troubleshooting where necessary"
    ]
    }
    objects = [
    "authentication module for secure user access management",
    "payment service enabling seamless transactions and invoicing",
    "landing page designed to attract and convert visitors",
    "ETL job that extracts, transforms, and loads data efficiently",
    "mobile app improving user engagement and accessibility on-the-go",
    "logging system capturing event data for monitoring and analysis",
    "KPI dashboard providing real-time insights into business performance",
    "search API facilitating user queries with quick, accurate results",
    "network configuration enhancing data flow and security",
    "access control mechanisms ensuring proper user permissions",
    "onboarding flow designed to help new users navigate the system",
    "report generator creating customizable reports for stakeholders"
    ]
    v = random.choice(verbs[cat])
    o = random.choice(objects)
    return f"{emp} {v} the {o}"

def progress_from_status(status):
    if status == "Done":
        return 100
    if status == "Planned":
        return random.choice([0, 5, 10])
    if status == "Blocked":
        return random.randint(0, 30)
    # In Progress
    return random.randint(20, 90)

def estimate_hours(cat, status):
    base = {
        "security": 10, "testing": 6, "documentation": 4, "frontend": 8, "backend": 9,
        "data": 8, "infrastructure": 9, "devops": 7, "recruiting": 5, "design": 6, "support": 3
    }[cat]
    mult = {"Planned": 0.2, "Blocked": 0.4, "In Progress": 0.8, "Done": 1.0}[status]
    noise = np.clip(np.random.normal(1.0, 0.25), 0.5, 1.5)
    return round(base * mult * noise, 1)

rows = []
for i in range(1, N+1):
    emp = random.choice(employees)
    cat = random.choice(task_categories)
    proj = random.choice(projects)
    status = random.choices(statuses, weights=[0.15, 0.45, 0.1, 0.3], k=1)[0]
    prog = progress_from_status(status)
    week = random_week()
    start_date, end_date = random_dates()
    desc = gen_description(emp, cat)
    hours = estimate_hours(cat, status)
    manager_note = random.choice([
        "Good progress, continue next week.",
        "Needs clarification from stakeholder.",
        "Blocked by dependency, escalated.",
        "Ready for review.",
        "Pair with teammate for faster delivery."
    ])
    ticket_id = f"TKT-{i:04d}"
    rows.append({
        "TicketID": ticket_id,
        "Employee": emp,
        "Project": proj,
        "TaskCategory": cat,
        "Description": desc,
        "Week": week,
        "Status": status,
        "ProgressPct": prog,
        "EstimatedHours": hours,
        "StartDate": start_date.isoformat(),
        "EndDate": end_date.isoformat(),
        "ManagerNote": manager_note
    })

df = pd.DataFrame(rows)

# Save files
out_dir = Path(".")  # change to your desired folder
csv_path = out_dir / "synthetic_work_logs.csv"
json_path = out_dir / "synthetic_work_logs.json"

df.to_csv(csv_path, index=False)
df.to_json(json_path, orient="records", lines=False)

print("Saved:", csv_path, json_path, "Shape:", df.shape)
print(df.head(5))

