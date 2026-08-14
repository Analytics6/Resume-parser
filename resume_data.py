import random
from pathlib import Path

import pandas as pd

random.seed(42)

CATEGORIES = [
    {
        "name": "Software Engineer",
        "titles": ["Software Engineer", "Full Stack Developer", "Backend Engineer", "Platform Engineer"],
        "skills": ["Python", "Java", "C++", "AWS", "Docker", "Kubernetes", "Microservices", "SQL", "REST APIs", "Agile"],
        "tools": ["GitHub", "Jenkins", "PostgreSQL", "Redis", "Terraform", "Linux"],
        "keywords": ["software", "developer", "python", "backend", "api", "cloud", "engineering"],
    },
    {
        "name": "Data Scientist",
        "titles": ["Data Scientist", "Machine Learning Engineer", "AI Research Analyst"],
        "skills": ["Python", "R", "SQL", "PyTorch", "TensorFlow", "scikit-learn", "NLP", "Time Series", "A/B Testing", "Statistics"],
        "tools": ["Jupyter", "MLflow", "Airflow", "Databricks", "Snowflake", "Spark"],
        "keywords": ["data", "machine learning", "analysis", "model", "python", "statistics", "ai"],
    },
    {
        "name": "Product Manager",
        "titles": ["Product Manager", "Senior Product Manager", "Growth Product Manager"],
        "skills": ["Roadmapping", "Stakeholder Management", "Analytics", "User Research", "SQL", "A/B Testing", "Sprint Planning", "Strategy", "Prioritization"],
        "tools": ["Jira", "Amplitude", "Mixpanel", "Notion", "Figma", "Tableau"],
        "keywords": ["product", "roadmap", "customer", "analytics", "growth", "strategy", "prioritization"],
    },
    {
        "name": "UX/UI Designer",
        "titles": ["UX Designer", "UI Designer", "Product Designer"],
        "skills": ["Figma", "Prototyping", "User Research", "Wireframing", "Design Systems", "Accessibility", "Interaction Design", "Adobe XD"],
        "tools": ["Figma", "Sketch", "InVision", "Maze", "Zeplin", "Miro"],
        "keywords": ["design", "ux", "ui", "prototype", "accessibility", "user research", "wireframes"],
    },
    {
        "name": "Cybersecurity Analyst",
        "titles": ["Cybersecurity Analyst", "Security Operations Analyst", "Threat Intelligence Analyst"],
        "skills": ["SIEM", "Incident Response", "Threat Hunting", "Network Security", "Penetration Testing", "Linux", "SOC", "Forensics", "IAM"],
        "tools": ["Splunk", "CrowdStrike", "SentinelOne", "Wireshark", "Qualys", "SOAR"],
        "keywords": ["security", "cyber", "incident", "threat", "network", "forensics", "siem"],
    },
    {
        "name": "Marketing Manager",
        "titles": ["Marketing Manager", "Brand Manager", "Digital Marketing Lead"],
        "skills": ["SEO", "SEM", "CRM", "Content Strategy", "Brand Positioning", "Campaign Management", "Social Media", "Budget Planning", "Email Marketing"],
        "tools": ["HubSpot", "Google Ads", "Meta Business Suite", "Mailchimp", "Salesforce", "Google Analytics"],
        "keywords": ["marketing", "brand", "campaign", "seo", "digital", "customer acquisition", "analytics"],
    },
    {
        "name": "Financial Analyst",
        "titles": ["Financial Analyst", "Corporate Finance Analyst", "Investment Analyst"],
        "skills": ["Financial Modeling", "Excel", "Forecasting", "Valuation", "Portfolio Analysis", "Risk Management", "SQL", "Budgeting"],
        "tools": ["Excel", "Power BI", "SAP", "Oracle", "Tableau", "Bloomberg"],
        "keywords": ["finance", "budget", "forecast", "valuation", "investment", "models", "risk"],
    },
    {
        "name": "HR Specialist",
        "titles": ["HR Specialist", "Talent Acquisition Specialist", "People Operations Manager"],
        "skills": ["Recruitment", "Employee Relations", "Onboarding", "HRIS", "Performance Management", "Compensation", "Compliance", "Hiring"],
        "tools": ["Workday", "Greenhouse", "BambooHR", "ADP", "Slack", "Zoom"],
        "keywords": ["hr", "human resources", "recruitment", "talent", "employee", "onboarding", "hiring"],
    },
    {
        "name": "Healthcare Administrator",
        "titles": ["Healthcare Administrator", "Clinical Operations Manager", "Healthcare Program Manager"],
        "skills": ["Compliance", "Patient Experience", "Clinical Workflow", "Healthcare Operations", "Quality Improvement", "EMR", "Policy Management", "Vendor Coordination"],
        "tools": ["Epic", "Cerner", "Power BI", "Excel", "Meditech", "Microsoft Teams"],
        "keywords": ["healthcare", "clinical", "patient", "operations", "care", "compliance", "medical"],
    },
    {
        "name": "Sales Representative",
        "titles": ["Sales Representative", "Business Development Manager", "Account Executive"],
        "skills": ["Sales Strategy", "Negotiation", "CRM", "Lead Generation", "Pipeline Management", "Account Management", "Presentation", "Closing"],
        "tools": ["Salesforce", "HubSpot", "LinkedIn Sales", "ZoomInfo", "Outlook", "Excel"],
        "keywords": ["sales", "account", "pipeline", "negotiation", "customer", "lead generation", "revenue"],
    },
]

FIRST_NAMES = [
    "Ava", "Lucas", "Mia", "Noah", "Emma", "Liam", "Olivia", "Ethan", "Sophia", "Mason",
    "Charlotte", "James", "Amelia", "Benjamin", "Harper", "Elijah", "Evelyn", "Henry", "Abigail", "Leo",
    "Nora", "Jack", "Isabella", "Daniel", "Ella", "Samuel", "Grace", "Alexander", "Chloe", "David",
    "Scarlett", "Joseph", "Luna", "Michael", "Aria", "Sebastian", "Hannah", "Matthew", "Avery", "Isaac"
]

LAST_NAMES = [
    "Patel", "Nguyen", "Martin", "Kim", "Brown", "Johnson", "Lee", "Garcia", "Wilson", "Smith",
    "Davis", "Taylor", "Moore", "Jackson", "Thomas", "Harris", "Walker", "Young", "Allen", "Scott"
]

COMPANIES = [
    "NorthStar Labs", "Summit Works", "Lakeview Health", "BrightPath Group", "Nova Metrics",
    "Mercury Systems", "BluePeak Solutions", "Harbor Analytics", "Evergreen Consulting",
    "Crestline Partners", "Signal Forge", "Pioneer Enterprise", "Vertex Studios", "Oak & Pine",
    "CloudBridge", "Greenline Ventures", "EchoWorks", "Validus Capital", "PrimeCore", "Horizon One"
]


def _random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _build_resume_text(category, name, title, years_experience, role_focus):
    skills = category["skills"]
    tools = category["tools"]
    skill_line = ", ".join(random.sample(skills, 6))
    tool_line = ", ".join(random.sample(tools, 4))
    company = random.choice(COMPANIES)
    clients = random.sample(["enterprise clients", "cross-functional teams", "remote-first teams", "startup stakeholders", "regional partners"], 2)
    results = [
        f"{years_experience} years of experience delivering {role_focus} across {clients[0]} and {clients[1]}",
        f"improved operational efficiency by {random.randint(18, 68)}%",
        f"drove measurable impact across product, delivery, and customer experience",
        f"led strategic initiatives impacting {random.randint(5, 25)} teams and {random.randint(1000, 15000)} stakeholders",
    ]
    result_text = ". ".join(random.sample(results, 3))
    paragraph_1 = (
        f"{name} is a {title} with {years_experience} years of experience helping organizations improve {role_focus}. "
        f"At {company}, they partnered with business stakeholders to design and execute solutions that improved performance, workflow, and customer outcomes. "
        f"{result_text}."
    )
    paragraph_2 = (
        f"Core capabilities include {skill_line}. Proficient with {tool_line}. "
        f"Strong background in process design, stakeholder communication, reporting, and performance optimization. "
        f"They are passionate about turning business goals into action plans that deliver measurable results in fast-moving environments."
    )
    return paragraph_1 + "\n\n" + paragraph_2


def generate_resume_dataset(num_per_category: int = 100, output_csv: str = "data/resumes.csv") -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        for i in range(num_per_category):
            name = _random_name()
            title = random.choice(category["titles"])
            years = random.randint(2, 14)
            role_focus = random.choice(category["keywords"])
            text = _build_resume_text(category, name, title, years, role_focus)
            rows.append(
                {
                    "resume_id": f"{category['name'].lower().replace(' ', '_')}_{i:03d}",
                    "name": name,
                    "category": category["name"],
                    "title": title,
                    "years_experience": years,
                    "resume_text": text,
                }
            )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def example_queries() -> dict:
    return {
        "Software Engineer": "Python backend engineer with cloud APIs, Docker, SQL, and microservices experience",
        "Data Scientist": "Machine learning scientist with Python, PyTorch, forecasting, and NLP research",
        "Product Manager": "Product manager with roadmap planning, analytics, customer research, and growth metrics",
        "UX/UI Designer": "UX product designer with wireframes, prototyping, accessibility, and design systems",
        "Cybersecurity Analyst": "Cybersecurity analyst handling threat hunting, SIEM, incident response, and network security",
        "Marketing Manager": "Marketing manager leading digital campaigns, SEO, CRM strategy, and brand growth",
        "Financial Analyst": "Financial analyst building forecasting models, valuation, Excel analysis, and risk reporting",
        "HR Specialist": "HR specialist managing hiring, recruitment, employee relations, and onboarding programs",
        "Healthcare Administrator": "Healthcare administrator improving patient operations, compliance, and clinical workflows",
        "Sales Representative": "Sales representative with account management, lead generation, negotiation, and revenue growth",
    }


def category_keywords() -> dict:
    return {category["name"]: category["keywords"] for category in CATEGORIES}
