from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import autogen
import os

app = FastAPI()

load_dotenv()  # Loads variables from .env

api_key = os.getenv("OPENAI_API_KEY")

# Allow requests from anywhere (or restrict to Netlify domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Pre-defined orgs
organizations = [
    {
        "name": "NeuroYouth",
        "mission": "Promotes neuroscience education and outreach for high school students",
        "fields": ["neuroscience", "education", "public speaking"],
        "age_range": "14-18"
    },
    {
        "name": "Girls Who Code",
        "mission": "Closing the gender gap in tech by teaching girls computer science",
        "fields": ["computer science", "gender equity"],
        "age_range": "13-18"
    },
    {
        "name": "BioPolicy",
        "mission": "Teaching policy about medical malpractice",
        "fields": ["biology", "politics", "law"],
        "age_range": "20-25"
    }
]

# LLM setup
config_list = [
    {
        'model': 'gpt-4',
        'api_key': api_key
    }
]

llm_config = {
    "seed": 42,
    "config_list": config_list,
    "temperature": 0
}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="Student_Input",
    human_input_mode="NEVER",  # no terminal input; handled via API
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "web"},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved. Otherwise, reply CONTINUE"""
)

@app.post("/match")
async def match_orgs(request: Request):
    data = await request.json()
    interest_summary = data.get("summary", "")

    task = f""" 
You are an agent that helps match students to organizations. You will receive a student's interest summary.

Interest summary: {interest_summary}

Then, choose the best matching organization from this list: {organizations}

For each match, explain *why* you chose it based on their mission, fields, and age range.
Output should look like:
Best Org Name — reason
"""

    chat_result = user_proxy.initiate_chat(
        assistant,
        message=task
    )

    return {"result": chat_result.summary}