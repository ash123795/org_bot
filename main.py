from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent

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
        "name": "Eunoia",
        "mission": "youth platform to educate others on diseases, raise health issue awareness, bridge disparity gaps, foster creativity and provide students a place to express their voices.",
        "fields": ["global health", "disease education", "creative expression"],
    },
    {
        "name": "NeuroNexus",
        "mission": "Promote mental health awareness and support through events and support initiatives",
        "fields": ["neuroscience", "mental health", "education", "fundraising"],
    },
    {
        "name": "BRAIN Project",
        "mission": "Inspire curiosity of neuroscience by fostering education, collaboration, and creativity.",
        "fields": ["teen research", "neuroscience", "community service"],
    },
    {
        "name": "Discover Mind",
        "mission": "Hosts neuroscience contests and essay competitions to inspire learning and connection. ",
        "fields": ["neuroscience", "competition", "collaboration"],
    },
    {
        "name": "Simply Neuroscience",
        "mission": "Dedicated to fostering students' interdisciplinary interests in neuroscience and psychology through education, outreach, and awareness.",
        "fields": ["psychology", "neuroscience", "awareness"],
    },
    {
        "name": "Academic Concordia",
        "mission": "Connects clubs to advance mental health, psychology, and leadership. One of their key branches is the Psychology Club, which leads mental health awareness campaigns and other support initiatives.",
        "fields": ["psychology", "mental health", "leadership", "policy"]
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
    max_consecutive_auto_reply=3,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "web", "use_docker": False},
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)