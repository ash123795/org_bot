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
        "mission": "Platform for youth to educate others on diseases, raise awareness on health issues, and bridge the gap on disparities. foster creativity and provide students a place to express their voices. Together, we can make a difference in global health.",
        "fields": ["global health awareness", "disease education", "creativity and expression"],
    },
    {
        "name": "NeuroNexus",
        "mission": "Promotes mental health awareness and supports individuals who struggle with it, through virtual fundraisers and hosting informational guest speakers!",
        "fields": ["neuroscience", "mental health awareness", "supporting patients", "fundraising", "education/advocation"],
    },
    {
        "name": "BRAIN Project",
        "mission": "aim to inspire curiosity and expand understanding of neuroscience by fostering education, collaboration, and creativity. Empower future innovators to explore the brain and make meaningful contributions to the field.",
        "fields": ["teen research", "neuroscience education", "community service"],
    },
    {
        "name": "Discover Mind",
        "mission": "Discover Mind is a place for anyone curious about the human mind. The platform hosts neuroscience contests and essay competitions to inspire learning and connection. ",
        "fields": ["neuroscience education", "competition", "collaboration"],
    },
    {
        "name": "Simply Neuroscience",
        "mission": "imply Neuroscience is an international, student-led nonprofit organization dedicated to fostering students' interdisciplinary interests in the brain, specifically through neuroscience and psychology education, outreach, and awareness.",
        "fields": ["psychology", "neuroscience", "advocation/awareness"],
    },
    {
        "name": "Academic Concordia",
        "mission": "Academia Concordia is a student-led umbrella organization that connects and empowers high school clubs across fields like science, arts, leadership, and mental health. One of our key branches is the Psychology Club, which leads mental health awareness campaigns, organizes brain science workshops, and runs peer support initiatives.",
        "fields": ["arts", "mental health", "psychology", "law-making", "diplomacy"]
    }
]

# LLM setup
config_list = [
    {
        'model': 'gpt-3.5-turbo',
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