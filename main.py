from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentType

import os

load_dotenv()

huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
repo_id="mistralai/Mistral-7B-Instruct-v0.3"


# def generate_pet_name():
#     print(huggingface_token)


#     llm=HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=huggingface_token)
#     name=llm.invoke("What do you know about Vietnam")

#     return name

def langchain_agent():
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=huggingface_token)

    tools=load_tools(["wikipedia", "llm-math"], llm = llm)

    agent=initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    result = agent.run(
        "What is the average of dogs? Multiply the age by 3"
    )

    print(result)


if __name__ == "__main__":
    # print(generate_pet_name())
    langchain_agent()
