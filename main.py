from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

def generate_pet_name():
    huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    print(huggingface_token)

    repo_id="mistralai/Mistral-7B-Instruct-v0.3"
    llm=HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=huggingface_token)
    name=llm.invoke("What do you know about Vietnam")

    return name

if __name__ == "__main__":
    print(generate_pet_name())
