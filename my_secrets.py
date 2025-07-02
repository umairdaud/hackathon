import os
from dotenv import load_dotenv
from rich import print

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print(
        "GEMINI_API_KEY is not set in the environment variables."
    )
    exit(1)

gemini_api_url = os.getenv("GEMINI_API_URL")
if not gemini_api_url:
    print(
        "GEMINI_API_URL is not set in the environment variables."
    )
    exit(1)

gemini_api_model = os.getenv("GEMINI_API_MODEL")
if not gemini_api_model:
    print(
        "GEMINI_API_MODEL is not set in the environment variables."
    )
    exit(1)

# Getting the configuration values from environment variables
deepseek_api_key = os.getenv("OPENROUTER_API_KEY")         
deepseek_api_url = os.getenv("OPENROUTER_BASEURL")         
deepseek_api_model = os.getenv("OPENROUTER_DEEPSEEK_R1_MODEL")     

# Check if any of the required environment variables are missing
if not deepseek_api_key or not deepseek_api_url or not deepseek_api_model:
    print("Please set the environment variables OPENROUTER_API_KEY, OPENROUTER_API_URL, and OPENROUTER_API_MODEL.")
    exit(1)



class MySecrets:
    def __init__(self):
        self.gemini_api_key = gemini_api_key
        self.gemini_api_url = gemini_api_url
        self.gemini_api_model = gemini_api_model

        # Assigning the environment variables to instance attributes
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_api_url = deepseek_api_url
        self.deepseek_api_model = deepseek_api_model