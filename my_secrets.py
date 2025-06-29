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

class MySecrets:
    def __init__(self):
        self.gemini_api_key = gemini_api_key
        self.gemini_api_url = gemini_api_url
        self.gemini_api_model = gemini_api_model