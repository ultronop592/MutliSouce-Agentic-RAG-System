from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import GEMINI_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=1024,
    google_api_key=GEMINI_API_KEY,
)