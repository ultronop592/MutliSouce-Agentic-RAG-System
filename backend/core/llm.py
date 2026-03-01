from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import GEMINI_API_KEY

# Using Gemini 2.5 Flash for extreme speed and higher rate limits on Free Tier.
# This addresses the user's request for faster responses ("too much time in replying").
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=2048,
    google_api_key=GEMINI_API_KEY,
)