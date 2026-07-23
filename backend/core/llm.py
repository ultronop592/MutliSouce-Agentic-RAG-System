from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import GEMINI_API_KEY

# temperature=0.0 — fully deterministic, eliminates creative deviation / hallucination.
# Gemini 2.5 Flash is used for speed on free tier.
# request_timeout=45 — if Gemini API hangs (rate limit, network blip), we abort after 45s
# rather than blocking the server indefinitely. 45s is generous for complex prompts.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,           # <- was 0.3; 0.0 = no creativity, grounded only
    max_tokens=2048,
    google_api_key=GEMINI_API_KEY,
    request_timeout=45,        # <- abort hanging LLM calls after 45s (prevents server stall)
)