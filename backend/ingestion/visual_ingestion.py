import os 
import io 
import uuid 
import base64
import logging
import asyncio
import fitz
from PIL import Image 
from pdf2image import convert_from_path 
from qdrant_client.models import PointStruct
from langhcain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from core.config import GEMINI_API_KEY
from core.embeddings import  embeddings
from core.qdrant_client import qdrant


logger = logging.getLogger(__name__)

VISUAL_COLLECTION = "visual_descriptions"

vision_llm = ChatGoogleGenerativeAI(
    model = "gemeini-2.0-falsh",
    temperature = 0.0,
    google_api_key = GEMINI_API_KEY,
    request_timeout=60,

)


VISUAL_PROMPT = """You are an expert document and diagram analyst.
Analyze this PDF page image in detail and extract all visual, tabular, and graphical information.
Follow these strict rules:
1. Tables: Reproduce the full structure (column headers, all rows, numbers, metrics, footnotes).
2. Charts / Graphs: Describe the chart type, axis labels, trends, peaks, anomalies, and key data points.
3. Diagrams / Architecture: List all components, arrows/flow directions, interactions, and labels.
4. Formulas / Equations: Transcribe any mathematical expressions accurately.
5. Captions / Notes: Include figure titles, table numbers, and surrounding context.
Be precise and factual. Output a clean, detailed summary of all visual information on this page."""

