"""
Visual Ingestion Module
-----------------------
Extracts and summarizes tables, charts, graphs, and diagrams from PDF pages
using PyMuPDF for visual detection and Gemini 2.0 Flash Vision for structured descriptions.
"""

import os
import io
import uuid
import base64
import logging
import asyncio
import fitz  # PyMuPDF
from PIL import Image
from pdf2image import convert_from_path
from qdrant_client.models import PointStruct
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from core.config import GEMINI_API_KEY
from core.embeddings import embeddings
from core.qdrant_client import qdrant

logger = logging.getLogger(__name__)

VISUAL_COLLECTION = "visual_descriptions"

# Dedicated fast vision LLM instance (Gemini 2.0 Flash)
vision_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=GEMINI_API_KEY,
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


def has_visual_content(page: fitz.Page) -> bool:
    """Detect if a PDF page contains images, tables, or complex vector drawings."""
    try:
        # Check 1: Embedded raster images
        if len(page.get_images(full=False)) > 0:
            return True

        # Check 2: Table grid lines detected by PyMuPDF
        tables = page.find_tables()
        if tables.tables and len(tables.tables) > 0:
            return True

        # Check 3: Vector drawings (charts, flowcharts, architecture boxes typically have >=10 vector paths)
        drawings = page.get_drawings()
        if len(drawings) >= 10:
            return True

        return False
    except Exception as e:
        logger.warning("Visual check failed for page: %s", e)
        return False


def _render_page_to_base64(file_path: str, page_number: int) -> str:
    """Render a specific 1-indexed PDF page to a base64-encoded JPEG image."""
    images = convert_from_path(
        file_path,
        first_page=page_number,
        last_page=page_number,
        dpi=150,  # 150 DPI provides sharp text for Gemini Vision while keeping byte size small
    )
    if not images:
        raise ValueError(f"Could not render page {page_number}")

    buffered = io.BytesIO()
    images[0].convert("RGB").save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def _describe_single_page(
    file_path: str, page_number: int, source_filename: str
) -> dict | None:
    """Send a page image to Gemini Vision and return the structured description."""
    try:
        b64_image = await asyncio.to_thread(_render_page_to_base64, file_path, page_number)

        message = HumanMessage(
            content=[
                {"type": "text", "text": VISUAL_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
            ]
        )

        response = await vision_llm.ainvoke([message])
        description = response.content.strip()

        if not description or len(description) < 20:
            return None

        # Format chunk header so it integrates seamlessly with RAG context and citations
        chunk_text = (
            f"[Source: {source_filename} | Page: {page_number} | VISUAL ANALYSIS]\n"
            f"{description}"
        )

        return {
            "page": page_number,
            "text": chunk_text,
            "source_filename": source_filename,
            "source_file": file_path,
        }
    except Exception as e:
        logger.warning(
            "Failed to describe visual content on page %d of '%s': %s",
            page_number, source_filename, e,
        )
        return None


async def ingest_pdf_visual(file_path: str, source_filename: str) -> int:
    """
    Scans PDF for pages with visual elements, generates descriptions with Gemini Vision,
    computes embeddings, and upserts them into Qdrant.
    """
    try:
        doc = fitz.open(file_path)
        visual_page_nums = []

        for idx, page in enumerate(doc):
            if has_visual_content(page):
                visual_page_nums.append(idx + 1)  # 1-indexed

        doc.close()

        if not visual_page_nums:
            logger.info("No visual elements detected in '%s'", source_filename)
            return 0

        logger.info(
            "Visual elements detected on pages %s in '%s'. Generating Gemini Vision descriptions...",
            visual_page_nums, source_filename,
        )

        # Describe visual pages concurrently (bounded to 5 to respect API rate limits)
        semaphore = asyncio.Semaphore(5)

        async def _bounded_describe(p_num: int):
            async with semaphore:
                return await _describe_single_page(file_path, p_num, source_filename)

        tasks = [_bounded_describe(p) for p in visual_page_nums]
        results = await asyncio.gather(*tasks)
        valid_items = [r for r in results if r is not None]

        if not valid_items:
            return 0

        # Embed all descriptions
        texts = [item["text"] for item in valid_items]
        vectors = await asyncio.to_thread(embeddings.embed_documents, texts)

        # Build Qdrant points
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={
                    "text": valid_items[i]["text"],
                    "page": valid_items[i]["page"],
                    "source_file": valid_items[i]["source_file"],
                    "source_filename": valid_items[i]["source_filename"],
                    "collection": VISUAL_COLLECTION,
                    "is_visual": True,
                },
            )
            for i in range(len(valid_items))
        ]

        # Upsert into visual_descriptions collection
        await asyncio.to_thread(
            qdrant.upsert,
            collection_name=VISUAL_COLLECTION,
            points=points,
        )

        logger.info(
            "Successfully ingested %d visual page descriptions for '%s'",
            len(points), source_filename,
        )
        return len(points)

    except Exception as e:
        logger.error("Visual ingestion failed for '%s': %s", source_filename, e)
        return 0
