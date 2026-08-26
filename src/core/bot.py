"""
Real-time Farm Advisory Bot Core Engine.
Combines Pinecone RAG, OpenWeatherMap API, and Gemini 2.5 Flash streaming.
"""
import os
import re
import requests
from typing import Iterator


# ── AI Safety: Prompt Injection Defense ──────────────────────────────────────
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(prior|previous|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+)?(you\s+are\s+)?", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:\s*", re.IGNORECASE),
]


def sanitize_input(text: str, max_length: int = 50000) -> str:
    """Strip prompt injection patterns and enforce length limits."""
    if not text:
        return text
    text = text[:max_length]
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text

# Cached singletons — building the embedding client and vector-store handle
# involves network setup; doing it per-query added ~1-2s latency per message.
_cached_vector_store = None


def _get_vector_store():
    """Lazily build and reuse the Pinecone vector store across queries."""
    global _cached_vector_store
    if _cached_vector_store is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        index_name = os.getenv("PINECONE_INDEX", "farm-advisory")
        _cached_vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
        )
    return _cached_vector_store

from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables in app.py

def get_current_weather(location: str = "Al Ain, AE") -> str:
    """Fetch current weather from OpenWeatherMap."""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "Weather API key missing."
        
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if response.status_code == 200:
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            desc = data["weather"][0]["description"]
            return f"{temp}°C, Humidity: {humidity}%, Conditions: {desc}"
        else:
            return f"Weather unavailable: {data.get('message', 'Error')}"
    except Exception as e:
        return f"Weather service error: {e}"


def get_farm_knowledge(query: str) -> str:
    """Retrieve relevant farming knowledge from Pinecone."""
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key or pinecone_key == "your_pinecone_api_key_here":
        return "Warning: Pinecone API key missing. RAG context unavailable."
        
    try:
        vector_store = _get_vector_store()
        results = vector_store.similarity_search(query, k=3)
        if not results:
            return "No relevant documents found in the knowledge base."
        # Include source metadata so the model's advice stays traceable.
        parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Doc {i} | source: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Warning: Failed to retrieve from Pinecone: {e}"


def stream_advisory_response(query: str, location: str = "Al Ain, AE") -> Iterator[str]:
    """
    Core function for real-time streaming advice.
    Combines weather, RAG context, and the farmer's query.
    """
    # AI Safety: sanitize user input before passing to LLM
    query = sanitize_input(query)
    weather = get_current_weather(location)
    knowledge_context = get_farm_knowledge(query)
    
    # Initialize Gemini with streaming enabled
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        streaming=True
    )
    
    system_prompt = f"""You are an expert UAE Agricultural Advisor. 
Provide immediate, actionable, and practical advice to farmers.

CURRENT WEATHER IN {location}: {weather}

RETRIEVED FARMING KNOWLEDGE:
{knowledge_context}

Follow these rules:
1. Be direct and concise. Farmers are busy in the field.
2. Incorporate the current weather into your advice if relevant (e.g., watering schedules).
3. If the retrieved knowledge contains a solution, use it. If not, use your general expertise but note that it's general advice.
4. Format with clear bullet points.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    # Stream the response chunks as they arrive; a mid-stream provider error
    # must degrade to a readable message instead of crashing the UI stream.
    try:
        for chunk in llm.stream(messages):
            yield chunk.content
    except Exception as e:
        yield f"\n\n⚠️ The advisory service hit an error while generating the response: {e}"
