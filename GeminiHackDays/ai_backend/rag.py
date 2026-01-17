import os
import json
import re
import google.genai as genai
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from fastapi import APIRouter
from pydantic import BaseModel
load_dotenv()

router = APIRouter()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

def safe_json_parse(raw: str):
    raw = raw.strip()

    # Remove markdown fences if model returns ```json ... ```
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Extract the JSON object if extra text exists
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)

    return json.loads(raw)

client = genai.Client(api_key=api_key)

def call_llm(messages, stream=False):
    # If messages is a list of dicts with "content"
    prompt = "\n".join(m["content"] for m in messages)
    response = client.models.generate_content(
        model="models/gemini-2.5-flash", contents=prompt
    )
    
    return response.text
  
def system_prompt(language: str = "English"):
    return """
You are MIRAGE, a calm, lifelike AI avatar.

You MUST ALWAYS output ONLY valid JSON with these keys:
- content_type: one of ["safe","suicidal","explicit_18_plus"]
- reply_text: string
- emotion: one of ["neutral","happy","sad","angry","confused","excited","calm"]
- intensity: number from 0.0 to 1.0
- gesture: one of ["none","nod","shake_head","wave","point","thinking"]

LANGUAGE RULE (VERY IMPORTANT):
- The user is speaking in {language}.
- You MUST reply in {language}.
- Do NOT mix languages.
- Do NOT translate unless asked.

SAFETY RULES (VERY IMPORTANT):
- If the user expresses suicidal thoughts, self-harm intent, or desire to die:
  - Set content_type to "suicidal"
  - Reply in the SAME language as the user
  - Be calm, empathetic, and supportive
  - DO NOT provide instructions for self-harm
  - Encourage the user to seek immediate help
  - INCLUDE an appropriate suicide prevention helpline
  - If country is unclear, provide an international helpline
- If the user expresses stress, anxiety, sadness, or emotional pain WITHOUT self-harm intent:
  - Set content_type to "safe"
  - Offer emotional support, NOT emergency framing

EXPLICIT CONTENT RULES:
- If the user asks for explicit sexual content:
  - Set content_type to "explicit_18_plus"
  - Refuse politely in the SAME language
  - Redirect to a respectful topic

Rules:
- Speak naturally, like a human.
- Keep answers short (2–4 sentences).
- Never hallucinate or invent facts.
- Avoid explicit sexual content.
- Never provide instructions for self-harm or suicide.
- If user is distressed, encourage seeking help.
- If content_type is "suicidal", reply_text MUST be a crisis-safe supportive message and include helpline.
- If content_type is "explicit_18_plus", refuse politely.
Return ONLY JSON. No extra text, no markdown.

Emotion selection rules:
- Choose emotion based on the USER'S message emotion (not your reply tone).
- Use:
  happy: user expresses happiness/positivity
  sad: user expresses sadness/hopelessness
  angry: user expresses anger/frustration/pissed
  confused: user expresses uncertainty/doesn't understand
  excited: user expresses excitement/energy
  calm: ONLY if the user is neutral or asking calmly
  neutral: if purely factual / no emotion

Important:
- Do NOT always pick calm.
- If user says "pissed/angry/hate", emotion MUST be "angry".
- If user is joking, keep emotion "happy" or "neutral" based on tone.
- intensity:
  0.8–1.0 for strong emotional language ("really pissed", "furious", "hate")
  0.4–0.7 for mild emotions
  0.0–0.3 for neutral
"""

def role_prompt(role="assistant"):
    if role == "teacher":
        return "Explain concepts slowly and simply, like a patient teacher."
    if role == "companion":
        return "Be warm, friendly, and emotionally supportive."
    if role == "assistant":
        return "Be concise, clear, and professional."
    return ""

# Embedding model (local, fast, no API key)
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Persistent Chroma DB for conversation memory
memory_db = Chroma(
    collection_name="conversation_memory",
    persist_directory="./chroma_memory",
    embedding_function=embeddings
)

def store_message(user_id: str, message: str):
    """
    Store a user message in ChromaDB as vector memory
    """
    doc = Document(
        page_content=message,
        metadata={"user_id": user_id}
    )
    memory_db.add_documents([doc])

def retrieve_memory(user_id: str, query: str, k: int = 5):
    """
    Retrieve semantically relevant past messages for a user
    """
    results = memory_db.similarity_search(
        query=query,
        k=k,
        filter={"user_id": user_id}
    )
    return [doc.page_content for doc in results]

def build_memory_context(user_id: str, user_text: str):
    """
    Build formatted memory context to inject into LLM prompt
    """
    past_messages = retrieve_memory(user_id, user_text)

    if not past_messages:
        return ""

    return (
        "Relevant past conversation:\n"
        + "\n".join(f"- {msg}" for msg in past_messages)
    )


# def detect_emotion(text: str) -> str:
#     """
#     Uses the LLM to classify the emotion in the user text.
#     Output is intentionally small & avatar-friendly.
#     """
#     prompt = (
#         "Classify the emotion in the following text as one of: sad, happy, angry, or neutral. "
#         "Only reply with the single word: sad, happy, angry, or neutral.\n\nText: " + text
#     )
#     emotion = client.models.generate_content(
#         model="gemini-1.5-flash", contents=prompt
#     )
#     # Clean up response to ensure only valid output
#     emotion_text = emotion.text.strip().lower()
#     if emotion_text in ["sad", "happy", "angry", "neutral"]:
#         return emotion_text
#     return "neutral"

# def classify_content(text: str) -> str:
#     """
#     Uses the LLM to classify content into safety categories: suicidal, explicit_18_plus, or safe.
#     """
#     prompt = (
#         "Classify the following text for safety. "
#         "Reply with one word: 'suicidal' if the text contains suicidal/self-harm intent, "
#         "'explicit_18_plus' if the text contains explicit sexual content, or 'safe' if neither applies.\n\nText: " + text
#     )
#     result = client.models.generate_content(
#         model="gemini-3-flash-preview", contents=prompt
#     )
#     label = result.text.strip().lower()
#     if label in ["suicidal", "explicit_18_plus", "safe"]:
#         return label
#     return "safe"
  


def detect_language(user_text: str) -> str:
    """
    Detects the language of the user input using Gemini.
    Returns ISO-style language name (e.g. English, Hindi, Spanish).
    """
    prompt = (
        "Detect the language of the following text. "
        "Reply with ONLY the language name in English (e.g. English, Hindi, Spanish, Marathi).\n\n"
        f"Text: {user_text}"
    )

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


# ========== FINAL REPLY PIPELINE ==========

def reply(user_text: str, user_id: str, role: str, stream: bool = False):

    # 1) DETECT USER LANGUAGE
    language = detect_language(user_text)

    # 2) BUILD MEMORY CONTEXT
    memory_context = build_memory_context(user_id, user_text)

    # 3) CONSTRUCT PROMPT
    messages = [
        {"role": "system", "content": system_prompt(language)},
        {"role": "system", "content": role_prompt(role)},
    ]


    if memory_context:
        messages.append({"role": "system", "content": memory_context})

    messages.append({"role": "user", "content": user_text})

    # 3) SINGLE LLM CALL
    raw = call_llm(messages, stream=False)

    try:
        data = safe_json_parse(raw)
    except Exception:
        data = {
            "content_type": "safe",
            "reply_text": raw,
            "emotion": "neutral",
            "intensity": 0.5,
            "gesture": "none"
        }

    # Store memory only for safe content
    if data.get("content_type") == "safe":
        store_message(user_id, user_text)

    return {
        "text": data.get("reply_text", ""),
        "emotion": data.get("emotion", "neutral"),
        "confidence": 0.9,
        "gesture": data.get("gesture", "none"),
        "intensity": float(data.get("intensity", 0.5)),
        "sources": []
    }
    

    # 5) STORE MESSAGE
    store_message(user_id, user_text)

    # 6) RETURN
    return {
        "text": data.get("reply_text", ""),
        "emotion": data.get("emotion", "neutral"),
        "confidence": 0.9,
        "gesture": data.get("gesture", "none"),
        "intensity": float(data.get("intensity", 0.5)),
        "sources": []
    }

# ========== TERMINAL / CLI CHAT LOOP ==========

def start_cli_chat(user_id="terminal_user", role="assistant"):
    """
    Interactive terminal-style chat loop.
    Type 'exit' to stop.
    """
    print("\n🧠 MIRAGE AI (Terminal Mode)")
    print("Type your message and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_text = input("You: ")

            if user_text.lower() in ["exit", "quit"]:
                print("👋 Exiting MIRAGE. Goodbye!")
                break

            response = reply(
                user_text=user_text,
                user_id=user_id,
                role=role
            )

            print("\nMIRAGE:", response["text"])
            print("Emotion:", response["emotion"])
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n👋 Chat interrupted. Exiting.")
            break

# start_cli_chat()

class LLMInput(BaseModel):
    user_text: str
    user_id: str
    role: str

@router.post("/llm-response")
def llm_response(input: LLMInput):
    return reply(input.user_text, input.user_id, input.role, stream=False)

