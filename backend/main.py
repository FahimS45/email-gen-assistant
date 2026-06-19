"""
main.py

FastAPI application exposing:

  WebSocket  /ws/generate          — streaming email generation
  POST       /generate             — non-streaming (useful for eval scripts)
  GET        /tones                — list of valid tone options
  GET        /health               — liveness probe
"""

import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import TONE_OPTIONS
from backend.models.schemas import EmailRequest, EmailResponse, ToneOptionsResponse
from backend.graph.graph import email_graph
from backend.graph.nodes import generate_email_stream, build_prompt, validate_input
from backend.graph.prompt_engineering import extract_cot_and_email

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Email Generation Assistant API",
    description="AI-powered professional email generator using LangGraph + advanced prompting.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check

@app.get("/health")
async def health():
    return {"status": "ok"}


# Tone options

@app.get("/tones", response_model=ToneOptionsResponse)
async def get_tones():
    return {"tones": TONE_OPTIONS}


# Non-streaming REST endpoint  (used by eval pipeline)

@app.post("/generate", response_model=EmailResponse)
async def generate_email_rest(request: EmailRequest):
    """
    Runs the full LangGraph pipeline and returns the completed email.
    Used by the evaluation scripts and for simple REST clients.
    """
    initial_state = {
        "intent":    request.intent,
        "key_facts": request.key_facts,
        "tone":      request.tone,
        "model":     request.model,
        "system_prompt": "",
        "user_prompt":   "",
        "cot_reasoning": "",
        "email_output":  "",
        "model_used":    "",
        "error":         None,
    }

    final_state = await email_graph.ainvoke(initial_state)

    if final_state.get("error"):
        raise HTTPException(status_code=422, detail=final_state["error"])

    return EmailResponse(
        email=final_state["email_output"],
        model_used=final_state["model_used"],
    )


# WebSocket streaming endpoint 

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket protocol:
      Client → JSON: { intent, key_facts, tone, model }
      Server → JSON stream:
          { type: "token",  content: "<token_string>" }   (many)
          { type: "cot",    content: "<reasoning_text>" } (once, after stream)
          { type: "email",  content: "<clean_email>" }    (once, final)
          { type: "done",   content: "<model_name>" }     (once, close signal)
          { type: "error",  content: "<message>" }        (on failure)

    The frontend should:
      1. Accumulate "token" messages to show a live typing effect.
      2. On "email", replace the accumulated buffer with the clean parsed email.
      3. On "done", close or re-enable the input form.
    """
    await websocket.accept()
    logger.info("WebSocket connection opened")

    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)

        # Build and validate state via the first two nodes
        initial_state = {
            "intent":    data.get("intent", ""),
            "key_facts": data.get("key_facts", []),
            "tone":      data.get("tone", "formal"),
            "model":     data.get("model", "openai"),
            "system_prompt": "",
            "user_prompt":   "",
            "cot_reasoning": "",
            "email_output":  "",
            "model_used":    "",
            "error":         None,
        }

        # Run validation node
        validation_update = await validate_input(initial_state)
        initial_state.update(validation_update)

        if initial_state.get("error"):
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": initial_state["error"],
            }))
            return

        # Build prompts
        prompt_update = await build_prompt(initial_state)
        initial_state.update(prompt_update)

        # Stream generation
        full_raw_output = ""
        model_name = ""

        async for token, m_name in generate_email_stream(initial_state):
            full_raw_output += token
            model_name = m_name
            await websocket.send_text(json.dumps({
                "type":    "token",
                "content": token,
            }))

        # Parse CoT and clean email
        cot, clean_email = extract_cot_and_email(full_raw_output)

        # Send structured results
        if cot:
            await websocket.send_text(json.dumps({
                "type":    "cot",
                "content": cot,
            }))

        await websocket.send_text(json.dumps({
            "type":    "email",
            "content": clean_email or full_raw_output,
        }))

        await websocket.send_text(json.dumps({
            "type":    "done",
            "content": model_name,
        }))

        logger.info(f"Email generated successfully via {model_name}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")

    except json.JSONDecodeError:
        await websocket.send_text(json.dumps({
            "type":    "error",
            "content": "Invalid JSON payload.",
        }))

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "type":    "error",
                "content": f"Server error: {str(e)}",
            }))
        except Exception:
            pass
