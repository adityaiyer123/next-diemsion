Smart Scheduler Agent

A FastAPI-based scheduling assistant that uses:

Groq-powered LLM (via LangChain) for conversational understanding

OpenAI Whisper for audio transcription (STT)

Google Calendar API for finding free slots and booking events

It supports both text (/chat) and audio (/audio-chat) endpoints, maintains multi-turn context, handles time zones (IST ↔ UTC), and resolves conflicts with suggested alternatives.

Features

Natural Language SchedulingParse user requests like:

"Schedule a 30-minute meeting tomorrow at 9:30am"

Multi-Turn ContextKeeps conversation history per session_id, asks for missing details (date, time, duration, title).

Audio Interface/audio-chat accepts audio uploads, transcribes via Whisper (with FFmpeg fallback), then processes the text.

Google Calendar Integration

[FIND_SLOTS start=… end=… duration=m] to list free time windows

[BOOK_EVENT datetime=… duration=m title="…"] to create calendar events

Automatic conflict detection and up to 3 alternative suggestions

Timezone Management

Inputs assumed in IST (Asia/Kolkata)

Converted to UTC for API calls, converted back for display

Year AssumptionIf the user omits the year, the assistant assumes the current year (2025).

Prerequisites

Python 3.9+

Google Cloud service account JSON (with Calendar API enabled)

FFmpeg installed, or update FFMPEG_BIN to your local path

Installation

Clone the repository

git clone <repo-url>
cd <repo-directory>

Create and activate a virtual environment

python -m venv venv
source venv/bin/activate      # Unix/macOS
venv\\Scripts\\activate     # Windows

Install dependencies

pip install fastapi uvicorn google-api-python-client google-auth whisper langchain-groq

Configuration

Edit the top of app.py to set:

GOOGLE_CRED_FILE: path to your Google service account JSON

CHATGROQ_KEY: your Groq API key

GMAIL_CALENDAR: your Google Calendar ID (e.g. primary email)

FFMPEG_BIN: path to FFmpeg if not on your system PATH

Usage

Start the server:

uvicorn app:app --reload --port 9000

Text Scheduling (/chat)

Send a POST request:

curl -X POST http://localhost:9000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user123","message":"Schedule a meeting tomorrow at 9am for 30 minutes"}'

Response:

{ "reply": "Sure—what’s the meeting title?" }

Continue the conversation by sending the title, and the assistant will book the slot.

Audio Scheduling (/audio-chat)

Upload an audio file:

curl -X POST http://localhost:9000/audio-chat \
  -F session_id=user123 \
  -F file=@meeting_request.m4a

The assistant will reply with transcription-based scheduling.

Reset Session (/reset)

Clears conversation history:

curl -X POST http://localhost:9000/reset \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user123"}'

Code Walkthrough

process_message: core function that appends user input, calls LLM, parses special [FIND_SLOTS] / [BOOK_EVENT] commands, invokes Calendar API, and returns a user-readable reply.

Conversation Memory: stored in SESSIONS[session_id]["messages"], replayed to the LLM on each turn.

Error Handling: logs traceback, returns HTTP 500 on unexpected errors.

Troubleshooting

FFmpeg errors: verify FFMPEG_BIN path or install globally.

400 Bad Request from Calendar API: ensure your service account has sharing rights to the target calendar.

LLM mistakes: you can refine SYSTEM_PROMPT or lower temperature for determinism.# new-dimension1
