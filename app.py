import os
import re
import shutil
import tempfile
import traceback
import logging
import subprocess
import io

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import whisper
from gtts import gTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scheduler")

FFMPEG_BIN = r"C:\Users\Aditya\Downloads\ffmpeg-7.1.1-essentials_build\bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

GOOGLE_CRED_FILE = r"C:\Users\Aditya\Downloads\practical-argon-463304-h6-afc503b10302.json"
CHATGROQ_KEY     = "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"
GMAIL_CALENDAR   = "adityaiyer495@gmail.com"
SCOPES           = ["https://www.googleapis.com/auth/calendar"]

if not os.path.isfile(GOOGLE_CRED_FILE):
    raise FileNotFoundError("Service-account JSON not found")

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

def parse_ist(dt_str: str) -> datetime:
    dt = datetime.fromisoformat(dt_str)
    return dt if dt.tzinfo is None else dt.astimezone(IST).replace(tzinfo=None)

def to_utc_iso(dt: datetime) -> str:
    return dt.replace(tzinfo=IST).astimezone(UTC).isoformat()

CURRENT_YEAR = datetime.now(IST).year
SYSTEM_PROMPT = (
    "You are an intelligent calendar assistant. Use exactly:\n"
    "[FIND_SLOTS start=YYYY-MM-DDTHH:MM end=YYYY-MM-DDTHH:MM duration=M]\n"
    "[BOOK_EVENT datetime=YYYY-MM-DDTHH:MM duration=M title=\"...\"]\n"
    "Ask follow-ups if missing details, understand natural times in IST.\n"
    f"If user omits year assume {CURRENT_YEAR}. Suggest 3 alternatives on conflict.\n"
    "Be clear and concise."
)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=CHATGROQ_KEY
)

stt_model = whisper.load_model("base")
if not shutil.which("ffmpeg"):
    logger.warning("ffmpeg not found on PATH — audio may fail")

def get_service():
    creds = service_account.Credentials.from_service_account_file(GOOGLE_CRED_FILE, scopes=SCOPES)
    return build("calendar", "v3", credentials=creds)

def ensure_subscribed(svc):
    try:
        svc.calendarList().get(calendarId=GMAIL_CALENDAR).execute()
    except HttpError as e:
        if e.resp.status == 404:
            svc.calendarList().insert(body={"id":GMAIL_CALENDAR}).execute()

def list_events(svc, start: datetime, end: datetime) -> List[Dict[str,Any]]:
    return svc.events().list(
        calendarId=GMAIL_CALENDAR,
        timeMin=to_utc_iso(start),
        timeMax=to_utc_iso(end),
        singleEvents=True,
        orderBy="startTime"
    ).execute().get("items", [])

def busy_blocks(svc, start: datetime, end: datetime) -> List[tuple]:
    blocks = []
    for e in list_events(svc, start, end):
        s = e["start"].get("dateTime", e["start"].get("date"))
        t = e["end"].get("dateTime",   e["end"].get("date"))
        blocks.append((parse_ist(s), parse_ist(t)))
    return sorted(blocks, key=lambda x: x[0])

def free_slots(svc, start: datetime, end: datetime, mins: int) -> List[str]:
    free, cur = [], start
    for bs, be in busy_blocks(svc, start, end):
        if (bs - cur).total_seconds() >= mins*60:
            free.append((cur, bs))
        cur = max(cur, be)
    if (end - cur).total_seconds() >= mins*60:
        free.append((cur, end))
    return [
        f"{s:%Y-%m-%d %I:%M %p} - {(s + timedelta(minutes=mins)):%I:%M %p}"
        for s,_ in free[:3]
    ]

def create_event(svc, start: datetime, mins: int, title: str) -> str:
    evt = {
        "summary": title,
        "start": {"dateTime": to_utc_iso(start), "timeZone":"UTC"},
        "end":   {"dateTime": to_utc_iso(start+timedelta(minutes=mins)), "timeZone":"UTC"}
    }
    return svc.events().insert(calendarId=GMAIL_CALENDAR, body=evt).execute().get("htmlLink")

SESSIONS: Dict[str,Any] = {}

def process_message(session_id: str, message: str) -> str:
    ctx = SESSIONS.setdefault(session_id, {"messages":[{"role":"system","content":SYSTEM_PROMPT}]})
    ctx["messages"].append({"role":"user","content":message})

    conv = []
    for m in ctx["messages"]:
        conv.append({
            "system":    SystemMessage,
            "user":      HumanMessage,
            "assistant": AIMessage
        }[m["role"]](content=m["content"]))

    reply = llm.invoke(conv).content
    ctx["messages"].append({"role":"assistant","content":reply})

    svc = get_service(); ensure_subscribed(svc)

    if (m := re.search(r"\[FIND_SLOTS start=(\S+) end=(\S+) duration=(\d+)\]", reply)):
        s,e,d = m.groups()
        slots = free_slots(svc, parse_ist(s), parse_ist(e), int(d))
        text = "Here are available slots:\n" + "\n".join(slots)
        ctx["messages"].append({"role":"assistant","content":text})
        return text

    if (m := re.search(r"\[BOOK_EVENT datetime=(.+?) duration=(\d+) title=\"(.+?)\"\]", reply)):
        dt_str,d,title = m.groups()
        dt = parse_ist(dt_str)
        if busy_blocks(svc, dt, dt+timedelta(minutes=int(d))):
            alts = free_slots(svc, dt.replace(hour=0,minute=0), dt.replace(hour=23,minute=59), int(d))
            text = "Sorry, that slot is busy. Alternatives:\n" + "\n".join(alts)
        else:
            link = create_event(svc, dt, int(d), title)
            text = f"Booked '{title}' on {dt:%Y-%m-%d %I:%M %p}. Link: {link}"
        ctx["messages"].append({"role":"assistant","content":text})
        return text

    return reply

app = FastAPI()

class ChatReq(BaseModel):
    session_id: str
    message: str

class ResetReq(BaseModel):
    session_id: str

@app.post("/chat")
async def chat_endpoint(req: ChatReq):
    try:
        return {"reply": process_message(req.session_id, req.message)}
    except:
        logger.error(traceback.format_exc())
        raise HTTPException(500, "Internal server error")

@app.post("/audio-chat")
async def audio_chat(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Uploaded file must be audio/*")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".m4a")
    try:
        tmp.write(await file.read())
        tmp.close()

        try:
            transcript = stt_model.transcribe(tmp.name)["text"].strip()
        except:
            wav = tmp.name + ".wav"
            subprocess.run(
                ["ffmpeg","-y","-i",tmp.name,"-ar","16000","-ac","1",wav],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            txt = stt_model.transcribe(wav)["text"]
            transcript = txt.strip() if isinstance(txt, str) else txt.decode("utf-8","ignore").strip()
            os.remove(wav)

        logger.info("Transcript: %s", transcript)

        reply = process_message(session_id, transcript)

        buf = io.BytesIO()
        tts = gTTS(reply, lang="en", slow=False)
        tts.write_to_fp(buf)
        audio_bytes = buf.getvalue()

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Length": str(len(audio_bytes))}
        )

    except:
        logger.error(traceback.format_exc())
        raise HTTPException(500, "Audio transcription failed")
    finally:
        os.remove(tmp.name)

@app.post("/reset")
async def reset(req: ResetReq):
    SESSIONS.pop(req.session_id, None)
    return {"status": "reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True)
