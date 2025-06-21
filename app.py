

import os
import re
import shutil
import tempfile
import traceback
import logging
import subprocess

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import whisper  


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scheduler")


FFMPEG_BIN = r"C:\Users\Aditya\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")


GOOGLE_CRED_FILE = r"C:\Users\Aditya\Downloads\practical-argon-463304-h6-afc503b10302.json"
CHATGROQ_KEY     = "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"
GMAIL_CALENDAR   = "adityaiyer495@gmail.com"
SCOPES           = ["https://www.googleapis.com/auth/calendar"]

if not os.path.isfile(GOOGLE_CRED_FILE):
    raise FileNotFoundError("Service‑account JSON not found")


IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

def parse_ist(dt_str: str) -> datetime:
    """Parse an ISO string (with or without offset), convert to IST naive."""
    dt = datetime.fromisoformat(dt_str)
    if dt.tzinfo is None:
        return dt  
    return dt.astimezone(IST).replace(tzinfo=None)

def to_utc_iso(dt: datetime) -> str:
    """Convert naive-IST dt into UTC ISO."""
    local = dt.replace(tzinfo=IST)
    return local.astimezone(UTC).isoformat()


CURRENT_YEAR = datetime.now(IST).year
SYSTEM_PROMPT = (
    "You are an intelligent calendar assistant. You must parse user requests to schedule meetings and use the following commands exactly when you have all details:\n"
    "  [FIND_SLOTS start=YYYY-MM-DDTHH:MM end=YYYY-MM-DDTHH:MM duration=M]\n"
    "  [BOOK_EVENT datetime=YYYY-MM-DDTHH:MM duration=M title=\"...\"]\n"
    "If any detail is missing (date, time, duration, title), ask a concise follow-up question.\n"
    "You must understand natural language dates like 'tomorrow', 'next week', 'Monday morning at 9:30am', '10:30pm', etc., and convert them into ISO datetimes in IST (Asia/Kolkata).\n"
    f"**If the user omits the year, assume it is {CURRENT_YEAR}.**\n"
    "If the requested slot conflicts with an existing event, suggest up to 3 alternatives.\n"
    "Be clear and concise."
)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=CHATGROQ_KEY)


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
        si = parse_ist(s)
        ti = parse_ist(t)
        blocks.append((si, ti))
    return sorted(blocks, key=lambda x: x[0])

def free_slots(svc, start: datetime, end: datetime, mins: int) -> List[str]:
    free, cur = [], start
    for bs, be in busy_blocks(svc, start, end):
        if (bs - cur).total_seconds() >= mins*60:
            free.append((cur, bs))
        cur = max(cur, be)
    if (end - cur).total_seconds() >= mins*60:
        free.append((cur, end))
    return [f"{s:%Y-%m-%d %I:%M %p} - {(s+timedelta(minutes=mins)):%I:%M %p}" for s,_ in free[:3]]

def create_event(svc, start: datetime, mins: int, title: str) -> str:
    evt = {
        "summary": title,
        "start": {"dateTime": to_utc_iso(start), "timeZone": "UTC"},
        "end":   {"dateTime": to_utc_iso(start+timedelta(minutes=mins)), "timeZone": "UTC"}
    }
    return svc.events().insert(calendarId=GMAIL_CALENDAR, body=evt).execute().get("htmlLink")


SESSIONS: Dict[str, Any] = {}

def process_message(session_id: str, message: str) -> str:
    ctx = SESSIONS.setdefault(session_id, {"messages":[{"role":"system","content":SYSTEM_PROMPT}]})
    ctx["messages"].append({"role":"user","content":message})

    # LLM call
    conv = []
    for m in ctx["messages"]:
        conv.append({
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage
        }[m["role"]](content=m["content"]))
    reply = llm.invoke(conv).content
    ctx["messages"].append({"role":"assistant","content":reply})

    svc = get_service()
    ensure_subscribed(svc)

    
    m = re.search(r"\[FIND_SLOTS start=(\S+) end=(\S+) duration=(\d+)\]", reply)
    if m:
        s,e,d = m.groups()
        start,end = parse_ist(s), parse_ist(e)
        slots = free_slots(svc, start, end, int(d))
        text = "Here are available slots:\n" + "\n".join(slots)
        ctx["messages"].append({"role":"assistant","content":text})
        return text

    
    m = re.search(r"\[BOOK_EVENT datetime=(.+?) duration=(\d+) title=\"(.+?)\"\]", reply)
    if m:
        dt_str,d,title = m.groups()
        dt = parse_ist(dt_str)
        if busy_blocks(svc, dt, dt+timedelta(minutes=int(d))):
            alt = free_slots(svc, dt.replace(hour=0,minute=0), dt.replace(hour=23,minute=59), int(d))
            text = "Sorry, that slot is busy. Alternatives:\n" + "\n".join(alt)
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
    except Exception:
        logger.error(traceback.format_exc())
        raise HTTPException(500, "Internal server error")

@app.post("/audio-chat")
async def audio_chat(session_id: str = Form(...), file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Uploaded file must be audio/*")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".m4a")
    try:
        tmp.write(await file.read()); tmp.close()
        try:
            transcript = stt_model.transcribe(tmp.name)["text"].strip()
        except:
            wav = tmp.name + ".wav"
            subprocess.run(
                ["ffmpeg","-y","-i",tmp.name,"-ar","16000","-ac","1",wav],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            txt = stt_model.transcribe(wav)["text"]
            transcript = txt.strip() if isinstance(txt,str) else txt.decode("utf-8","ignore").strip()
            os.remove(wav)
        logger.info("Transcript: %s", transcript)
        return {"reply": process_message(session_id, transcript)}
    except Exception:
        logger.error(traceback.format_exc())
        raise HTTPException(500, "Audio transcription failed")
    finally:
        os.remove(tmp.name)

@app.post("/reset")
async def reset(req: ResetReq):
    SESSIONS.pop(req.session_id, None)
    return {"status":"reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True)
