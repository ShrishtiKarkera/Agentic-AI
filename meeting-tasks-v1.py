from __future__ import print_function
import sounddevice as sd
import numpy as np
import whisper
import re
import queue
import threading
import time
import json
# Recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024
DURATION = 0  # 0 means unlimited until stopped

import datetime
import os.path
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# If modifying these SCOPES, delete the token.json file first.
SCOPES = ['https://www.googleapis.com/auth/calendar']

q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())

def record_audio(filename="meeting.wav"):
    print("🔴 Recording... Press Ctrl+C to stop.")
    with sf.SoundFile(filename, mode='x', samplerate=SAMPLE_RATE,
                      channels=CHANNELS, subtype='PCM_16') as file:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            callback=audio_callback, blocksize=BLOCK_SIZE):
            while True:
                file.write(q.get())

def transcribe_audio(filename="meeting.wav"):
    print("📝 Transcribing...")
    model = whisper.load_model("base")  # "small" or "medium" for better accuracy
    result = model.transcribe(filename)
    return result["text"]

def extract_tasks(text):
    # Look for sentences with "you need to", "please", "can you", "assign", etc.
    task_patterns = r"(?:you need to|please|can you|assign(ed)? to you|your task is)\s+([^\.!?]+)"
    tasks = re.findall(task_patterns, text, re.IGNORECASE)
    return [t[1].strip() for t in tasks]

def get_calendar_service():
    creds = None
    if os.path.exists('token.json'):
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('calendar', 'v3', credentials=creds)


def add_task_to_calendar(task, start_time=None, duration_minutes=30):
    service = get_calendar_service()

    if not start_time:
        # Default: schedule for today at next hour
        now = datetime.datetime.now()
        start_time = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)

    event = {
        'summary': f"Task: {task}",
        'description': f"Auto-added from meeting transcript",
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'America/New_York',  # change if needed
        },
        'end': {
            'dateTime': (start_time + datetime.timedelta(minutes=duration_minutes)).isoformat(),
            'timeZone': 'America/New_York',
        },
    }

    event = service.events().insert(calendarId='primary', body=event).execute()
    print(f"✅ Task added to calendar: {event.get('htmlLink')}")




if __name__ == "__main__":
    import soundfile as sf

   try:
       record_audio("meeting.wav")
   except KeyboardInterrupt:
       print("\n🛑 Recording stopped.")

  # This part was for the non LLM regex matcher
   transcript = transcribe_audio("meeting.wav")
   print("\nFull Transcript:\n", transcript)

   tasks = extract_tasks(transcript)

    if tasks:
        print("\n✅ Tasks Assigned to You:")
        for i, t in enumerate(tasks, 1):
            print(f"{i}. {t}")

        for t in tasks:
            add_task_to_calendar(t)
    else:
        print("\nNo explicit tasks found.")


