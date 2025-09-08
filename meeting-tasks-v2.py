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


import anthropic

def extract_tasks_with_claude(transcript, assignee="Shrishti Karkera"):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""
    You are a meeting assistant.
    Extract and list only the tasks explicitly assigned to {assignee} from the transcript below.
    Return them as a simple bullet list.

    Transcript:
    {transcript}
    """

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",  # free tier should support Sonnet or Haiku
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Claude responses are structured — extract text content
    tasks_text = response.content[0].text.strip()
    tasks = [line.strip("-• ").strip() for line in tasks_text.splitlines() if line.strip()]
    return tasks

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

# Speaker diarization with whisperx

import whisperx
from whisperx.diarize import DiarizationPipeline

def transcribe_with_diarization(audio_file, hf_token):
    device = "cpu"  # or "cuda" if you have GPU
    batch_size = 16

    # Step 1: Transcribe with WhisperX
    model = whisperx.load_model("base", device, compute_type="float32")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio)

    # Step 2: Load alignment model + metadata
    alignment_model, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )

    # Step 3: Align transcript with audio
    aligned_result = whisperx.align(
        result["segments"],
        alignment_model,
        metadata,
        audio,
        device=device
    )

    # Step 4: Diarization
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)

    # Merge diarization into aligned transcript
    aligned_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)

    return aligned_result

# Identify which diarized speaker is me

from pyannote.audio import Model
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import torch, torchaudio
import numpy as np

def get_embedding(audio_input, model_name="speechbrain/spkrec-ecapa-voxceleb"):
    embedding_model = PretrainedSpeakerEmbedding(model_name, device="cpu")

    # Case 1: audio_input is a file path
    if isinstance(audio_input, str):
        waveform, sample_rate = torchaudio.load(audio_input)

    # Case 2: audio_input is already a tensor
    elif isinstance(audio_input, torch.Tensor):
        waveform = audio_input
        # If you know the sample rate elsewhere, set it here.
        # Default to 16k (common for speech models).
        sample_rate = 16000

    else:
        raise TypeError(f"Unsupported input type: {type(audio_input)}")

    # Ensure waveform has shape (batch, channels, time)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)

    emb = embedding_model(waveform)

    # Convert to numpy if needed
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    elif not isinstance(emb, np.ndarray):
        raise TypeError(f"Unexpected embedding type: {type(emb)}")

    return emb[0]



def identify_your_speaker(reference_file, diarized_transcript, meeting_file):
    # Load reference embedding
    ref_emb = get_embedding(reference_file)

    # Get the sample rate of meeting audio
    _, sr = torchaudio.load(meeting_file)
    sr = sr  # sample_rate

    identified_transcript = []
    segments = diarized_transcript["segments"] if isinstance(diarized_transcript, dict) else diarized_transcript

    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        #    for segment in diarized_transcript:
        #        start = float(segment["start"])
        #        end = float(segment["end"])

        # Load only the relevant slice of the meeting audio
        waveform, _ = torchaudio.load(
            meeting_file,
            frame_offset=int(start * sr),
            num_frames=int((end - start) * sr)
        )

        # Ensure correct shape for embedding
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        seg_emb = get_embedding(waveform)

        # Cosine similarity
        #        similarity = np.dot(ref_emb, seg_emb[0]) / (np.linalg.norm(ref_emb) * np.linalg.norm(seg_emb[0]))
        similarity = np.dot(ref_emb, seg_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(seg_emb) + 1e-10)

        if similarity > 0.7:  # threshold can be tuned
            segment["speaker"] = "Shrishti"
        identified_transcript.append(segment)

    return identified_transcript


# Send it to Claude diarized and extract tasks
def extract_tasks_with_claude(transcript, your_speaker, assignee="Shrishti"):
    import json
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""
You are a meeting assistant. make sense of the tasks assigned to Shrishti and list them in the following JSON format:

{{
  "tasks": [
    "assigned task 1",
    "assigned task 2"
  ]
}}

Transcript:
{transcript}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text




if __name__ == "__main__":
    import soundfile as sf

   try:
       record_audio("meeting.wav")
   except KeyboardInterrupt:
       print("\n🛑 Recording stopped.")



    diarized_transcript = transcribe_with_diarization("meeting.wav", os.environ["HF_TOKEN"])
    print(diarized_transcript)
    identified_transcript = identify_your_speaker("shrishti_sample.wav", diarized_transcript, "meeting.wav")

    response = extract_tasks_with_claude(identified_transcript, "Shrishti", assignee="Shrishti")

    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        # Step 3: Parse JSON
        data = json.loads(json_str)
        tasks = data.get("tasks", [])
        print("Extracted tasks:", tasks)
    else:
        print("No tasks found")

    if tasks:
        print("\n✅ Tasks Assigned to You:")
        for i, t in enumerate(tasks, 1):
            print(f"{i}. {t}")

        for t in tasks:
            add_task_to_calendar(t)
    else:
        print("\nNo explicit tasks found.")


