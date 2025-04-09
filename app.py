from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import torch
import whisper
import subprocess
import re
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip

app = Flask(__name__)
CORS(app)

# Global model variable
whisper_model = None

@app.before_first_request
def load_model():
    global whisper_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("medium", device=device)

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def extract_audio(video_path):
    if not check_ffmpeg():
        return None, "ffmpeg is not installed or not working properly"
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_path = temp_audio.name
        
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            return None, "No audio stream found in video"
        
        clip.audio.write_audiofile(audio_path, codec="pcm_s16le", fps=16000)
        clip.close()
        return audio_path, None
    except Exception as e:
        return None, str(e)

def transcribe_audio(audio_path):
    try:
        if whisper_model is None:
            raise Exception("Whisper model not loaded")
            
        result = whisper_model.transcribe(audio_path)
        return result["text"], None
    except Exception as e:
        return None, str(e)

def extract_info(text):
    data = {"name": None, "location": None}
    name_patterns = [
        r"my name is ([A-Za-z\s]+)",
        r"myself ([A-Za-z\s]+)",
        r"i am ([A-Za-z\s]+)",
        r"this is me ([A-Za-z\s]+)",
        r"i'm ([A-Za-z\s]+)"
    ]
    
    location_patterns = [
        r"i'm from ([A-Za-z\s]+)",
        r"i live in ([A-Za-z\s]+)",
        r"i am from ([A-Za-z\s]+)",
        r"then i moved to ([A-Za-z\s]+)"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["name"] = match.group(1).strip()
            break
    
    for pattern in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["location"] = match.group(1).strip()
            break
    
    return data

@app.route('/')
def home():
    return "Whisper Transcription API is running", 200

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "whisper_ready": whisper_model is not None
    }), 200

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files["video"]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        video_path = temp_video.name
        file.save(video_path)
    
    try:
        audio_path, audio_error = extract_audio(video_path)
        if audio_error:
            raise Exception(audio_error)
        
        transcription, transcribe_error = transcribe_audio(audio_path)
        if transcribe_error:
            raise Exception(transcribe_error)
        
        extracted_data = extract_info(transcription)
        
        return jsonify({
            "transcription": transcription,
            "extracted_info": extracted_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)

@app.route("/download_excel", methods=["POST"])
def download_excel():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        df = pd.DataFrame([{
            "Location": data.get("extracted_info", {}).get("location", "N/A"),
            "Name": data.get("extracted_info", {}).get("name", "N/A"),
            "Transcription": data.get("transcription", "")
        }])

        excel_path = "transcription_data.xlsx"
        df.to_excel(excel_path, index=False)
        return send_file(excel_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
