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

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpeg is installed and working!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg is NOT installed or not working correctly.")
        return False
    return True

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("medium", device=device)
        
        result = model.transcribe(audio_path, language="en", fp16=torch.cuda.is_available())
        return result["text"], None
    except Exception as e:
        return None, str(e)

def extract_info(text):
    data = {"name": None, "location": None}
    
    name_patterns = [
        r"my name is ([A-Za-z\s]+)",
        r"myself ([A-Za-z\s]+)",
        r"i am ([A-Za-z\s]+)",
        r"this is me ([A-Za-z\s]+)",  # Added for cases like "Hi, this is me Payal"
        r"i'm ([A-Za-z\s]+)"
    ]
    
    location_patterns = [
        r"i'm from ([A-Za-z\s]+)",
        r"i live in ([A-Za-z\s]+)",
        r"i am from ([A-Za-z\s]+)",
        r"then i moved to ([A-Za-z\s]+)"  # Added for "Then I moved to India"
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


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400
    
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
        print("Received Data:", data)  # Debugging step

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        extracted_info = data.get("extracted_info", {})
        transcription = data.get("transcription", "")

        # Properly structure the data for Excel
        df = pd.DataFrame([{
            "Location": extracted_info.get("location", "N/A"),
            "Name": extracted_info.get("name", "N/A"),
            "Transcription": transcription
        }])

        excel_path = "transcription_data.xlsx"
        df.to_excel(excel_path, index=False)

        print("Excel file created successfully at:", excel_path)  # Debugging step
        return send_file(excel_path, as_attachment=True, download_name="transcription_data.xlsx")

    except Exception as e:
        print("Error:", str(e))  # Show actual error
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
