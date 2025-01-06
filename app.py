from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
import speech_recognition as sr
import os, tempfile,  pydub, logging, shutil, wave


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cleanup_file(path: str):
    """Cleanup temporary files"""
    try:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    except Exception as e:
        logger.error(f"Error cleaning up file {path}: {e}")

def validate_audio_file(filename: str) -> bool:
    """Validate audio file extension"""
    allowed_extensions = {'.wav', '.m4a', '.mp3', '.ogg'}
    file_extension = os.path.splitext(filename.lower())[1]
    return file_extension in allowed_extensions

def is_valid_wav(file_path: str) -> bool:
    """Check if WAV file is valid"""
    try:
        with wave.open(file_path, 'rb') as wave_file:
            return wave_file.getnchannels() > 0
    except Exception:
        return False

def convert_to_wav(input_path: str, input_format: str) -> str:
    """Convert audio file to WAV format"""
    logger.info(f"Converting {input_format} file to WAV: {input_path}")
    
    temp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(temp_dir, 'converted.wav')
    
    try:
        # Load audio using pydub
        if input_format == 'm4a':
            audio = pydub.AudioSegment.from_file(input_path, format='m4a')
        elif input_format == 'mp3':
            audio = pydub.AudioSegment.from_mp3(input_path)
        elif input_format == 'ogg':
            audio = pydub.AudioSegment.from_ogg(input_path)
        elif input_format == 'wav':
            audio = pydub.AudioSegment.from_wav(input_path)
        else:
            audio = pydub.AudioSegment.from_file(input_path)

        # Convert to WAV with specific parameters
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
        audio = audio.set_sample_width(2)  # Set sample width to 16-bit
        
        # Export to WAV
        audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        
        if not is_valid_wav(wav_path):
            raise ValueError("Converted WAV file is invalid")
            
        return wav_path
        
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        logger.error(f"Error converting audio: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")

@app.post("/speech-to-text/")
async def speech_to_text(audio_file: UploadFile = File(...)):
    """Convert speech to text"""
    logger.info(f"Received file: {audio_file.filename}")
    
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not validate_audio_file(audio_file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Supported formats: WAV, M4A, MP3, OGG"
        )

    temp_dir = None
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded file
        temp_input_path = os.path.join(temp_dir, 'input' + os.path.splitext(audio_file.filename)[1])
        content = await audio_file.read()
        with open(temp_input_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Saved uploaded file to: {temp_input_path}")
        
        # Always convert to WAV to ensure consistent format
        file_extension = os.path.splitext(audio_file.filename)[1].lower()[1:]  # Remove the dot
        temp_wav_path = convert_to_wav(temp_input_path, file_extension)

        # Initialize recognizer with adjusted settings
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        
        # Perform recognition
        with sr.AudioFile(temp_wav_path) as source:
            logger.info("Recording audio from file")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
            
        logger.info("Performing speech recognition")
        text = recognizer.recognize_google(audio, language="en-US")
        
        return {"text": text, "status": "success"}

    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Could not understand audio. Please ensure the audio contains clear speech.")
    except sr.RequestError as e:
        raise HTTPException(status_code=503, detail="Speech recognition service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.post("/text-to-speech/")
async def text_to_speech(background_tasks: BackgroundTasks, text: str = Form(...)):
    """Convert text to speech"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()
        
        # Generate the audio file
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(temp_file.name)
        
        # Verify file exists and has content
        if not os.path.exists(temp_file.name):
            raise HTTPException(status_code=500, detail="Failed to create audio file")
        
        if os.path.getsize(temp_file.name) == 0:
            raise HTTPException(status_code=500, detail="Generated audio file is empty")
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(cleanup_file, temp_file.name)
        
        return FileResponse(
            temp_file.name,
            media_type="audio/mpeg",
            filename="output.mp3",
            headers={"Content-Disposition": "attachment; filename=output.mp3"}
        )
    
    except Exception as e:
        # Clean up if error occurs
        if 'temp_file' in locals():
            cleanup_file(temp_file.name)
        logger.error(f"Error in text-to-speech conversion: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating speech from text: {str(e)}")

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
