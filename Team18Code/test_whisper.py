import pyaudio
import wave
import whisper
import time
import os
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def record_audio(seconds=5, filename="test.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    p = pyaudio.PyAudio()
    
    print("=== Recording will start in 3 seconds ===")
    time.sleep(1)
    print("=== 2 seconds ===")
    time.sleep(1)
    print("=== 1 second ===")
    time.sleep(1)
    print("=== Start speaking... ===")
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("=== Recording finished! ===")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(filename, model_size="base"):
    print("=== Transcribing... ===")
    
    download_dir = os.path.join(os.getcwd(), "whisper_models")
    os.makedirs(download_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = whisper.load_model(model_size, download_root=download_dir, device=device)
    
    options = {
        "language": "en", 
        "task": "transcribe", 
        "fp16": False if device == "cpu" else True,
        "initial_prompt": "The following is an English conversation.", 
    }
    
    result = model.transcribe(filename, **options)
    return result["text"]

def main():
    try:
        RECORD_SECONDS = 10
        MODEL_SIZE = "small" 
        

        record_audio(RECORD_SECONDS, "test.wav")
        
        print("\n=== Model is processing (this might take a moment)... ===")
        transcription = transcribe_audio("test.wav", MODEL_SIZE)
        
        print("\n=== Transcription Result: ===")
        print(transcription)

        with open("transcription.txt", "w", encoding="utf-8") as f:
            f.write(transcription)
        print("\nTranscription saved to 'transcription.txt'")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()