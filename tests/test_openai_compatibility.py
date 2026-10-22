import os
import sys
import httpx
import base64
import time

BASE_URL = "http://localhost:8030"

async def main():
    print("=== Testing OpenAI Compatibility Gateway Endpoints ===")
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        # 1. Fetch available models
        print("\n1. Fetching available models...")
        models_resp = await client.get(f"{BASE_URL}/api/models")
        if models_resp.status_code != 200:
            print(f"FAILED: /api/models returned {models_resp.status_code}")
            sys.exit(1)
        
        models_data = models_resp.json()
        models = models_data.get("models", [])
        if not models:
            print("FAILED: No models available in the system.")
            sys.exit(1)
            
        test_model = models[0]["name"]
        print(f"Using test model: {test_model}")
        
        # 2. Test /v1/chat/completions
        print("\n2. Testing /v1/chat/completions (Text)...")
        chat_req = {
            "model": test_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Respond with the word 'SUCCESS' and nothing else."}
            ],
            "temperature": 0.0,
            "max_tokens": 16,
            "thinking": False
        }
        
        chat_resp = await client.post(f"{BASE_URL}/v1/chat/completions", json=chat_req)
        if chat_resp.status_code != 200:
            print(f"FAILED: /v1/chat/completions returned {chat_resp.status_code} - {chat_resp.text}")
            sys.exit(1)
            
        chat_data = chat_resp.json()
        print("Response structure:")
        print(chat_data)
        
        # Check standard fields
        assert chat_data["choices"][0]["message"]["content"] is not None
        assert "usage" in chat_data
        print(f"SUCCESS: Chat Completion returned content: '{chat_data['choices'][0]['message']['content']}'")
        
        # 3. Test /v1/audio/speech
        print("\n3. Testing /v1/audio/speech...")
        speech_req = {
            "model": "tts-1",
            "input": "This is a compatibility test.",
            "voice": "female, young adult, whisper"
        }
        
        speech_resp = await client.post(f"{BASE_URL}/v1/audio/speech", json=speech_req)
        if speech_resp.status_code != 200:
            print(f"FAILED: /v1/audio/speech returned {speech_resp.status_code} - {speech_resp.text}")
            sys.exit(1)
            
        audio_content = speech_resp.content
        print(f"SUCCESS: Speech generated {len(audio_content)} bytes of audio.")
        
        # 4. Test /v1/audio/transcriptions
        print("\n4. Testing /v1/audio/transcriptions...")
        # Save temp file
        temp_audio_path = "temp_test_speech.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_content)
            
        try:
            files = {"file": (temp_audio_path, open(temp_audio_path, "rb"), "audio/wav")}
            stt_resp = await client.post(
                f"{BASE_URL}/v1/audio/transcriptions",
                files=files,
                data={"model": "whisper-1"}
            )
            if stt_resp.status_code != 200:
                print(f"FAILED: /v1/audio/transcriptions returned {stt_resp.status_code} - {stt_resp.text}")
                sys.exit(1)
                
            stt_data = stt_resp.json()
            print("Transcription result:")
            print(stt_data)
            assert "text" in stt_data
            print(f"SUCCESS: Transcription text: '{stt_data['text']}'")
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
    print("\n=== All OpenAI Compatibility Gateway tests PASSED! ===")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
