import httpx
import asyncio
import time

BASE_URL = "http://localhost:8080"

async def manage_llms():
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. List available models
        print("--- Available Models ---")
        models_resp = await client.get(f"{BASE_URL}/api/models")
        models = models_resp.json()["models"]
        for m in models:
            print(f"- {m['name']}")

        if not models:
            print("No models found!")
            return

        # 2. Start an instance on port 8099
        print("\n--- Starting Instance ---")
        model = models[0]
        start_resp = await client.post(f"{BASE_URL}/api/start", json={
            "model_path": model["path"],
            "port": 8099,
            "gpu_id": 0
        })
        print(f"Launch Status: {start_resp.json()['status']}")

        # 3. Wait for Ready state
        print("Waiting for instance to be READY...")
        while True:
            status_resp = await client.get(f"{BASE_URL}/api/status")
            instances = status_resp.json()["instances"]
            inst = next((i for i in instances if i["port"] == 8099), None)
            
            if inst and inst["ready"]:
                print("Instance is READY!")
                break
            
            print(".", end="", flush=True)
            await asyncio.sleep(2)

        # 4. Stop the instance
        print("\n--- Stopping Instance ---")
        stop_resp = await client.post(f"{BASE_URL}/api/stop?port=8099")
        print(f"Stop Status: {stop_resp.json()['status']}")

if __name__ == "__main__":
    asyncio.run(manage_llms())
