"""
Test script for the VQA Chat Application
"""

import requests
import json
from PIL import Image
import io
import base64
import numpy as np


def create_test_image():
    """Create a simple test image"""
    # Create a simple colored image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"


def test_api_endpoints():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing VQA Chat Application...")
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the app is running on port 8000")
        return
    
    # Test 2: Load model
    try:
        response = requests.post(f"{base_url}/api/load-model")
        if response.status_code == 200:
            print("✅ Model loaded successfully")
        else:
            print(f"❌ Failed to load model: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test 3: Test prediction with sample image
    try:
        test_image = create_test_image()
        question = "What color is this image?"
        
        # Convert base64 to file-like object
        image_data = base64.b64decode(test_image.split(",")[1])
        
        files = {"image": ("test.png", image_data, "image/png")}
        data = {"question": question}
        
        response = requests.post(f"{base_url}/api/predict", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful: {result['answer']}")
        else:
            print(f"❌ Prediction failed: {response.text}")
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")


def test_websocket():
    """Test WebSocket functionality"""
    import websockets
    import asyncio
    
    async def test_ws():
        uri = "ws://localhost:8000/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                print("✅ WebSocket connected")
                
                # Test model loading
                await websocket.send(json.dumps({"type": "load_model"}))
                response = await websocket.recv()
                data = json.loads(response)
                
                if data["status"] == "loaded":
                    print("✅ Model loaded via WebSocket")
                    
                    # Test prediction
                    test_image = create_test_image()
                    await websocket.send(json.dumps({
                        "type": "predict",
                        "question": "What do you see in this image?",
                        "image": test_image
                    }))
                    
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if data["status"] == "success":
                        print(f"✅ WebSocket prediction successful: {data['answer']}")
                    else:
                        print(f"❌ WebSocket prediction failed: {data['message']}")
                else:
                    print(f"❌ WebSocket model loading failed: {data['message']}")
        
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
    
    # Run WebSocket test
    try:
        asyncio.run(test_ws())
    except Exception as e:
        print(f"❌ WebSocket test error: {e}")


if __name__ == "__main__":
    print("🚀 VQA Chat Application Test Suite")
    print("=" * 50)
    
    # Test API endpoints
    test_api_endpoints()
    
    print("\n" + "=" * 50)
    
    # Test WebSocket
    test_websocket()
    
    print("\n✅ Test suite completed!") 