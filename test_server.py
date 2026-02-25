#!/usr/bin/env python3
"""
Quick test script to verify the server is working
"""

import requests
import os
import time
from pathlib import Path

def test_server():
    """Test if the server is working properly"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Skin Lesion Classification Server")
    print("=" * 50)
    
    # Test 1: Root endpoint
    try:
        print("1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data.get('status')}")
            print(f"   ✅ HF Status: {data.get('hf_status')}")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Health check
    try:
        print("\n2. Testing health check...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data.get('status')}")
            print(f"   ✅ Service: {data.get('prediction_service')}")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: System info
    try:
        print("\n3. Testing system info...")
        response = requests.get(f"{base_url}/system/info")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Service: {data.get('prediction_service')}")
            print(f"   ✅ HF Status: {data.get('hf_status')}")
            print(f"   ✅ Directories: {data.get('directories')}")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Web app
    try:
        print("\n4. Testing web app...")
        response = requests.get(f"{base_url}/app")
        if response.status_code == 200:
            print(f"   ✅ Web app accessible")
            print(f"   ✅ Response length: {len(response.text)} chars")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: API docs
    try:
        print("\n5. Testing API docs...")
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print(f"   ✅ API docs accessible")
        else:
            print(f"   ❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test Results:")
    print(f"   Server URL: {base_url}")
    print(f"   Web App: {base_url}/app")
    print(f"   API Docs: {base_url}/docs")
    print("\n💡 If all tests pass, your server is working correctly!")

def check_environment():
    """Check environment configuration"""
    print("\n🔍 Checking Environment Configuration")
    print("=" * 50)
    
    # Check .env file
    if Path(".env").exists():
        print("✅ .env file found")
    else:
        print("❌ .env file not found")
    
    # Check important environment variables
    important_vars = [
        "HUGGINGFACE_API_TOKEN",
        "SECRET_KEY",
        "DATABASE_URL",
        "HOST",
        "PORT"
    ]
    
    for var in important_vars:
        value = os.getenv(var)
        if value:
            if var == "HUGGINGFACE_API_TOKEN":
                print(f"✅ {var}: {'*' * 20}...{value[-10:]}")  # Hide most of the token
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    # Check directories
    dirs = ["uploads", "templates", "static", "models", "logs"]
    print(f"\n📁 Directory Status:")
    for directory in dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (missing)")

def main():
    """Main function"""
    print("🩺 Skin Lesion Classification - Server Test")
    print("=" * 60)
    
    # Check environment first
    check_environment()
    
    # Wait a moment
    print("\n⏳ Waiting 3 seconds before testing server...")
    time.sleep(3)
    
    # Test server
    test_server()
    
    print("\n🎉 Testing complete!")
    print("If you see errors, check the server logs and your .env file.")

if __name__ == "__main__":
    main()