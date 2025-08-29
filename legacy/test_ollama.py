#!/usr/bin/env python3
"""
Simple test script to verify Ollama client works with remote host
"""

import ollama
import os

def test_ollama_connection():
    print("Testing Ollama client connection...")
    
    # Test with your Ubuntu machine
    host = "http://192.168.0.87:11434"
    
    try:
        print(f"Connecting to {host}...")
        
        # Set environment variable for Ollama host
        os.environ['OLLAMA_HOST'] = host
        
        # Test basic connection
        response = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[
                {
                    'role': 'user',
                    'content': 'Hello! Please respond with "Connection successful!"'
                }
            ]
        )
        
        print("✅ Connection successful!")
        print(f"Response: {response['message']['content']}")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is running on 192.168.0.87:11434")
        print("2. Check if the qwen2.5vl:7b model is installed")
        print("3. Verify network connectivity between machines")
        print("4. Check firewall settings")

if __name__ == '__main__':
    test_ollama_connection()
