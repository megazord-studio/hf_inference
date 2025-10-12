#!/usr/bin/env python3
"""
Simple script to start the FastAPI server.
Usage: python start_server.py [--port 8000] [--host 0.0.0.0]
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Start the HF Inference FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    
    import uvicorn
    from app.main import app
    
    print(f"Starting HF Inference API on http://{args.host}:{args.port}")
    print(f"  - Health check: http://{args.host}:{args.port}/healthz")
    print(f"  - API docs: http://{args.host}:{args.port}/docs")
    print(f"  - Inference endpoint: http://{args.host}:{args.port}/inference")
    
    uvicorn.run(
        "app.main:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
