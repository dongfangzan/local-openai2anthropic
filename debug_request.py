#!/usr/bin/env python3
"""Debug script to capture requests from Claude Code."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class DebugHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        print(f"\n{'='*60}")
        print(f"Path: {self.path}")
        print(f"Headers: {dict(self.headers)}")
        print(f"Body:\n{json.dumps(json.loads(body), indent=2, ensure_ascii=False)}")
        
        # Return fake response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Debug response"}],
            "model": "test",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }).encode())
    
    def log_message(self, format, *args):
        pass  # Suppress logs

if __name__ == "__main__":
    server = HTTPServer(('0.0.0.0', 8081), DebugHandler)
    print("Debug server running on http://localhost:8081")
    server.serve_forever()
