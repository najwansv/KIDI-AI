import http.server
import socketserver

# Specify the directory containing your web files
DIRECTORY = "Dashboard"
PORT = 80  # You can choose any port

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

# Start the server
with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"Serving HTTP on localhost:{PORT} (http://127.0.0.1:{PORT})")
    httpd.serve_forever()
