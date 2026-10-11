"""Lightweight diagnostic HTTP server serving XMCA studio and /api/diagnose."""

import json
import logging
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostic_engine import XMCADiagnosticEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("XMCAServer")

WEB_DIR = Path(__file__).resolve().parent


class XMCARequestHandler(SimpleHTTPRequestHandler):
    engine = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_POST(self):
        if self.path == "/api/diagnose":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                question = data.get("question", "")
                answer = data.get("answer", "")
                evidence = data.get("evidence", [])
                image = data.get("image", "dummy.jpg")
                threshold = float(data.get("threshold", 0.5))

                preset_scores = data.get("preset_scores", None)

                if self.__class__.engine is None:
                    self.__class__.engine = XMCADiagnosticEngine(threshold=threshold)

                self.__class__.engine.threshold = threshold
                report = self.__class__.engine.diagnose(
                    image=image,
                    question=question,
                    answer=answer,
                    retrieved_passages=evidence,
                    override_scores=preset_scores
                )

                response_bytes = json.dumps(report.to_dict()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as e:
                logger.error(f"Diagnostic error: {e}", exc_info=True)
                err_bytes = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8080):
    server_address = ("127.0.0.1", port)
    logger.info(f"Starting XMCA-Net Studio server on http://localhost:{port}")
    httpd = HTTPServer(server_address, XMCARequestHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
