"""Local-only transparent OBS browser source. Static whitelist, no media/credentials."""
import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import urllib.request
from urllib.parse import urlsplit
from labeler import ROOT


class OverlayHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        port=self.server.server_port
        if self.headers.get('Host') not in {f'127.0.0.1:{port}',f'localhost:{port}'}:
            self.send_error(403);return
        route=urlsplit(self.path).path
        if route=='/health':
            data=json.dumps({'app':'sound-captions-overlay','websocket_port':self.server.ws_port}).encode();kind='application/json'
        elif route=='/config.json':
            data=json.dumps({'websocket':f'ws://127.0.0.1:{self.server.ws_port}','display_duration_s':3,'max_captions':2}).encode();kind='application/json'
        elif route in {'/','/overlay','/overlay.css','/overlay.js'}:
            file={'/':'index.html','/overlay':'index.html','/overlay.css':'overlay.css','/overlay.js':'overlay.js'}[route]
            data=(ROOT/'overlay'/file).read_bytes();kind={'html':'text/html','css':'text/css','js':'text/javascript'}[file.split('.')[-1]]
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',f"default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self' ws://127.0.0.1:{self.server.ws_port}; base-uri 'none'; object-src 'none'")
        self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass


def make_server(port=8771,ws_port=8770):
    server=ThreadingHTTPServer(('127.0.0.1',port),OverlayHandler);server.ws_port=ws_port;return server


@contextmanager
def overlay_service(port=8771,ws_port=8770):
    try:server=make_server(port,ws_port)
    except OSError as error:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:value=json.load(response)
            if value!={'app':'sound-captions-overlay','websocket_port':ws_port}:raise ValueError('Different service')
        except Exception:raise RuntimeError(f'Port {port} is busy with a different application') from error
        yield;return
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8771);parser.add_argument('--ws-port',type=int,default=8770);args=parser.parse_args()
    print(f'OBS browser source: http://127.0.0.1:{args.port}/overlay (1920 x 1080)',flush=True)
    with make_server(args.port,args.ws_port) as server:
        try:server.serve_forever()
        except KeyboardInterrupt:pass

if __name__=='__main__':main()
