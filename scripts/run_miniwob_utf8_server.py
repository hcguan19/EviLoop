"""Serve the local MiniWoB++ tree with explicit UTF-8 response headers."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_ROOT = Path(os.getenv('MINIWOB_HTML_ROOT', 'datasets/repos/miniwob/miniwob/html'))


class ServerConfig(BaseModel):
	"""Validated MiniWoB server configuration."""

	host: str = '127.0.0.1'
	port: int = Field(default=8008, ge=1, le=65535)
	root: Path = DEFAULT_ROOT


class Utf8MiniWoBHandler(SimpleHTTPRequestHandler):
	"""Simple static handler that makes HTML/JS/CSS decoding deterministic."""

	extensions_map = {
		**SimpleHTTPRequestHandler.extensions_map,
		'.html': 'text/html; charset=utf-8',
		'.htm': 'text/html; charset=utf-8',
		'.js': 'text/javascript; charset=utf-8',
		'.css': 'text/css; charset=utf-8',
		'.json': 'application/json; charset=utf-8',
		'.svg': 'image/svg+xml; charset=utf-8',
	}

	def end_headers(self) -> None:
		self.send_header('Cache-Control', 'no-store')
		super().end_headers()


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--host', default='127.0.0.1')
	parser.add_argument('--port', type=int, default=8008)
	parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
	args = parser.parse_args()
	config = ServerConfig(host=args.host, port=args.port, root=args.root.resolve())
	if not config.root.is_dir():
		raise FileNotFoundError(f'MiniWoB root does not exist: {config.root}')

	handler = partial(Utf8MiniWoBHandler, directory=str(config.root))
	server = ThreadingHTTPServer((config.host, config.port), handler)
	print(f'Serving MiniWoB++ with UTF-8 headers at http://{config.host}:{config.port}/')
	print(f'Root: {config.root}')
	server.serve_forever()


if __name__ == '__main__':
	main()
