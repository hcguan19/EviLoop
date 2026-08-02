"""Run a lightweight local WebShop-compatible site for Browser Use evaluation.

This server uses the official WebShop human-instruction data but avoids the
heavy Lucene/Java indexing stack. It is intended as a stable local browser
automation benchmark target.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, redirect, request, url_for


@dataclass(frozen=True)
class ProductTask:
	index: int
	asin: str
	instruction: str
	attributes: tuple[str, ...]
	options: tuple[str, ...]


def _tokens(value: str) -> set[str]:
	return {token for token in re.findall(r'[a-z0-9]+', value.lower()) if len(token) > 2}


def load_tasks(path: Path, limit: int) -> list[ProductTask]:
	data: Any = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(data, dict):
		raise ValueError(f'Expected {path} to be a JSON object keyed by ASIN')

	tasks: list[ProductTask] = []
	for asin, items in data.items():
		if not isinstance(items, list):
			continue
		for item in items:
			if not isinstance(item, dict) or not item.get('instruction'):
				continue
			tasks.append(
				ProductTask(
					index=len(tasks),
					asin=str(item.get('asin') or asin),
					instruction=str(item['instruction']).strip(),
					attributes=tuple(str(attribute) for attribute in item.get('instruction_attributes') or []),
					options=tuple(str(option) for option in item.get('instruction_options') or []),
				)
			)
			if limit and len(tasks) >= limit:
				return tasks
	return tasks


def _page(title: str, body: str) -> str:
	return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 32px auto; color: #222; }}
    header {{ border-bottom: 1px solid #ddd; margin-bottom: 24px; padding-bottom: 12px; }}
    input[type="text"] {{ width: 70%; padding: 10px; font-size: 16px; }}
    button, a.button {{ padding: 10px 14px; margin: 4px; border: 1px solid #555; background: #f8f8f8; color: #111; text-decoration: none; cursor: pointer; }}
    .instruction {{ background: #fff9db; border: 1px solid #e5cf66; padding: 12px; margin: 12px 0; }}
    .product {{ border: 1px solid #ddd; padding: 14px; margin: 12px 0; }}
    .meta {{ color: #555; font-size: 14px; }}
    .done {{ background: #e9f8ee; border: 1px solid #8bc89a; padding: 16px; }}
  </style>
</head>
<body>
  <header><h1>WebShop Lite</h1><div class="meta">Stable local benchmark site</div></header>
  {body}
</body>
</html>"""


def create_app(tasks: list[ProductTask]) -> Flask:
	app = Flask(__name__)
	task_by_session: dict[str, ProductTask] = {}
	task_by_asin = {task.asin: task for task in tasks}
	task_tokens = {task.asin: _tokens(' '.join((task.instruction, *task.attributes, *task.options))) for task in tasks}

	def current_task(session_id: str) -> ProductTask:
		if session_id not in task_by_session:
			match = re.search(r'(\d+)$', session_id)
			index = int(match.group(1)) if match else 0
			task_by_session[session_id] = tasks[index % len(tasks)]
		return task_by_session[session_id]

	@app.route('/')
	def home() -> str:
		return redirect(url_for('index', session_id='browseruse_fixed_0'))

	@app.route('/<session_id>', methods=['GET', 'POST'])
	def index(session_id: str) -> str:
		task = current_task(session_id)
		if request.method == 'POST':
			query = request.form.get('search_query', '')
			return redirect(url_for('search_results', session_id=session_id, query=query))

		body = f"""
<h2>Shopping Task</h2>
<div class="instruction">{html.escape(task.instruction)}</div>
<form method="post">
  <label for="search_query">Search products</label><br>
  <input id="search_query" name="search_query" type="text" autofocus>
  <button type="submit">Search</button>
</form>"""
		return _page('WebShop Lite', body)

	@app.route('/search/<session_id>')
	def search_results(session_id: str) -> str:
		task = current_task(session_id)
		query = request.args.get('query', '')
		query_tokens = _tokens(query)
		scored = []
		for candidate in tasks:
			score = len(query_tokens & task_tokens[candidate.asin])
			if candidate.asin == task.asin:
				score += 3
			scored.append((score, candidate.index, candidate))
		scored.sort(reverse=True)
		results = [candidate for score, _, candidate in scored[:10] if score > 0] or [task]
		cards = []
		for candidate in results:
			cards.append(
				f"""<div class="product">
  <h3>{html.escape(candidate.instruction[:90])}</h3>
  <div class="meta">ASIN: {html.escape(candidate.asin)}</div>
  <p>Attributes: {html.escape(', '.join(candidate.attributes) or 'n/a')}</p>
  <p>Options: {html.escape(', '.join(candidate.options) or 'n/a')}</p>
  <a class="button" href="{url_for('product', session_id=session_id, asin=candidate.asin)}">View product</a>
</div>"""
			)
		body = f"""
<h2>Search Results</h2>
<div class="instruction">{html.escape(task.instruction)}</div>
<p>Query: {html.escape(query)}</p>
{''.join(cards)}
<p><a href="{url_for('index', session_id=session_id)}">Back to search</a></p>"""
		return _page('Search Results', body)

	@app.route('/product/<session_id>/<asin>', methods=['GET', 'POST'])
	def product(session_id: str, asin: str) -> str:
		task = current_task(session_id)
		product_task = task_by_asin.get(asin, task)
		if request.method == 'POST':
			selected_options = request.form.getlist('option')
			return redirect(url_for('done', session_id=session_id, asin=asin, options='|'.join(selected_options)))

		option_controls = []
		for option in product_task.options:
			option_controls.append(
				f'<label><input type="checkbox" name="option" value="{html.escape(option)}"> {html.escape(option)}</label><br>'
			)
		if not option_controls:
			option_controls.append('<div class="meta">No special option required.</div>')
		body = f"""
<h2>Product Detail</h2>
<div class="instruction">Task: {html.escape(task.instruction)}</div>
<div class="product">
  <h3>{html.escape(product_task.instruction)}</h3>
  <div class="meta">ASIN: {html.escape(product_task.asin)}</div>
  <p>Attributes: {html.escape(', '.join(product_task.attributes) or 'n/a')}</p>
  <form method="post">
    <h4>Options</h4>
    {''.join(option_controls)}
    <button type="submit">Buy Now</button>
  </form>
</div>
<p><a href="{url_for('index', session_id=session_id)}">Back to search</a></p>"""
		return _page('Product Detail', body)

	@app.route('/done/<session_id>/<asin>')
	def done(session_id: str, asin: str) -> str:
		task = current_task(session_id)
		selected_options = [option for option in request.args.get('options', '').split('|') if option]
		success = asin == task.asin
		if task.options:
			success = success and set(task.options).issubset(set(selected_options))
		status = 'SUCCESS' if success else 'PARTIAL_OR_WRONG_PRODUCT'
		body = f"""
<h2>Checkout Complete</h2>
<div class="done">
  <strong>Status: {status}</strong>
  <p>Target ASIN: {html.escape(task.asin)}</p>
  <p>Purchased ASIN: {html.escape(asin)}</p>
  <p>Required options: {html.escape(', '.join(task.options) or 'n/a')}</p>
  <p>Selected options: {html.escape(', '.join(selected_options) or 'n/a')}</p>
</div>"""
		return _page('Checkout Complete', body)

	return app


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--input', type=Path, required=True)
	parser.add_argument('--host', default='127.0.0.1')
	parser.add_argument('--port', type=int, default=3000)
	parser.add_argument('--limit', type=int, default=1000)
	args = parser.parse_args()

	tasks = load_tasks(args.input, args.limit)
	if not tasks:
		raise RuntimeError(f'No WebShop tasks loaded from {args.input}')
	app = create_app(tasks)
	app.run(host=args.host, port=args.port)


if __name__ == '__main__':
	main()
