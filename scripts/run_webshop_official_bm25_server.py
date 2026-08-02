"""Run a WebShop HTML server using official templates/reward and BM25 search."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import types
from ast import literal_eval
from pathlib import Path

from flask import Flask, redirect, request, url_for
from rank_bm25 import BM25Okapi


def _install_optional_dependency_shims() -> None:
	"""Provide tiny shims for heavy optional dependencies when unavailable."""

	if 'pyserini' not in sys.modules:
		pyserini = types.ModuleType('pyserini')
		search = types.ModuleType('pyserini.search')
		lucene = types.ModuleType('pyserini.search.lucene')

		class LuceneSearcher:  # pragma: no cover - only used to satisfy imports
			def __init__(self, *_args, **_kwargs):
				raise RuntimeError('LuceneSearcher is replaced by the BM25 server.')

		lucene.LuceneSearcher = LuceneSearcher
		search.lucene = lucene
		pyserini.search = search
		sys.modules['pyserini'] = pyserini
		sys.modules['pyserini.search'] = search
		sys.modules['pyserini.search.lucene'] = lucene

	if 'spacy' not in sys.modules:
		spacy = types.ModuleType('spacy')

		class Token:
			def __init__(self, text: str):
				self.text = text
				self.pos_ = 'NOUN'

		class SimpleNlp:
			def __call__(self, text: str):
				return [Token(token) for token in re.findall(r'[a-zA-Z0-9]+', text)]

		spacy.load = lambda *_args, **_kwargs: SimpleNlp()
		sys.modules['spacy'] = spacy


def _tokenize(text: str) -> list[str]:
	return re.findall(r'[a-z0-9]+', text.lower())


def create_app(webshop_root: Path, num_products: int | None, data_dir: Path | None = None) -> Flask:
	sys.path.insert(0, str(webshop_root))
	_install_optional_dependency_shims()

	if data_dir is not None:
		from web_agent_site import utils  # noqa: PLC0415

		utils.DEFAULT_FILE_PATH = str(data_dir / 'items_shuffle_1000.json')
		utils.DEFAULT_ATTR_PATH = str(data_dir / 'items_ins_v2_1000.json')
		utils.HUMAN_ATTR_PATH = str(data_dir / 'items_human_ins.json')

	from web_agent_site.engine.engine import (  # noqa: PLC0415
		END_BUTTON,
		convert_web_app_string_to_var,
		get_product_per_page,
		load_products,
		map_action_to_html,
	)
	from web_agent_site.engine.goal import get_goals, get_reward  # noqa: PLC0415
	from web_agent_site.utils import DEFAULT_FILE_PATH, generate_mturk_code  # noqa: PLC0415

	app = Flask(__name__, static_folder=str(webshop_root / 'web_agent_site' / 'static'))
	all_products, product_item_dict, product_prices, attribute_to_asins = load_products(
		filepath=DEFAULT_FILE_PATH,
		num_products=num_products,
	)
	goals = get_goals(all_products, product_prices)
	random.seed(233)
	random.shuffle(goals)
	weights = [goal['weight'] for goal in goals]
	user_sessions: dict[str, dict] = {}

	corpus = []
	for product in all_products:
		text = ' '.join(
			[
				product.get('Title', ''),
				product.get('Description', ''),
				' '.join(product.get('BulletPoints') or []),
				' '.join(product.get('Attributes') or []),
				product.get('query', ''),
			]
		)
		corpus.append(_tokenize(text))
	bm25 = BM25Okapi(corpus)

	def _session_goal(session_id: str) -> dict:
		if session_id in user_sessions:
			return user_sessions[session_id]['goal']
		if 'fixed' in session_id:
			index = int(session_id.split('_')[-1])
			goal = goals[index % len(goals)]
		else:
			goal = random.choices(goals, weights)[0]
		user_sessions[session_id] = {'goal': goal, 'done': False}
		return goal

	def _search_products(keywords: list[str]) -> list[dict]:
		if keywords[0] == '<r>':
			return random.sample(all_products, k=min(50, len(all_products)))
		if keywords[0] == '<a>':
			attribute = ' '.join(keywords[1:]).strip()
			asins = attribute_to_asins[attribute]
			return [product for product in all_products if product['asin'] in asins]
		if keywords[0] == '<c>':
			category = keywords[1].strip()
			return [product for product in all_products if product['category'] == category]
		if keywords[0] == '<q>':
			query = ' '.join(keywords[1:]).strip()
			return [product for product in all_products if product['query'] == query]
		query_tokens = _tokenize(' '.join(keywords))
		scores = bm25.get_scores(query_tokens)
		indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:50]
		return [all_products[index] for index in indexes if scores[index] > 0] or all_products[:50]

	@app.route('/')
	def home():
		return redirect(url_for('index', session_id='browseruse_fixed_0'))

	@app.route('/<session_id>', methods=['GET', 'POST'])
	def index(session_id):
		goal = _session_goal(session_id)
		if request.method == 'POST' and 'search_query' in request.form:
			keywords = request.form['search_query'].lower().split(' ')
			return redirect(url_for('search_results', session_id=session_id, keywords=keywords, page=1))
		return map_action_to_html('start', session_id=session_id, instruction_text=goal['instruction_text'])

	@app.route('/search_results/<session_id>/<keywords>/<page>', methods=['GET', 'POST'])
	def search_results(session_id, keywords, page):
		goal = _session_goal(session_id)
		page = convert_web_app_string_to_var('page', page)
		keywords = convert_web_app_string_to_var('keywords', keywords)
		top_n_products = _search_products(keywords)
		products = get_product_per_page(top_n_products, page)
		return map_action_to_html(
			'search',
			session_id=session_id,
			products=products,
			keywords=keywords,
			page=page,
			total=len(top_n_products),
			instruction_text=goal['instruction_text'],
		)

	@app.route('/item_page/<session_id>/<asin>/<keywords>/<page>/<options>', methods=['GET', 'POST'])
	def item_page(session_id, asin, keywords, page, options):
		goal = _session_goal(session_id)
		options = literal_eval(options)
		product_info = product_item_dict[asin]
		product_info['goal_instruction'] = goal['instruction_text']
		return map_action_to_html(
			'click',
			session_id=session_id,
			product_info=product_info,
			keywords=keywords,
			page=page,
			asin=asin,
			options=options,
			instruction_text=goal['instruction_text'],
			show_attrs=True,
		)

	@app.route('/item_sub_page/<session_id>/<asin>/<keywords>/<page>/<sub_page>/<options>', methods=['GET', 'POST'])
	def item_sub_page(session_id, asin, keywords, page, sub_page, options):
		goal = _session_goal(session_id)
		options = literal_eval(options)
		product_info = product_item_dict[asin]
		product_info['goal_instruction'] = goal['instruction_text']
		return map_action_to_html(
			f'click[{sub_page}]',
			session_id=session_id,
			product_info=product_info,
			keywords=keywords,
			page=page,
			asin=asin,
			options=options,
			instruction_text=goal['instruction_text'],
		)

	@app.route('/done/<session_id>/<asin>/<options>', methods=['GET', 'POST'])
	def done(session_id, asin, options):
		options = literal_eval(options)
		goal = _session_goal(session_id)
		purchased_product = product_item_dict[asin]
		price = product_prices[asin]
		reward, reward_info = get_reward(purchased_product, goal, price=price, options=options, verbose=True)
		user_sessions[session_id]['done'] = True
		user_sessions[session_id]['reward'] = reward
		return map_action_to_html(
			f'click[{END_BUTTON}]',
			session_id=session_id,
			reward=reward,
			asin=asin,
			options=options,
			reward_info=reward_info,
			goal_attrs=goal['attributes'],
			purchased_attrs=purchased_product['Attributes'],
			goal=goal,
			mturk_code=generate_mturk_code(session_id),
			query=purchased_product['query'],
			category=purchased_product['category'],
			product_category=purchased_product['product_category'],
		)

	return app


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--webshop-root', type=Path, required=True)
	parser.add_argument('--host', default='127.0.0.1')
	parser.add_argument('--port', type=int, default=3002)
	parser.add_argument('--num-products', type=int, default=None)
	parser.add_argument('--data-dir', type=Path, default=None)
	args = parser.parse_args()

	logging.getLogger('werkzeug').setLevel(logging.WARNING)
	app = create_app(args.webshop_root, args.num_products, args.data_dir)
	app.run(host=args.host, port=args.port)


if __name__ == '__main__':
	main()
