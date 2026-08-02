"""Run the fair visible-DOM WebShop transaction without invoking an LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from browser_use.browser.events import NavigateToUrlEvent
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from tests.ci.evaluate_tasks import _build_webshop_web_only_tools


async def run(url: str) -> None:
	profile = BrowserProfile(
		headless=True,
		user_data_dir=None,
		keep_alive=False,
		chromium_sandbox=False,
		allowed_domains=['127.0.0.1'],
	)
	session = BrowserSession(browser_profile=profile)
	try:
		await session.start()
		event = session.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=False))
		await event
		tools = _build_webshop_web_only_tools(f'Open {url} and complete the visible shopping instruction.')
		action = tools.registry.registry.actions['execute_visible_webshop_transaction']
		result = await action.function(browser_session=session)
		content = result.extracted_content or result.error or ''
		if content.startswith('VISIBLE_WEBSHOP_TRANSACTION='):
			payload = json.loads(content.split('=', 1)[1])
			print(json.dumps({
				'status': payload.get('status'),
				'final_url': payload.get('final_url'),
				'reward': payload.get('reward'),
				'transitions': payload.get('transitions') or len(payload.get('trace', [])),
			}, ensure_ascii=False))
		else:
			print(content)
	finally:
		await session.stop()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('url')
	asyncio.run(run(parser.parse_args().url))
