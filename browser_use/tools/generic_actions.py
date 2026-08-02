"""Cross-site browser actions that resolve targets from the current visible state."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from pydantic import BaseModel, Field

from browser_use.agent.views import ActionResult
from browser_use.browser.events import ClickElementEvent
from browser_use.browser.session import BrowserSession
from browser_use.tools.control_state import validate_commit_click, validate_control_mutation
from browser_use.tools.generic_task_runtime import (
	StructuredVisibleRecord,
	rank_visible_comparison_candidates,
	resolve_comparison_spec,
)


class ActivateVisibleLabelAction(BaseModel):
	"""Resolve a visible control by its current accessible label and activate it once."""

	label: str = Field(min_length=1, max_length=200)


class ActivateRecordControlAction(BaseModel):
	"""Activate a control inside the current row/card/section containing record text."""

	record_text: str = Field(min_length=1, max_length=200)
	control_label: str = Field(min_length=1, max_length=200)


class InspectFormStateAction(BaseModel):
	"""Inspect live values and validity for visible form controls."""


class InspectRankedCandidatesAction(BaseModel):
	"""Rank candidate cards currently rendered in the visible-page DOM."""


class ClearExistingCollectionAction(BaseModel):
	"""Clear an explicitly requested pre-existing cart or basket before adding a new target."""


class SelectOnlyVisibleChoiceAction(BaseModel):
	"""Select one unambiguous visible radio choice in the current form."""


class NavigateVisibleLinkAction(BaseModel):
	"""Navigate through a currently visible link, optionally scoped by nearby text."""

	label: str = Field(min_length=1, max_length=200)
	context: str = Field(default='', max_length=200)


class SetCurrentUrlQueryAction(BaseModel):
	"""Set an observed query parameter to an exact task-requested value."""

	parameter: str = Field(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_.\-\[\]]+$')
	value: str = Field(min_length=1, max_length=200)


class SubmitSiteSearchAction(BaseModel):
	"""Fill and submit the current site's visible search form as one action."""

	query: str = Field(min_length=1, max_length=500)


class FillVisibleFormFieldsAction(BaseModel):
	"""Fill several visible form controls by their labels and verify live values."""

	fields: dict[str, str] = Field(min_length=1, max_length=20)


class HoverVisibleLabelAction(BaseModel):
	"""Hover a currently visible control so its menu or tooltip becomes visible."""

	label: str = Field(min_length=1, max_length=200)


def normalize_label(value: str) -> str:
	return re.sub(r'[^a-z0-9]+', ' ', value.casefold()).strip()


def element_labels(node: Any) -> list[str]:
	labels: list[str] = []
	if getattr(node, 'ax_node', None) and node.ax_node.name:
		labels.append(str(node.ax_node.name))
	attributes = getattr(node, 'attributes', None) or {}
	for key in ('aria-label', 'title', 'placeholder', 'value', 'name'):
		if attributes.get(key):
			labels.append(str(attributes[key]))
	return list(dict.fromkeys(label for label in labels if normalize_label(label)))


def rank_labeled_elements(label: str, selector_map: dict[int, Any]) -> list[tuple[int, int, str]]:
	"""Rank current controls without retaining a DOM index across page changes."""

	target = normalize_label(label)
	if not target:
		return []
	ranked: list[tuple[int, int, str]] = []
	for index, node in selector_map.items():
		node_name = str(getattr(node, 'node_name', '')).casefold()
		attributes = getattr(node, 'attributes', None) or {}
		role = str(attributes.get('role') or '').casefold()
		interactive_bonus = 12 if node_name in {'a', 'button', 'input', 'select', 'summary'} else 0
		interactive_bonus += 8 if role in {'button', 'link', 'tab', 'menuitem', 'option'} else 0
		for candidate in element_labels(node):
			normalized = normalize_label(candidate)
			if normalized == target:
				score = 100 + interactive_bonus
			elif normalized.startswith(target) or target.startswith(normalized):
				score = 75 + interactive_bonus
			elif target in normalized:
				score = 55 + interactive_bonus
			else:
				continue
			ranked.append((score, index, candidate))
	return sorted(ranked, key=lambda item: (item[0], -item[1]), reverse=True)


def rank_record_controls(
	record_text: str,
	control_label: str,
	selector_map: dict[int, Any],
) -> list[tuple[int, int, str]]:
	"""Resolve repeated controls by the nearest visible ancestor containing a record key."""

	record = normalize_label(record_text)
	ranked: list[tuple[int, int, str]] = []
	for label_score, index, label in rank_labeled_elements(control_label, selector_map):
		node = selector_map[index]
		ancestor = node
		for depth in range(9):
			if ancestor is None:
				break
			try:
				context = normalize_label(ancestor.get_all_children_text(max_depth=6))
			except Exception:
				context = normalize_label(
					str(getattr(ancestor, 'node_value', '') or '')
					+ ' '
					+ str(getattr(getattr(ancestor, 'ax_node', None), 'name', '') or '')
				)
			if record and record in context:
				ranked.append((label_score + 200 - depth, index, label))
				break
			ancestor = getattr(ancestor, 'parent_node', None)
	return sorted(ranked, key=lambda item: (item[0], -item[1]), reverse=True)


async def reveal_navigation_target(browser_session: BrowserSession, label: str) -> dict[str, Any]:
	"""Probe visible menu triggers and return a target only after it becomes visibly rendered."""

	state = await browser_session.get_browser_state_summary(include_screenshot=False)
	selector_map = state.dom_state.selector_map if state.dom_state else {}
	for _, index, resolved_label in rank_labeled_elements(label, selector_map):
		node = selector_map[index]
		href = str((getattr(node, 'attributes', None) or {}).get('href') or '').strip()
		if href:
			return {'found': True, 'href': urljoin(state.url, href), 'label': resolved_label, 'parent': ''}

	cdp_session = await browser_session.get_or_create_cdp_session()
	async def visible_links() -> list[dict[str, Any]]:
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={
				'expression': r"""
Array.from(document.querySelectorAll('a[href], [role="menuitem"]')).map((el) => {
  const style = getComputedStyle(el);
  const rect = el.getBoundingClientRect();
  const visible = style.display !== 'none' && style.visibility !== 'hidden' &&
    Number(style.opacity || 1) > 0 && rect.width > 0 && rect.height > 0;
  return visible ? {
    label: String(el.innerText || el.getAttribute('aria-label') || el.title || '').replace(/\s+/g, ' ').trim(),
    href: el.href || '',
    in_nav: Boolean(el.closest('nav, [role="menubar"]')),
    x: rect.x, y: rect.y, width: rect.width, height: rect.height,
  } : null;
}).filter(Boolean)
""",
				'returnByValue': True,
			},
			session_id=cdp_session.session_id,
		)
		return [
			item for item in (result.get('result', {}).get('value') or [])
			if isinstance(item, dict) and normalize_label(str(item.get('label') or ''))
		]

	def target_from(values: list[dict[str, Any]]) -> dict[str, Any] | None:
		target = normalize_label(label)
		for item in values:
			candidate = normalize_label(str(item.get('label') or ''))
			if candidate == target or candidate == target + 's' or candidate + 's' == target:
				return item
		return None

	initial = await visible_links()
	direct = target_from(initial)
	if direct and direct.get('href'):
		return {'found': True, 'href': direct['href'], 'label': direct['label'], 'parent': ''}
	queue: list[tuple[dict[str, Any], list[str]]] = [
		(item, []) for item in initial if item.get('in_nav')
	]
	seen_triggers: set[str] = set()
	explored: list[str] = []
	target_terms = normalize_label(label).split()

	def lexical_score(value: str) -> int:
		terms = normalize_label(value).split()
		return sum(
			1
			for target_term in target_terms
			if any(
				term == target_term
				or term.startswith(target_term)
				or target_term.startswith(term)
				for term in terms
			)
		)

	while queue and len(explored) < 120:
		trigger, path = queue.pop(0)
		trigger_label = str(trigger.get('label') or '')
		trigger_key = f'{normalize_label(trigger_label)}|{trigger.get("href") or ""}'
		if trigger_key in seen_triggers or len(path) >= 3:
			continue
		seen_triggers.add(trigger_key)
		explored.append(trigger_label)
		before_keys = {
			f'{normalize_label(str(item.get("label") or ""))}|{item.get("href") or ""}'
			for item in await visible_links()
		}
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={
				'type': 'mouseMoved',
				'x': float(trigger.get('x') or 0) + float(trigger.get('width') or 0) / 2,
				'y': float(trigger.get('y') or 0) + float(trigger.get('height') or 0) / 2,
			},
			session_id=cdp_session.session_id,
		)
		await asyncio.sleep(0.5)
		revealed = await visible_links()
		target = target_from(revealed)
		if target and target.get('href'):
			return {
				'found': True,
				'href': target['href'],
				'label': target['label'],
				'parent': ' > '.join(path + [trigger_label]),
			}
		new_path = path + [trigger_label]
		new_triggers = sorted(
			[
				(item, new_path) for item in revealed
				if item.get('in_nav')
				and f'{normalize_label(str(item.get("label") or ""))}|{item.get("href") or ""}' not in before_keys
				and lexical_score(str(item.get('label') or '')) > 0
			],
			key=lambda item: lexical_score(str(item[0].get('label') or '')),
			reverse=True,
		)
		queue = new_triggers + queue
	return {'found': False, 'explored': explored[:60]}


def register_generic_actions(tools: Any, *, task_intent: str = '') -> None:
	"""Register actions that depend only on live visible DOM semantics."""

	existing_collection_cleared = False

	@tools.action(
		description=(
			'Hover over a currently visible link, menu item, or control by accessible label to reveal a submenu or '
			'other hover-only content. Use this before choosing child links in a hierarchical navigation menu.'
		),
		param_model=HoverVisibleLabelAction,
	)
	async def hover_visible_label(
		params: HoverVisibleLabelAction,
		browser_session,
	) -> ActionResult:
		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		selector_map = state.dom_state.selector_map if state.dom_state else {}
		ranked = rank_labeled_elements(params.label, selector_map)
		if not ranked:
			return ActionResult(error=f'No current visible element matches label {params.label!r}.')
		_, index, label = ranked[0]
		node = selector_map[index]
		position = getattr(node, 'absolute_position', None)
		if position is None or position.width <= 0 or position.height <= 0:
			return ActionResult(error=f'Visible element {label!r} has no hoverable bounds.')
		cdp_session = await browser_session.get_or_create_cdp_session()
		await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
			params={
				'type': 'mouseMoved',
				'x': position.x + position.width / 2,
				'y': position.y + position.height / 2,
			},
			session_id=cdp_session.session_id,
		)
		await asyncio.sleep(0.5)
		return ActionResult(
			extracted_content=f'Hovered current visible element {label!r}; inspect the refreshed state for submenu links.',
			long_term_memory=f'Hovered {label!r} to reveal its current menu content.',
			metadata={'resolved_index': index, 'resolved_label': label},
		)

	@tools.action(
		description=(
			'Inspect every visible form control before saving, submitting, or declaring a draft ready. Returns labels, '
			'current values, required/valid state, and validation messages. Use it after filling a multi-field form and '
			'again when a submit click does not produce confirmation.'
		),
		param_model=InspectFormStateAction,
	)
	async def inspect_form_state(
		params: InspectFormStateAction,
		browser_session,
	) -> ActionResult:
		cdp_session = await browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={
				'expression': r"""
(() => {
  const visible = (el) => {
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      rect.width > 0 && rect.height > 0;
  };
  const controls = Array.from(document.querySelectorAll('input, textarea, select'))
    .filter(visible)
    .map((el) => {
      const id = el.id || '';
      const explicit = id ? document.querySelector(`label[for="${CSS.escape(id)}"]`) : null;
      const wrapped = el.closest('label');
      const label = (explicit?.innerText || wrapped?.innerText ||
        el.getAttribute('aria-label') || el.name || el.placeholder || el.type || '').trim();
      const value = el instanceof HTMLSelectElement
        ? Array.from(el.selectedOptions).map((option) => option.text.trim()).join(', ')
        : el.type === 'checkbox' || el.type === 'radio'
          ? (el.checked ? (el.value || 'checked') : '')
          : el.value;
      return {
        label: label.slice(0, 160),
        name: el.name || '',
        type: el.type || el.tagName.toLowerCase(),
        value: String(value || '').slice(0, 500),
        required: Boolean(el.required),
        valid: typeof el.checkValidity === 'function' ? el.checkValidity() : true,
        validation_message: String(el.validationMessage || '').slice(0, 240),
      };
    });
  return {
    controls,
    missing_required: controls
      .filter((item) => item.required && (!item.value || !item.valid))
      .map((item) => item.label || item.name),
  };
})()
""",
				'returnByValue': True,
				'awaitPromise': True,
			},
			session_id=cdp_session.session_id,
		)
		value = result.get('result', {}).get('value') or {'controls': [], 'missing_required': []}
		missing = value.get('missing_required', []) if isinstance(value, dict) else []
		return ActionResult(
			extracted_content=json.dumps(value, ensure_ascii=False),
			long_term_memory=(
				f'Form inspection found {len(missing)} missing or invalid required controls: {missing[:8]}.'
				if missing
				else 'Form inspection found no missing visible required controls.'
			),
			metadata={'missing_required': missing},
		)

	@tools.action(
		description=(
			'Select the only currently visible, enabled, unchecked radio choice in the active form. Use when a '
			'dynamic form has finished loading and exposes exactly one valid choice before a Next or Continue '
			'button. The action refuses to choose when zero or multiple choices are available.'
		),
		param_model=SelectOnlyVisibleChoiceAction,
	)
	async def select_only_visible_choice(
		params: SelectOnlyVisibleChoiceAction,
		browser_session,
	) -> ActionResult:
		cdp_session = await browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={
				'expression': r"""
(() => {
  const visible = (el) => {
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      rect.width > 0 && rect.height > 0;
  };
  const choices = Array.from(document.querySelectorAll('input[type="radio"]'))
    .filter((el) => visible(el) && !el.disabled && !el.checked);
  const describe = (el) => {
    const id = el.id || '';
    const explicit = id ? document.querySelector(`label[for="${CSS.escape(id)}"]`) : null;
    const row = el.closest('tr, li, label, fieldset, [role="radiogroup"], .field, .control');
    return (explicit?.innerText || row?.innerText || el.getAttribute('aria-label') ||
      el.value || el.name || 'radio').trim().replace(/\s+/g, ' ').slice(0, 240);
  };
  if (choices.length !== 1) {
    return {selected: false, choice_count: choices.length, choices: choices.map(describe).slice(0, 12)};
  }
  const choice = choices[0];
  choice.scrollIntoView({block: 'center', inline: 'nearest'});
  choice.click();
  choice.dispatchEvent(new Event('input', {bubbles: true}));
  choice.dispatchEvent(new Event('change', {bubbles: true}));
  return {selected: Boolean(choice.checked), choice_count: 1, choice: describe(choice), value: choice.value || ''};
})()
""",
				'returnByValue': True,
				'awaitPromise': True,
			},
			session_id=cdp_session.session_id,
		)
		value = result.get('result', {}).get('value') or {}
		if not value.get('selected'):
			return ActionResult(
				error=(
					'The current form does not expose exactly one unambiguous visible radio choice; '
					'inspect current choices and select by visible label.'
				),
				metadata=value,
			)
		await asyncio.sleep(1)
		return ActionResult(
			extracted_content=f"Selected the only visible form choice: {value.get('choice')!r}.",
			long_term_memory=(
				f"Current form choice selected from live DOM: {value.get('choice')!r}. "
				'Continue with the current Next or Continue control.'
			),
			metadata=value,
		)

	@tools.action(
		description=(
			'Inspect and deterministically rank the candidate cards currently rendered on the page for the user '
			'request. Use after navigating to the exact requested collection, applying a useful visible sort, and '
			'choosing the largest page size. Returns same-record title, price, rating, link, controls, and whether '
			'the current price ordering is sufficient to cover a requested price bound.'
		),
		param_model=InspectRankedCandidatesAction,
	)
	async def inspect_ranked_candidates(
		params: InspectRankedCandidatesAction,
		browser_session,
	) -> ActionResult:
		if not task_intent.strip():
			return ActionResult(error='No task intent is available for deterministic candidate ranking.')
		cdp_session = await browser_session.get_or_create_cdp_session()
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={
				'expression': r"""
(() => {
  const clean = (value) => String(value || '').replace(/\s+/g, ' ').trim();
  const visible = (el) => {
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      Number(style.opacity || 1) > 0 && rect.width > 0 && rect.height > 0;
  };
  const selectors = [
    'li.product-item', '.product-item-info', '[data-container="product-grid"]',
    'article', '[role="listitem"]', 'tr'
  ];
  const seen = new Set();
  const records = [];
  for (const el of document.querySelectorAll(selectors.join(','))) {
    if (!visible(el)) continue;
    const text = clean(el.innerText);
    if (text.length < 12 || text.length > 3000 || !/\$[\d,]+|rating|stars?/i.test(text)) continue;
    const signature = text.toLowerCase();
    if (seen.has(signature)) continue;
    seen.add(signature);
    const controls = Array.from(el.querySelectorAll('button, input[type="submit"], [role="button"], a'))
      .filter(visible)
      .map((control) => clean(
        control.innerText || control.value || control.getAttribute('aria-label') ||
        control.title || control.name
      ))
      .filter(Boolean)
      .slice(0, 30);
    const links = Array.from(el.querySelectorAll('a[href]'))
      .filter(visible)
      .map((link) => ({label: clean(link.innerText || link.title), href: link.href}))
      .filter((link) => link.href)
      .slice(0, 30);
    records.push({
      container_signature: el.tagName.toLowerCase() + '|' + clean(el.className).slice(0, 120),
      text,
      controls,
      links,
    });
    if (records.length >= 120) break;
  }
  const sorter = document.querySelector('select[data-role="sorter"], select#sorter');
  const limiter = document.querySelector('select[data-role="limiter"], select#limiter');
  const direction = document.querySelector('[data-role="direction-switcher"], .sorter-action');
  return {
    url: location.href,
    records,
    sort_label: sorter ? clean(sorter.selectedOptions?.[0]?.text || sorter.value) : '',
    page_size: limiter ? clean(limiter.selectedOptions?.[0]?.text || limiter.value) : '',
    direction: direction ? clean(
      direction.title || direction.getAttribute('aria-label') ||
      direction.getAttribute('data-value') || direction.className
    ) : '',
  };
})()
""",
				'returnByValue': True,
				'awaitPromise': True,
			},
			session_id=cdp_session.session_id,
		)
		value = result.get('result', {}).get('value') or {}
		url = str(value.get('url') or await browser_session.get_current_page_url())
		records = [
			StructuredVisibleRecord(
				record_id=f'live-{index}',
				url=url,
				container_signature=str(item.get('container_signature') or 'candidate-card'),
				text=str(item.get('text') or ''),
				controls=[str(control) for control in item.get('controls', []) if str(control).strip()],
				links=[
					{'label': str(link.get('label') or ''), 'href': str(link.get('href') or '')}
					for link in item.get('links', [])
					if isinstance(link, dict) and str(link.get('href') or '').strip()
				],
			)
			for index, item in enumerate(value.get('records', []))
			if isinstance(item, dict) and str(item.get('text') or '').strip()
		]
		ranked = rank_visible_comparison_candidates(task_intent, records)
		spec = resolve_comparison_spec(task_intent)
		sort_label = str(value.get('sort_label') or '')
		direction = str(value.get('direction') or '')
		price_descending = 'price' in sort_label.casefold() and (
			'ascending' in direction.casefold()
			or 'asc' in direction.casefold()
			or 'desc' in urlsplit(url).query.casefold()
		)
		visible_prices = [item.price for item in ranked if item.price is not None]
		coverage_complete = bool(
			spec.min_price is not None
			and price_descending
			and records
			and any(
				(price_match := re.search(r'\$\s*([\d,]+(?:\.\d{1,2})?)', record.text))
				and float(price_match.group(1).replace(',', '')) <= spec.min_price
				for record in records
			)
		)
		payload = {
			'spec': spec.model_dump(mode='json'),
			'sort_label': sort_label,
			'direction_control': direction,
			'page_size': value.get('page_size'),
			'coverage_complete_for_price_bound': coverage_complete,
			'candidates': [item.model_dump(mode='json') for item in ranked],
		}
		if not ranked:
			return ActionResult(
				error=(
					'No rendered candidate card satisfies the comparison constraints. Navigate to the exact '
					'collection, sort by the numeric bound, increase page size, or inspect the next page.'
				),
				metadata=payload,
			)
		best = ranked[0]
		return ActionResult(
			extracted_content=json.dumps(payload, ensure_ascii=False),
			long_term_memory=(
				f'Current deterministic leader is {best.title!r} '
				f'(rating={best.rating}, price={best.price}); '
				f'price-bound coverage complete={coverage_complete}.'
			),
			metadata=payload,
		)

	@tools.action(
		description=(
			'Clear all pre-existing items from the current cart or basket as one verified prerequisite phase. Use '
			'exactly once, before adding the requested new item, and only when the user explicitly requested '
			'discarding, removing, clearing, or emptying existing cart/basket contents. Each removal and confirmation '
			'is executed through current live controls and the action stops if item count does not decrease.'
		),
		param_model=ClearExistingCollectionAction,
	)
	async def clear_existing_collection(
		params: ClearExistingCollectionAction,
		browser_session,
	) -> ActionResult:
		nonlocal existing_collection_cleared
		if existing_collection_cleared:
			return ActionResult(error='The pre-existing collection cleanup phase has already been executed once.')
		if not (
			re.search(r'\b(?:discard|clear|empty|remove|delete)\b', task_intent, re.IGNORECASE)
			and re.search(r'\b(?:cart|basket)\b', task_intent, re.IGNORECASE)
		):
			return ActionResult(
				error='The user request does not explicitly authorize clearing an existing cart or basket.'
			)

		removed = 0
		initial_remove_count: int | None = None
		initial_item_count: int | None = None
		transient_missing_controls = 0

		def visible_item_count(text: str, current_map: dict[int, Any]) -> int | None:
			for node in current_map.values():
				for label in element_labels(node):
					normalized = normalize_label(label)
					match = re.search(r'\b(?:my\s+)?(?:cart|basket)\s+(\d+)\b', normalized)
					if match:
						return int(match.group(1))
			for pattern in (
				r'\b(\d+)\s+Items?\s+in\s+(?:Cart|Basket)\b',
				r'\b(?:Cart|Basket)\s*\(\s*(\d+)\s*\)',
			):
				match = re.search(pattern, text, re.IGNORECASE)
				if match:
					return int(match.group(1))
			return None

		def visibly_empty(text: str) -> bool:
			return bool(
				re.search(
					r'\b(?:you\s+have\s+no\s+items|cart\s+is\s+empty|basket\s+is\s+empty|0\s+items?\s+in\s+(?:cart|basket))\b',
					text,
					re.IGNORECASE,
				)
			)

		for _ in range(30):
			state = await browser_session.get_browser_state_summary(include_screenshot=False)
			selector_map = state.dom_state.selector_map if state.dom_state else {}
			page_text = state.dom_state.llm_representation() if state.dom_state else ''
			scope = f'{state.url} {page_text}'.casefold()
			if not re.search(r'\b(?:cart|basket|items?\s+in\s+cart|cart\s+subtotal)\b', scope):
				return ActionResult(
					error='Current visible state is not a cart or basket; navigate there before cleanup.',
					metadata={'removed': removed},
				)
			before_count = visible_item_count(page_text, selector_map)
			if initial_item_count is None and before_count is not None:
				initial_item_count = before_count
			if visibly_empty(page_text) or before_count == 0:
				existing_collection_cleared = True
				return ActionResult(
					extracted_content=f'Cleared {removed} pre-existing items; the current cart or basket is empty.',
					long_term_memory=(
						f'Prerequisite cleanup completed before target selection: removed {removed} existing items.'
					),
					metadata={
						'removed': removed,
						'initial_item_count': initial_item_count,
						'initial_remove_controls': initial_remove_count,
						'verified_empty': True,
					},
				)

			remove_ranked: list[tuple[int, int, str]] = []
			for label in ('Remove', 'Remove This Item', 'Delete'):
				remove_ranked.extend(rank_labeled_elements(label, selector_map))
			seen_indices: set[int] = set()
			remove_ranked = [
				item for item in remove_ranked
				if not (item[1] in seen_indices or seen_indices.add(item[1]))
			]
			if initial_remove_count is None:
				initial_remove_count = len(remove_ranked)
			if not remove_ranked:
				transient_missing_controls += 1
				if transient_missing_controls <= 5:
					await asyncio.sleep(0.6)
					continue
				return ActionResult(
					error=(
						f'Cart or basket still reports {before_count!r} items, but no visible remove control '
						'became stable; empty state was not verified.'
					),
					metadata={'removed': removed, 'visible_item_count': before_count, 'verified_empty': False},
				)
			transient_missing_controls = 0

			_, remove_index, _ = remove_ranked[0]
			remove_node = selector_map[remove_index]
			commit_validation = await validate_commit_click(browser_session, remove_node)
			if commit_validation.is_commit and not commit_validation.allowed:
				return ActionResult(
					error=f'Cleanup commit blocked: {commit_validation.reason}',
					metadata={'removed': removed},
				)
			mutation_validation = await validate_control_mutation(browser_session, remove_node)
			if not mutation_validation.allowed:
				return ActionResult(
					error=f'Cleanup mutation blocked: {mutation_validation.reason}',
					metadata={'removed': removed},
				)
			event = browser_session.event_bus.dispatch(ClickElementEvent(node=remove_node))
			await event
			await event.event_result(raise_if_any=True, raise_if_none=False)
			await asyncio.sleep(0.4)

			confirmation_clicked = False
			for _confirmation_attempt in range(10):
				confirm_state = await browser_session.get_browser_state_summary(include_screenshot=False)
				confirm_map = confirm_state.dom_state.selector_map if confirm_state.dom_state else {}
				confirm_ranked: list[tuple[int, int, str]] = []
				for label in ('OK', 'Confirm', 'Yes', 'Yes, Remove', 'Confirm Removal'):
					confirm_ranked.extend(rank_labeled_elements(label, confirm_map))
				if confirm_ranked:
					_, confirm_index, _ = confirm_ranked[0]
					confirm_node = confirm_map[confirm_index]
					confirm_validation = await validate_commit_click(browser_session, confirm_node)
					if confirm_validation.is_commit and not confirm_validation.allowed:
						return ActionResult(
							error=f'Cleanup confirmation blocked: {confirm_validation.reason}',
							metadata={'removed': removed},
						)
					confirm_event = browser_session.event_bus.dispatch(ClickElementEvent(node=confirm_node))
					await confirm_event
					await confirm_event.event_result(raise_if_any=True, raise_if_none=False)
					confirmation_clicked = True
					break
				await asyncio.sleep(0.3)
			await asyncio.sleep(0.4 if confirmation_clicked else 0.8)

			progress_observed = False
			after_count: int | None = None
			after_indices: set[int] = set()
			for _progress_attempt in range(15):
				after_state = await browser_session.get_browser_state_summary(include_screenshot=False)
				after_text = after_state.dom_state.llm_representation() if after_state.dom_state else ''
				after_map = after_state.dom_state.selector_map if after_state.dom_state else {}
				after_count = visible_item_count(after_text, after_map)
				after_indices = {
					item[1]
					for label in ('Remove', 'Remove This Item', 'Delete')
					for item in rank_labeled_elements(label, after_map)
				}
				if visibly_empty(after_text) or after_count == 0:
					progress_observed = True
					break
				if before_count is not None and after_count is not None and after_count < before_count:
					progress_observed = True
					break
				if (
					before_count is None
					and len(after_indices) < len({item[1] for item in remove_ranked})
				):
					progress_observed = True
					break
				await asyncio.sleep(0.4)
			if not progress_observed:
				# Dynamic collection pages can briefly expose neither the old count nor the new empty state
				# after a successful removal. Reload once, then require stable visible evidence.
				cdp_session = await browser_session.get_or_create_cdp_session()
				await cdp_session.cdp_client.send.Page.reload(
					params={'ignoreCache': True},
					session_id=cdp_session.session_id,
				)
				await asyncio.sleep(2)
				await browser_session.navigate_to(state.url, new_tab=False)
				await asyncio.sleep(3)
				stable_state = await browser_session.get_browser_state_summary(include_screenshot=False)
				stable_text = stable_state.dom_state.llm_representation() if stable_state.dom_state else ''
				stable_map = stable_state.dom_state.selector_map if stable_state.dom_state else {}
				after_count = visible_item_count(stable_text, stable_map)
				after_indices = {
					item[1]
					for label in ('Remove', 'Remove This Item', 'Delete')
					for item in rank_labeled_elements(label, stable_map)
				}
				if visibly_empty(stable_text) or after_count == 0:
					progress_observed = True
				elif before_count is not None and after_count is not None and after_count < before_count:
					progress_observed = True
			if not progress_observed:
				return ActionResult(
					error=(
						'Visible cart item count did not decrease after the current remove/confirmation action; '
						'stop rather than repeating an unverified mutation.'
					),
					metadata={
						'removed': removed,
						'before_item_count': before_count,
						'after_item_count': after_count,
						'remaining_remove_controls': len(after_indices),
						'confirmation_clicked': confirmation_clicked,
					},
				)
			removed += max(1, before_count - after_count) if before_count is not None and after_count is not None else 1

		return ActionResult(
			error='Cleanup exceeded the bounded limit of 30 existing items.',
			metadata={'removed': removed},
		)

	@tools.action(
		description=(
			'Navigate through a current visible link by accessible label, optionally requiring nearby row, menu, '
			'or section text. Prefer this for hierarchical destinations when a same-named sidebar facet would lose '
			'the requested parent path. If the exact link is inside a collapsed hover menu, this action probes current '
			'visible menu triggers, requires the target to become visibly rendered, and then follows its live href.'
		),
		param_model=NavigateVisibleLinkAction,
	)
	async def navigate_visible_link(
		params: NavigateVisibleLinkAction,
		browser_session,
	) -> ActionResult:
		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		selector_map = state.dom_state.selector_map if state.dom_state else {}
		ranked = (
			rank_record_controls(params.context, params.label, selector_map)
			if params.context.strip()
			else rank_labeled_elements(params.label, selector_map)
		)
		for _, index, label in ranked:
			node = selector_map[index]
			attributes = getattr(node, 'attributes', None) or {}
			href = str(attributes.get('href') or '').strip()
			if not href:
				continue
			current = await browser_session.get_current_page_url()
			target = urljoin(current, href)
			await browser_session.navigate_to(target, new_tab=False)
			return ActionResult(
				extracted_content=f'Navigated through visible link {label!r} to {target}.',
				long_term_memory=f'Used the live href for {label!r}; destination is {target}.',
				metadata={'resolved_index': index, 'resolved_label': label, 'href': target},
			)
		if not params.context.strip():
			value = await reveal_navigation_target(browser_session, params.label)
			if value.get('found') and value.get('href'):
				target = str(value['href'])
				await browser_session.navigate_to(target, new_tab=False)
				return ActionResult(
					extracted_content=(
						f'Revealed {value.get("label")!r} under visible menu '
						f'{value.get("parent")!r} and navigated to {target}.'
					),
					long_term_memory=(
						f'Found exact navigation target {value.get("label")!r} by probing visible menus; '
						f'parent was {value.get("parent")!r}.'
					),
					metadata=value,
				)
			return ActionResult(
				error=(
					f'No exact navigation link {params.label!r} became visible after probing current menu triggers. '
					f'Explored: {value.get("explored", [])}.'
				)
			)
		return ActionResult(error=f'No current visible link with an href matches label {params.label!r}.')

	@tools.action(
		description=(
			'Replace one query parameter in the current URL with the exact value required by the task. Use only when '
			'the parameter is already present in the current URL or in a visible link, such as after the site exposed '
			'a filter parameter but its preset ranges are not equivalent to the requested bound.'
		),
		param_model=SetCurrentUrlQueryAction,
	)
	async def set_current_url_query(
		params: SetCurrentUrlQueryAction,
		browser_session,
	) -> ActionResult:
		current = await browser_session.get_current_page_url()
		parts = urlsplit(current)
		current_pairs = parse_qsl(parts.query, keep_blank_values=True)
		observed = {key for key, _ in current_pairs}
		if params.parameter not in observed:
			cdp_session = await browser_session.get_or_create_cdp_session()
			result = await cdp_session.cdp_client.send.Runtime.evaluate(
				params={
					'expression': (
						"Array.from(document.querySelectorAll('a[href]'))."
						"some(a => { try { return new URL(a.href).searchParams.has("
						+ json.dumps(params.parameter)
						+ "); } catch (_) { return false; } })"
					),
					'returnByValue': True,
				},
				session_id=cdp_session.session_id,
			)
			if not result.get('result', {}).get('value'):
				return ActionResult(
					error=f'Query parameter {params.parameter!r} has not been observed in the current page.'
				)
		updated = [(key, value) for key, value in current_pairs if key != params.parameter]
		updated.append((params.parameter, params.value))
		target = urlunsplit(parts._replace(query=urlencode(updated)))
		await browser_session.navigate_to(target, new_tab=False)
		return ActionResult(
			extracted_content=f'Set observed URL query parameter {params.parameter!r} to {params.value!r}.',
			long_term_memory=f'Navigated to exact visible-site constraint URL {target}.',
			metadata={'url': target, 'parameter': params.parameter, 'value': params.value},
		)

	@tools.action(
		description=(
			'Enter a concise query into the current website search box and submit it immediately. Use this instead '
			'of a separate input followed by waiting for autocomplete. It never uses an external search engine.'
		),
		param_model=SubmitSiteSearchAction,
	)
	async def submit_site_search(
		params: SubmitSiteSearchAction,
		browser_session,
	) -> ActionResult:
		cdp_session = await browser_session.get_or_create_cdp_session()
		script = r"""
(async ({query}) => {
  const visible = (el) => {
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
  };
  const inputs = Array.from(document.querySelectorAll('input, textarea')).filter(visible);
  const ranked = inputs.map((el) => {
    const text = [
      el.type, el.name, el.id, el.placeholder, el.getAttribute('aria-label'), el.getAttribute('role')
    ].filter(Boolean).join(' ').toLowerCase();
    let score = 0;
    if (el.type === 'search' || el.getAttribute('role') === 'searchbox') score += 20;
    if (/(search|query|keyword)/.test(text)) score += 10;
    return {el, score};
  }).filter(item => item.score > 0).sort((a, b) => b.score - a.score);
  if (!ranked.length) return {submitted: false, error: 'No visible site search input found.'};
  const input = ranked[0].el;
  const setter = Object.getOwnPropertyDescriptor(
    input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype,
    'value'
  )?.set;
  setter ? setter.call(input, query) : (input.value = query);
  input.dispatchEvent(new Event('input', {bubbles: true}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
  const form = input.form || input.closest('form');
  if (form) {
    form.requestSubmit ? form.requestSubmit() : form.submit();
  } else {
    input.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', bubbles: true}));
    input.dispatchEvent(new KeyboardEvent('keyup', {key: 'Enter', code: 'Enter', bubbles: true}));
  }
  return {submitted: true, query, label: input.getAttribute('aria-label') || input.placeholder || input.name || ''};
})(%s)
""" % json.dumps({'query': params.query}, ensure_ascii=False)
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': script, 'returnByValue': True, 'awaitPromise': True},
			session_id=cdp_session.session_id,
		)
		value = result.get('result', {}).get('value') or {}
		if not value.get('submitted'):
			return ActionResult(error=str(value.get('error') or 'Site search could not be submitted.'))
		return ActionResult(
			extracted_content=f'Submitted current-site search for {params.query!r}.',
			long_term_memory=f'Submitted site search query {params.query!r}; inspect the resulting page.',
			metadata=value,
		)

	@tools.action(
		description=(
			'Fill multiple visible form controls in one atomic pass using their current labels. Use this for address, '
			'profile, checkout, and other multi-field forms. It dispatches normal input/change events and returns '
			'unmatched fields plus the live values. Inspect the result before clicking the normal submit button.'
		),
		param_model=FillVisibleFormFieldsAction,
	)
	async def fill_visible_form_fields(
		params: FillVisibleFormFieldsAction,
		browser_session,
	) -> ActionResult:
		cdp_session = await browser_session.get_or_create_cdp_session()
		script = r"""
(({fields}) => {
  const norm = (value) => String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const visible = (el) => {
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 && style.display !== 'none' && style.visibility !== 'hidden';
  };
  const controls = Array.from(document.querySelectorAll('input, textarea, select')).filter(visible);
  const descriptorValue = (el, value) => {
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    setter ? setter.call(el, value) : (el.value = value);
  };
  const labels = (el) => {
    const explicit = el.id ? document.querySelector(`label[for="${CSS.escape(el.id)}"]`) : null;
    return [
      explicit?.innerText, el.closest('label')?.innerText, el.getAttribute('aria-label'),
      el.placeholder, el.name, el.id
    ].filter(Boolean).map(norm);
  };
  const filled = [];
  const unmatched = [];
  for (const [requestedLabel, requestedValue] of Object.entries(fields)) {
    const target = norm(requestedLabel);
    const ranked = controls.map((el) => {
      const names = labels(el);
      const best = Math.max(0, ...names.map(name =>
        name === target ? 100 : name.includes(target) || target.includes(name) ? 60 : 0
      ));
      return {el, score: best, names};
    }).filter(item => item.score > 0).sort((a, b) => b.score - a.score);
    if (!ranked.length) {
      unmatched.push(requestedLabel);
      continue;
    }
    const el = ranked[0].el;
    if (el instanceof HTMLSelectElement) {
      const wanted = norm(requestedValue);
      const option = Array.from(el.options).find(opt =>
        norm(opt.text) === wanted || norm(opt.value) === wanted ||
        norm(opt.text).includes(wanted) || wanted.includes(norm(opt.text))
      );
      if (!option) {
        unmatched.push(requestedLabel);
        continue;
      }
      el.value = option.value;
    } else {
      descriptorValue(el, requestedValue);
    }
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
    filled.push({label: requestedLabel, name: el.name || el.id || '', value: el.value});
  }
  return {filled, unmatched};
})(%s)
""" % json.dumps({'fields': params.fields}, ensure_ascii=False)
		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': script, 'returnByValue': True},
			session_id=cdp_session.session_id,
		)
		value = result.get('result', {}).get('value') or {'filled': [], 'unmatched': list(params.fields)}
		return ActionResult(
			extracted_content=json.dumps(value, ensure_ascii=False),
			long_term_memory=(
				f'Filled {len(value.get("filled", []))} form fields; '
				f'unmatched fields: {value.get("unmatched", [])}.'
			),
			metadata=value,
		)

	@tools.action(
		description=(
			'Activate a visible non-commit control by its exact current label. Use this after an index became stale, '
			'or when the same label is exposed by nested elements. The target is resolved again from the current DOM.'
		),
		param_model=ActivateVisibleLabelAction,
	)
	async def activate_visible_label(
		params: ActivateVisibleLabelAction,
		browser_session,
	) -> ActionResult:
		if re.search(
			r'\b(?:buy|checkout|delete|remove|submit|send|confirm|place order|purchase|book|reserve)\b',
			params.label,
			flags=re.IGNORECASE,
		):
			return ActionResult(
				error='Commit-like controls must use the normal indexed click so standard validation remains active.'
			)
		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		selector_map = state.dom_state.selector_map if state.dom_state else {}
		ranked = rank_labeled_elements(params.label, selector_map)
		if not ranked:
			return ActionResult(error=f'No current visible control matches label {params.label!r}.')
		best_score, best_index, best_label = ranked[0]
		ambiguous = [item for item in ranked[1:] if item[0] == best_score and item[2] != best_label]
		if ambiguous:
			choices = [best_label] + [item[2] for item in ambiguous[:4]]
			return ActionResult(
				error=f'Label is ambiguous in the current state. Use a more specific label from: {choices}'
			)
		node = selector_map[best_index]
		event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
		await event
		await event.event_result(raise_if_any=True, raise_if_none=False)
		return ActionResult(
			extracted_content=f'Activated current visible control {best_label!r}.',
			long_term_memory=f'Activated label {best_label!r} after resolving it in the current page state.',
			metadata={'resolved_index': best_index, 'resolved_label': best_label},
		)

	@tools.action(
		description=(
			'Activate a repeated visible control inside the current row, card, list entry, or section identified by '
			'unique visible record text. Use this for controls such as View, Edit, or Open that appear many times.'
		),
		param_model=ActivateRecordControlAction,
	)
	async def activate_record_control(
		params: ActivateRecordControlAction,
		browser_session,
	) -> ActionResult:
		commit_like = bool(re.search(
			r'\b(?:add|buy|checkout|delete|remove|submit|send|confirm|place order|purchase|book|reserve)\b',
			params.control_label,
			flags=re.IGNORECASE,
		))
		state = await browser_session.get_browser_state_summary(include_screenshot=False)
		selector_map = state.dom_state.selector_map if state.dom_state else {}
		ranked = rank_record_controls(params.record_text, params.control_label, selector_map)
		if not ranked:
			return ActionResult(
				error=(
					f'No current {params.control_label!r} control is inside a visible record containing '
					f'{params.record_text!r}.'
				)
			)
		_, index, label = ranked[0]
		node = selector_map[index]
		if commit_like:
			commit_validation = await validate_commit_click(browser_session, node)
			if commit_validation.is_commit and not commit_validation.allowed:
				return ActionResult(error=f'Commit blocked: {commit_validation.reason}')
			mutation_validation = await validate_control_mutation(browser_session, node)
			if not mutation_validation.allowed:
				return ActionResult(error=f'Control mutation blocked: {mutation_validation.reason}')
		event = browser_session.event_bus.dispatch(ClickElementEvent(node=node))
		await event
		await event.event_result(raise_if_any=True, raise_if_none=False)
		return ActionResult(
			extracted_content=(
				f'Activated {label!r} in the visible record containing {params.record_text!r}.'
			),
			long_term_memory=(
				f'Activated control {label!r} after resolving record {params.record_text!r} in the current page.'
			),
			metadata={'resolved_index': index, 'resolved_label': label, 'record_text': params.record_text},
		)
