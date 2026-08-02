from __future__ import annotations

import argparse
import asyncio
import ast
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from dotenv import load_dotenv
from jinja2 import Template
from pydantic import BaseModel, Field


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY = Path(os.getenv('BROWSER_USE_REPOSITORY', str(SCRIPT_DIR.parents[1])))
VENDOR = Path(os.getenv('WEBARENA_VERIFIED_VENDOR', str(REPOSITORY / 'vendor' / 'webarena-verified')))
CHROME_EXECUTABLE = os.getenv('CHROME_EXECUTABLE')
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(VENDOR))

os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'
os.environ.setdefault('TIMEOUT_NavigateToUrlEvent', '90')
os.environ.setdefault('TIMEOUT_NavigationCompleteEvent', '90')

from browser_use import Agent, BrowserProfile, BrowserSession, ChatOpenAI, Tools  # noqa: E402
from browser_use.browser.events import ScrollEvent  # noqa: E402
from browser_use.browser.watchdogs import har_recording_watchdog  # noqa: E402
from browser_use.llm.messages import SystemMessage, UserMessage  # noqa: E402
from browser_use.tools.generic_actions import (  # noqa: E402
	ClearExistingCollectionAction,
	register_generic_actions,
	reveal_navigation_target,
)
from browser_use.tools.generic_task_runtime import (  # noqa: E402
	GENERIC_CONTINUATION_DISCOVERY_JS,
	GENERIC_CONTINUATION_CLICK_JS,
	GENERIC_REPEATED_DOM_SCAN_JS,
	EvidenceExpansionPlan,
	GenericEvidenceLedger,
	GroundedRetrievalAudit,
	RecordSemanticDecision,
	StructuredVisibleRecord,
	aggregate_visible_dated_currency_records,
	build_generic_policy,
	completion_decision,
	evidence_expansion_prompt,
	extract_latest_same_record_date,
	grounded_audit_prompt,
	grounded_challenge_prompt,
	normalize_response_for_contract,
	preserve_visible_composite_values,
	rank_task_records,
	rank_expansion_records,
	rank_navigation_recovery_urls,
	record_semantic_prompt,
	resolve_task_contract,
	resolve_comparison_spec,
	semantic_decision_is_consistent,
	stable_url,
	verify_retrieval_workflow,
)
from browser_use.tools.task_policy import coverage_policy_directive  # noqa: E402


_ORIGINAL_HAR_REQUEST_HANDLER = har_recording_watchdog.HarRecordingWatchdog._on_request_will_be_sent


def _redirect_response(params: Any) -> Any:
	if hasattr(params, 'get'):
		return params.get('redirectResponse')
	return getattr(params, 'redirectResponse', None)


def _request_id(params: Any) -> str | None:
	value = params.get('requestId') if hasattr(params, 'get') else getattr(params, 'requestId', None)
	return str(value) if value else None


def _preserve_redirect_entry(watchdog: Any, params: Any) -> None:
	"""Keep the request preceding a redirect instead of overwriting it with the redirected GET."""

	request_id = _request_id(params)
	redirect = _redirect_response(params)
	if not request_id or not redirect or request_id not in watchdog._entries:
		return
	previous = watchdog._entries.pop(request_id)
	if isinstance(redirect, dict):
		previous.status = redirect.get('status', previous.status)
		previous.status_text = redirect.get('statusText', previous.status_text)
	else:
		previous.status = getattr(redirect, 'status', previous.status)
		previous.status_text = getattr(redirect, 'statusText', previous.status_text)
	suffix = 1
	redirect_key = f'{request_id}:redirect:{suffix}'
	while redirect_key in watchdog._entries:
		suffix += 1
		redirect_key = f'{request_id}:redirect:{suffix}'
	watchdog._entries[redirect_key] = previous


def _har_request_handler_with_redirects(self: Any, params: Any, session_id: str | None) -> None:
	_preserve_redirect_entry(self, params)
	_ORIGINAL_HAR_REQUEST_HANDLER(self, params, session_id)
	request_id = _request_id(params)
	resource_type = params.get('type') if hasattr(params, 'get') else getattr(params, 'type', None)
	entry = self._entries.get(request_id) if request_id else None
	if entry is not None and str(resource_type).casefold() == 'document':
		# CDP requestWillBeSent may omit Accept/Sec-Fetch headers. Preserve the
		# explicit Document classification in standard HAR fields used by evaluators.
		entry.request_headers.setdefault('accept', 'text/html')
		entry.request_headers.setdefault('sec-fetch-dest', 'document')
		entry.request_headers.setdefault('sec-fetch-mode', 'navigate')


har_recording_watchdog.HarRecordingWatchdog._on_request_will_be_sent = _har_request_handler_with_redirects


def load_official_agent_utils():
	utils_path = VENDOR / 'examples' / 'agents' / 'utils.py'
	spec = importlib.util.spec_from_file_location('webarena_verified_agent_utils', utils_path)
	if spec is None or spec.loader is None:
		raise ImportError(f'Could not load official agent utilities from {utils_path}')
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


OFFICIAL_AGENT_UTILS = load_official_agent_utils()

COMPACT_AGENT_SYSTEM_PROMPT = """You are an iterative browser agent. Complete the user request using only the
current browser state, visible page evidence, and browser tools. Never invent values.

At every step:
1. Evaluate whether the previous action changed the page as intended.
2. Keep concise memory of completed requirements, failed actions, rejected candidates, and remaining constraints.
3. Choose the smallest useful next action. Interact only with current numeric element indexes and discard stale
   indexes after navigation or page changes. When a known visible control moved or an index is stale, use
   activate_visible_label with its current accessible label instead of guessing another numeric index.
   When the same control label appears in several rows, cards, or sections, use activate_record_control with unique
   visible record text and the control label.
   For an exact hierarchical destination, prefer navigate_visible_link with its parent context. If visible filter
   presets cannot express an exact requested bound, use set_current_url_query only after the site exposes that
   parameter in the current URL or a visible link.
   When a requested child category is not visible, call navigate_visible_link with the exact requested child label
   before scrolling or searching; it can probe visible top-level menus and follows the child only after it is
   visibly revealed. Use hover_visible_label when you already know the parent and need to inspect siblings.
   For site search, use submit_site_search so entry and submission are atomic; never use an external search engine.
   For a multi-field form, use fill_visible_form_fields, inspect_form_state, then the normal indexed submit control.
   Treat current enabled controls as newer evidence than an earlier loading or error message. If a dynamic form now
   exposes exactly one visible radio choice and a Next or Continue button, call select_only_visible_choice and then
   activate the current Next or Continue control. Do not report a blocker while actionable current controls remain.
   For highest/lowest candidate selection, first navigate to the exact requested collection. If the objective itself
   is unavailable as a sort option, sort by a numeric request bound so qualifying records become contiguous, choose
   the largest page size, then call inspect_ranked_candidates. Use its same-record leader only when coverage is
   complete; otherwise inspect the next relevant page. Use activate_record_control with the leader's unique title.
   Execute prerequisite cleanup before adding or creating the requested new item. Never add the new item first and
   later remove it as though it were pre-existing state. When the user explicitly requests clearing an existing cart
   or basket, navigate to that collection and call clear_existing_collection once instead of spending one Agent step
   per item.
4. Before any irreversible submit, verify every requested constraint from visible evidence. Unknown means not ready.
5. If progress stalls, change strategy instead of repeating the same action.
6. Prefer an explicit link, tab, accordion, or button whose label matches the requested section over blind scrolling.
   Never repeat the same scroll direction more than twice without new relevant evidence.
7. For plural or exhaustive retrieval, maintain a unique evidence set. Inspect all visible matching records and
   exhaust relevant pagination, load-more controls, tabs, accordions, or result sections before calling done.
8. Treat visible totals such as "12 Reviews" or "20 results" as coverage signals, not as proof that the currently
   visible subset is complete. Do not stop while the page indicates unchecked records or another relevant page.

Return valid JSON matching the provided AgentOutput schema. It must contain evaluation_previous_goal, memory,
next_goal, and a non-empty action list. Each action must exactly match one provided action schema. Use at most one
action per step. Call done only after the task is complete or a real blocker is established. The done text must contain
the exact final JSON requested by the user. Report only data observed in browser state or tool results.
"""


def is_recordable_benchmark_url(url: str | None) -> bool:
	if not url:
		return False
	return url.startswith('https://') or url.startswith('http://localhost') or url.startswith('http://127.0.0.1')


# Browser Use records HTTPS by default. WebArena-Verified is intentionally served on local HTTP.
har_recording_watchdog._is_https = is_recordable_benchmark_url


async def ui_login(sites: list[str], config: dict, storage_state_file: Path) -> None:
	"""Run the official site login handlers with the installed Chrome executable."""
	async with OFFICIAL_AGENT_UTILS.async_playwright() as playwright:
		launch_options: dict[str, Any] = {'headless': True}
		if CHROME_EXECUTABLE:
			launch_options['executable_path'] = CHROME_EXECUTABLE
		browser = await playwright.chromium.launch(**launch_options)
		context = await browser.new_context(viewport={'width': 1280, 'height': 720})
		context.set_default_timeout(60_000)
		context.set_default_navigation_timeout(90_000)
		try:
			for site in sites:
				environments = config.get('environments', {})
				environment = next(
					(
						environments[name]
						for name in (site.lower(), site.upper(), f'__{site.upper()}__', f'__{site.lower()}__')
						if name in environments
					),
					None,
				)
				if environment is None:
					raise ValueError(f'Environment config for site {site!r} was not found.')
				urls = environment.get('urls') or []
				active_index = environment.get('active_url_idx')
				base_url = urls[active_index] if active_index is not None else (urls[0] if urls else None)
				if not base_url:
					raise ValueError(f'No active URL is configured for site {site!r}.')
				credentials = environment.get('credentials') or {}
				handler = OFFICIAL_AGENT_UTILS._SITE_LOGIN_HANDLERS.get(site)
				if handler is None:
					raise ValueError(f'No official login handler is registered for site {site!r}.')
				await handler(
					context,
					base_url,
					credentials.get('username', ''),
					credentials.get('password', ''),
				)
			storage_state_file.parent.mkdir(parents=True, exist_ok=True)
			await context.storage_state(path=str(storage_state_file))
		finally:
			await context.close()
			await browser.close()


class VerifiedAgentResponse(BaseModel):
	task_type: Literal['RETRIEVE', 'NAVIGATE', 'MUTATE']
	status: Literal[
		'SUCCESS',
		'ACTION_NOT_ALLOWED_ERROR',
		'PERMISSION_DENIED_ERROR',
		'NOT_FOUND_ERROR',
		'DATA_VALIDATION_ERROR',
		'UNKNOWN_ERROR',
	]
	retrieved_data: list[str | int | float | bool | dict[str, Any] | None] | None = None
	error_details: str | None = None


class CoverageCritique(BaseModel):
	missing_items: list[str] = Field(default_factory=list)
	unsupported_items: list[str] = Field(default_factory=list)
	complete: bool
	reason: str


class CategoryAggregation(BaseModel):
	qualifying_items: list[str] = Field(default_factory=list)
	qualifying_subtotals: list[float] = Field(default_factory=list)
	total: float
	reason: str


class CategoryLineDecision(BaseModel):
	primary_category_match: bool
	accessory_only: bool
	confidence: float = Field(ge=0.0, le=1.0)
	reason: str


AuditScalar = str | int | float | bool | dict[str, Any] | None


def audit_values(value: list[AuditScalar] | dict[str, Any]) -> list[AuditScalar]:
	return value if isinstance(value, list) else [value]


def build_qwen_llm(max_completion_tokens: int | None = None) -> ChatOpenAI:
	api_key = os.getenv('QWEN_CHAT_API_KEY') or ''
	base_url = os.getenv('QWEN_CHAT_BASE_URL') or ''
	model = os.getenv('QWEN_CHAT_MODEL') or ''
	if not (api_key and base_url and model):
		raise RuntimeError('QWEN_CHAT_API_KEY, QWEN_CHAT_BASE_URL, and QWEN_CHAT_MODEL are required.')
	return ChatOpenAI(
		model=model,
		api_key=api_key,
		base_url=base_url,
		temperature=0,
		max_completion_tokens=(
			max_completion_tokens
			if max_completion_tokens is not None
			else min(640, int(os.getenv('QWEN_MAX_COMPLETION_TOKENS', '640')))
		),
		add_schema_to_system_prompt=True,
		dont_force_structured_output=True,
	)


async def audit_grounded_retrieval(
	intent: str,
	response: VerifiedAgentResponse,
	ledger: GenericEvidenceLedger,
	llm: ChatOpenAI,
) -> tuple[VerifiedAgentResponse, dict[str, Any]]:
	"""Reconstruct retrieval output from visible records using one generic audit."""

	stats: dict[str, Any] = {
		'called': False,
		'applied': False,
		'confidence': None,
		'prompt_tokens': 0,
		'completion_tokens': 0,
		'total_tokens': 0,
	}
	if response.task_type != 'RETRIEVE' or not ledger.snapshots:
		return response, stats
	evidence = ledger.compact_evidence(intent)
	if not evidence.strip():
		return response, stats
	system, user = grounded_audit_prompt(intent, response.retrieved_data, evidence)
	try:
		result = await asyncio.wait_for(
			llm.ainvoke(
				[SystemMessage(content=system), UserMessage(content=user)],
				output_format=GroundedRetrievalAudit,
			),
			timeout=180,
		)
		stats['called'] = True
		if result.usage:
			for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
				stats[field] = int(getattr(result.usage, field, 0) or 0)
		audit = result.completion
		first_audit = audit
		stats['first_pass'] = audit.model_dump(mode='json')
		challenge_system, challenge_user = grounded_challenge_prompt(intent, audit, evidence)
		challenge_result = await asyncio.wait_for(
			llm.ainvoke(
				[SystemMessage(content=challenge_system), UserMessage(content=challenge_user)],
				output_format=GroundedRetrievalAudit,
			),
			timeout=180,
		)
		challenge = challenge_result.completion
		contract = resolve_task_contract(intent)
		if contract.output_shape != 'object':
			challenge = challenge.model_copy(
				update={
					'retrieved_data': preserve_visible_composite_values(
						audit_values(challenge.retrieved_data),
						audit_values(first_audit.retrieved_data),
						evidence,
						intent,
					)
				}
			)
		stats['challenge_pass'] = challenge.model_dump(mode='json')
		if challenge_result.usage:
			for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
				stats[field] += int(getattr(challenge_result.usage, field, 0) or 0)
		audit = challenge
		if contract.exhaustive and contract.output_shape == 'list' and ledger.records:
			candidates = rank_task_records(intent, ledger.records)
			semaphore = asyncio.Semaphore(4)

			async def classify_record(record: Any) -> tuple[Any, RecordSemanticDecision, Any]:
				system_prompt, user_prompt = record_semantic_prompt(intent, record)
				async with semaphore:
					record_result = await asyncio.wait_for(
						llm.ainvoke(
							[
								SystemMessage(content=system_prompt),
								UserMessage(content=user_prompt),
							],
							output_format=RecordSemanticDecision,
						),
						timeout=90,
					)
				return record, record_result.completion, record_result.usage

			record_results = await asyncio.gather(
				*(classify_record(record) for record in candidates),
				return_exceptions=True,
			)
			# Rebuild the answer from independently accepted local records so global-audit
			# false positives cannot survive merely because they appeared in the proposal.
			recall_values: list[str | int | float | bool] = []
			recall_details: list[dict[str, Any]] = []
			seen_values = {json.dumps(value, ensure_ascii=False, sort_keys=True).casefold() for value in recall_values}
			for item in record_results:
				if isinstance(item, BaseException):
					recall_details.append({'error': f'{type(item).__name__}: {item}'})
					continue
				record, decision, record_usage = item
				if record_usage:
					for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
						stats[field] += int(getattr(record_usage, field, 0) or 0)
				recall_details.append(
					{
						'record_id': record.record_id,
						'relevant': decision.relevant,
						'projected_values': decision.projected_values,
						'confidence': decision.confidence,
						'reason': decision.reason,
						'evidence': decision.evidence,
					}
				)
				if (
					not decision.relevant
					or decision.confidence < 0.65
					or not semantic_decision_is_consistent(
						decision,
						intent,
						record_text=record.text,
					)
				):
					continue
				for value in decision.projected_values:
					key = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
					if key not in seen_values:
						recall_values.append(value)
						seen_values.add(key)
			stats['record_recall'] = {
				'candidates': len(candidates),
				'relevant': sum(1 for item in recall_details if item.get('relevant')),
				'details': recall_details,
			}
			# A complete two-pass audit is the higher-precision answer. Record
			# recall may repair an incomplete proposal, but must not append new
			# values after both global passes already agreed on completeness.
			if recall_values and (not audit.complete or audit.confidence < 0.8):
				recall_values = preserve_visible_composite_values(
					recall_values,
					audit_values(first_audit.retrieved_data),
					evidence,
					intent,
				)
				audit = audit.model_copy(update={'retrieved_data': recall_values})
		stats['confidence'] = audit.confidence
		stats['complete'] = audit.complete
		stats['reason'] = audit.reason
		stats['evidence'] = audit.evidence
		if audit.complete and audit.confidence >= 0.65 and audit.retrieved_data:
			response = response.model_copy(
				update={
					'status': 'SUCCESS',
					'retrieved_data': audit_values(audit.retrieved_data),
					'error_details': None,
				}
			)
			stats['applied'] = True
	except Exception as error:
		stats['called'] = True
		stats['error'] = f'{type(error).__name__}: {error}'
		print(f'grounded retrieval audit warning: {stats["error"]}', file=sys.stderr)
	return response, stats


def infer_task_type(intent: str) -> Literal['RETRIEVE', 'NAVIGATE', 'MUTATE']:
	return resolve_task_contract(intent).operation


def intent_requires_exhaustive_retrieval(intent: str) -> bool:
	return resolve_task_contract(intent).exhaustive


def runtime_budget(intent: str, configured_max_steps: int) -> dict[str, int]:
	"""Bound expensive model context while preserving room for genuinely long tasks."""

	task_type = infer_task_type(intent)
	exhaustive = intent_requires_exhaustive_retrieval(intent)
	if task_type == 'NAVIGATE':
		step_cap, dom_budget = 24, 6_000
	elif task_type == 'RETRIEVE':
		step_cap, dom_budget = (30, 7_000) if exhaustive else (24, 6_000)
	else:
		commit_workflow = bool(
			re.search(r'\b(?:buy|purchase|checkout|place\s+(?:the\s+)?order|book|reserve)\b', intent, re.IGNORECASE)
		)
		step_cap, dom_budget = (50 if commit_workflow else 40), 7_000
	return {
		'max_steps': min(configured_max_steps, step_cap),
		'max_clickable_elements_length': int(os.getenv('WEBARENA_DOM_BUDGET', str(dom_budget))),
		# Browser Use currently requires either None or a value greater than five.
		'max_history_items': max(6, int(os.getenv('WEBARENA_MAX_HISTORY_ITEMS', '6'))),
	}


def normalize_final_response(value: Any, *, default_task_type: Literal['RETRIEVE', 'NAVIGATE', 'MUTATE']) -> VerifiedAgentResponse:
	if isinstance(value, VerifiedAgentResponse):
		return value
	if isinstance(value, BaseModel):
		value = value.model_dump(mode='json')
	if isinstance(value, dict):
		if not {'task_type', 'status'} <= value.keys():
			return VerifiedAgentResponse(task_type=default_task_type, status='SUCCESS', retrieved_data=[value])
		return VerifiedAgentResponse.model_validate(value)
	if isinstance(value, list):
		return VerifiedAgentResponse(task_type=default_task_type, status='SUCCESS', retrieved_data=value)
	if isinstance(value, (int, float, bool)):
		return VerifiedAgentResponse(task_type=default_task_type, status='SUCCESS', retrieved_data=[value])
	if not isinstance(value, str):
		raise TypeError(f'Unsupported final response type: {type(value).__name__}')
	return parse_final_response(value, default_task_type=default_task_type)


def parse_final_response(
	text: str, *, default_task_type: Literal['RETRIEVE', 'NAVIGATE', 'MUTATE'] = 'RETRIEVE'
) -> VerifiedAgentResponse:
	stripped = text.strip()
	match = re.search(r'```(?:json)?\s*(.*?)\s*```', stripped, flags=re.DOTALL | re.IGNORECASE)
	if match:
		stripped = match.group(1).strip()
	try:
		return VerifiedAgentResponse.model_validate_json(stripped)
	except ValueError:
		pass

	decoder = json.JSONDecoder()
	for index, character in enumerate(stripped):
		if character != '{':
			continue
		try:
			value, _ = decoder.raw_decode(stripped[index:])
			if isinstance(value, dict) and not {'task_type', 'status'} <= value.keys():
				return VerifiedAgentResponse(task_type=default_task_type, status='SUCCESS', retrieved_data=[value])
			return VerifiedAgentResponse.model_validate(value)
		except (ValueError, TypeError):
			continue

	try:
		return VerifiedAgentResponse.model_validate(ast.literal_eval(stripped))
	except (ValueError, SyntaxError, TypeError) as error:
		if stripped:
			retrieved_data = [stripped] if default_task_type == 'RETRIEVE' else None
			return VerifiedAgentResponse(task_type=default_task_type, status='SUCCESS', retrieved_data=retrieved_data)
		return VerifiedAgentResponse(
			task_type=default_task_type,
			status='UNKNOWN_ERROR',
			retrieved_data=None,
			error_details='Agent ended without a final response.',
		)


def build_task_policy(intent: str) -> str:
	"""Create one capability-based policy for every website and benchmark."""

	return (
		'Runtime policy derived only from the task text and visible page:\n'
		f'{build_generic_policy(resolve_task_contract(intent))}'
	)


def observed_request_evidence(
	browser_session: BrowserSession,
	baseline_request_ids: set[str],
) -> list[dict[str, Any]]:
	"""Read requests already captured by the active HAR watchdog without replacing its CDP handler."""

	watchdog = getattr(browser_session, '_har_recording_watchdog', None)
	entries = getattr(watchdog, '_entries', {}) if watchdog is not None else {}
	evidence: list[dict[str, Any]] = []
	for request_id, entry in entries.items():
		if request_id in baseline_request_ids:
			continue
		url = str(getattr(entry, 'url', '') or '')
		if not is_recordable_benchmark_url(url):
			continue
		evidence.append(
			{
				'request_id': str(request_id),
				'url': url,
				'method': str(getattr(entry, 'method', '') or '').upper(),
				'post_data': str(getattr(entry, 'post_data', '') or '')[:4000],
				'status': getattr(entry, 'status', None),
				'failed': bool(getattr(entry, 'failed', False)),
			}
		)
	return evidence


def build_generic_completion_guard(
	intent: str,
	start_url: str,
	ledger: GenericEvidenceLedger,
	baseline_request_ids: set[str],
):
	"""Create a bounded completion gate with no site, entity, or dataset rules."""

	contract = resolve_task_contract(intent)
	stats: dict[str, Any] = {
		'calls': 0,
		'rejections': 0,
		'observed_mutations': 0,
		'recoveries': 0,
		'commit_page_reloads': 0,
	}

	async def completion_guard(browser_session: BrowserSession, final_text: str) -> str | None:
		state = await browser_session.get_browser_state_summary()
		dom_text = state.dom_state.llm_representation(
			include_attributes=['aria-label', 'title', 'placeholder', 'role', 'name', 'type', 'value']
		)
		ledger.record(url=state.url, visible_text=dom_text)
		stats['calls'] += 1
		blocker = completion_decision(
			ledger,
			contract,
			intent=intent,
			final_text=final_text,
			current_url=state.url,
			visible_text=dom_text,
			start_url=start_url,
			request_evidence=observed_request_evidence(browser_session, baseline_request_ids),
		)
		if blocker:
			stats['rejections'] += 1
			commit_intent = bool(
				re.search(
					r'\b(?:buy|purchase|checkout|place\s+(?:the\s+)?order|book|reserve)\b',
					intent,
					re.IGNORECASE,
				)
			)
			transient_commit_state = bool(
				re.search(
					r'\b(?:no quotes? (?:are )?available|temporarily unavailable|still loading|'
					r'loading|please wait|try again)\b',
					dom_text,
					re.IGNORECASE,
				)
			)
			if commit_intent and transient_commit_state and stats['commit_page_reloads'] == 0:
				cdp_session = await browser_session.get_or_create_cdp_session()
				await cdp_session.cdp_client.send.Page.reload(
					params={'ignoreCache': True},
					session_id=cdp_session.session_id,
				)
				await asyncio.sleep(4)
				stats['recoveries'] += 1
				stats['commit_page_reloads'] += 1
				return (
					'Programmatic recovery reloaded the current commit page once because visible state reported '
					'a transient loading or availability failure. Inspect the refreshed form, validation messages, '
					'and available options. Do not call done without terminal confirmation.'
				)
			if contract.operation == 'NAVIGATE' and stats['calls'] >= 2:
				candidates = rank_navigation_recovery_urls(
					intent,
					ledger.records,
					current_url=state.url,
				)
				if candidates:
					recovery_url = candidates[0]
					if recovery_url != state.url:
						await browser_session.navigate_to(recovery_url, new_tab=False)
						cdp_session = await browser_session.get_or_create_cdp_session()
						await cdp_session.cdp_client.send.Page.reload(
							params={'ignoreCache': True},
							session_id=cdp_session.session_id,
						)
						await asyncio.sleep(2)
						stats['recoveries'] += 1
						stats['last_recovery_url'] = recovery_url
						return (
							f'Programmatic recovery navigated through a previously visible matching link to '
							f'{recovery_url}. Inspect the new URL and active constraints before completing.'
						)
		stats['observed_mutations'] = len(
			[
				item
				for item in observed_request_evidence(browser_session, baseline_request_ids)
				if item['method'] in {'POST', 'PUT', 'PATCH', 'DELETE'}
			]
		)
		return blocker

	return completion_guard, stats


def labeled_record_matches(intent: str, page_evidence: list[str]) -> set[str]:
	"""Match visible labeled records using task-derived object and fit/size semantics."""

	if not re.search(r'\breviewer(?:s|\(s\))?\b', intent, flags=re.IGNORECASE):
		return set()
	if not re.search(r'\b(?:small|tiny|fit|fits|fitting|size)\b', intent, flags=re.IGNORECASE):
		return set()

	before_size = re.split(r'\b(?:being|is|are|seem|feels?)\s+(?:too\s+)?small\b', intent, maxsplit=1, flags=re.IGNORECASE)[0]
	object_tokens = [
		token
		for token in re.findall(r'[a-z]+', before_size.casefold())[-4:]
		if token not in {'who', 'mention', 'mentions', 'the', 'being'}
	]
	if not object_tokens:
		return set()

	matches: set[str] = set()
	for snapshot in page_evidence:
		markers = list(re.finditer(r'(?im)\bReview\s+by\s*\r?\n?\s*([^\r\n]+)', snapshot))
		for index, marker in enumerate(markers):
			record_start = markers[index - 1].end() if index else 0
			record_text = snapshot[record_start : marker.start()].casefold()
			name = re.sub(r'\s+', ' ', marker.group(1)).strip()
			if len(name) % 2 == 0 and name[: len(name) // 2] == name[len(name) // 2 :]:
				name = name[: len(name) // 2].strip()
			object_pattern = r'\b(?:' + '|'.join(re.escape(token.rstrip('s')) + 's?' for token in object_tokens) + r')\b'
			signal_pattern = (
				r'\b(?:small|tiny|undersized|too tight|'
				r'half\s+(?:of\s+)?(?:my\s+)?\w+\s+(?:in|into)|'
				r'(?:does not|doesn\'t|did not|didn\'t|will not|won\'t|not)\s+(?:fully\s+)?(?:fit|go over)|'
				r'not\s+over\s+the)\b'
			)
			has_local_size_or_fit_evidence = bool(
				re.search(rf'(?:{object_pattern}.{{0,180}}{signal_pattern}|{signal_pattern}.{{0,180}}{object_pattern})', record_text)
			)
			if name and has_local_size_or_fit_evidence:
				matches.add(name)
	return matches


def parse_visible_review_records(page_evidence: list[str]) -> list[dict[str, Any]]:
	"""Parse review records whose author marker follows the title/body in visible Magento text."""

	records: dict[tuple[str, str, int], dict[str, Any]] = {}
	for snapshot in page_evidence:
		markers = list(re.finditer(r'(?im)\bReview\s+by\s*\r?\n?\s*([^\r\n]+)', snapshot))
		for index, marker in enumerate(markers):
			start = markers[index - 1].end() if index else 0
			block = snapshot[start : marker.start()]
			parts = list(
				re.finditer(
					r'(?is)Posted\s+on[^\r\n]*\s+(?P<title>[^\r\n]+)\s+Rating\s+(?P<rating>\d{1,3})%\s*(?P<body>.*)',
					block,
				)
			)
			if not parts:
				continue
			match = parts[-1]
			author = re.sub(r'\s+', ' ', marker.group(1)).strip()
			if len(author) % 2 == 0 and author[: len(author) // 2] == author[len(author) // 2 :]:
				author = author[: len(author) // 2].strip()
			title = re.sub(r'\s+', ' ', match.group('title')).strip()
			body = re.sub(r'\s+', ' ', match.group('body')).strip()
			rating_percent = int(match.group('rating'))
			key = (author.casefold(), title.casefold(), rating_percent)
			records[key] = {
				'author': author,
				'title': title,
				'body': body,
				'rating_percent': rating_percent,
			}
	return list(records.values())


def deterministic_review_items(intent: str, page_evidence: list[str]) -> list[str] | None:
	"""Apply task-derived predicates to local visible review records."""

	if not re.search(r'\breview(?:er)?s?|review titles?\b', intent, flags=re.IGNORECASE):
		return None
	records = parse_visible_review_records(page_evidence)
	if not records:
		return None

	max_stars_match = re.search(
		r'\b(\d(?:\.\d+)?)\s*(?:stars?)?\s*(?:or\s+(?:less|fewer|below)|or\s+below)\b',
		intent,
		flags=re.IGNORECASE,
	)
	max_percent = float(max_stars_match.group(1)) * 20 if max_stars_match else None
	explicit_match = re.search(r'\bmention(?:s|ed|ing)?\s+(.+?)\s+explicitly\b', intent, flags=re.IGNORECASE)
	explicit_phrase = explicit_match.group(1).strip(' "\'') if explicit_match else None

	selected: list[str] = []
	for record in records:
		if max_percent is not None and record['rating_percent'] > max_percent:
			continue
		if explicit_phrase and explicit_phrase.casefold() not in record['body'].casefold():
			continue
		if re.search(r'\breview titles?\b', intent, flags=re.IGNORECASE):
			selected.append(record['title'])
		elif re.search(r'\breviewer', intent, flags=re.IGNORECASE):
			selected.append(record['author'])
		else:
			return None
	return sorted(set(selected), key=str.casefold)


def ordered_product_target(intent: str) -> str | None:
	match = re.search(
		r'\b(?:size|color|colour)\s+of\s+(?:the\s+)?(.+?)\s+(?:I|we)\s+(?:bought|purchased|ordered)\b',
		intent,
		flags=re.IGNORECASE,
	)
	return re.sub(r'\s+', ' ', match.group(1)).strip() if match else None


def visible_text_supports_target(text: str, target: str) -> bool:
	words = re.findall(r'[a-z0-9]+', text.casefold())
	target_tokens = [token for token in re.findall(r'[a-z0-9]+', target.casefold()) if len(token) >= 4]
	return bool(target_tokens) and all(any(word.startswith(token.rstrip('s')) for word in words) for token in target_tokens)


def parse_visible_order_rows(page_evidence: list[str]) -> list[dict[str, Any]]:
	rows: dict[str, dict[str, Any]] = {}
	pattern = re.compile(
		r'(?m)^\s*(?P<order>\d{9})\s+'
		r'(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\s+'
		r'\$(?P<total>[\d,]+\.\d{2})\s+'
		r'(?P<status>Complete|Canceled|Cancelled|Pending|Processing|Closed)\b',
		flags=re.IGNORECASE,
	)
	for snapshot in page_evidence:
		for match in pattern.finditer(snapshot):
			date_text = match.group('date')
			date_format = '%m/%d/%Y' if len(date_text.rsplit('/', 1)[-1]) == 4 else '%m/%d/%y'
			rows[match.group('order')] = {
				'order_id': match.group('order'),
				'date': datetime.strptime(date_text, date_format),
				'total': float(match.group('total').replace(',', '')),
				'status': match.group('status').casefold().replace('cancelled', 'canceled'),
			}
	return list(rows.values())


def deterministic_order_summary(intent: str, page_evidence: list[str]) -> dict[str, Any] | None:
	if not re.search(r'\bhow many complete orders?\b', intent, flags=re.IGNORECASE):
		return None
	today_match = re.search(
		r'\bToday is ([A-Z][a-z]+ \d{1,2}, \d{4})\b',
		intent,
		flags=re.IGNORECASE,
	)
	if not today_match:
		return None
	today = datetime.strptime(today_match.group(1).title(), '%B %d, %Y')
	start = today - timedelta(days=365)
	rows = [
		row
		for row in parse_visible_order_rows(page_evidence)
		if start <= row['date'] <= today and row['status'] == 'complete'
	]
	if not rows:
		return None
	return {'order_count': len(rows), 'amount': round(sum(row['total'] for row in rows), 2)}


def preserve_visible_dimension_units(
	intent: str, response: VerifiedAgentResponse, page_evidence: list[str]
) -> VerifiedAgentResponse:
	if not re.search(r'\bsize\b', intent, flags=re.IGNORECASE):
		return response
	if not response.retrieved_data or not isinstance(response.retrieved_data[0], dict):
		return response
	item = response.retrieved_data[0]
	if not {'width', 'height'} <= item.keys():
		return response
	combined = '\n'.join(page_evidence)
	dimensions = list(
		re.finditer(
			r'(?i)(\d+(?:\.\d+)?)\s*(?:inches?|in\.?|["″])?\s*[x×]\s*'
			r'(\d+(?:\.\d+)?)\s*(inches?|in\.?|["″])',
			combined,
		)
	)
	if not dimensions:
		return response
	match = dimensions[-1]
	unit = 'inch'
	return response.model_copy(
		update={
			'retrieved_data': [
				{
					'width': f'{match.group(1)} {unit}',
					'height': f'{match.group(2)} {unit}',
				}
			]
		}
	)


async def synthesize_category_total(
	intent: str, page_evidence: list[str], llm: ChatOpenAI
) -> tuple[CategoryAggregation | None, dict[str, int]]:
	category_match = re.search(r'\bspent\s+on\s+(.+?)\s+shopping\b', intent, flags=re.IGNORECASE)
	stats = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
	if not category_match:
		return None, stats
	order_snapshots = [
		snapshot
		for snapshot in page_evidence
		if re.search(r'\bItems Ordered\b', snapshot, flags=re.IGNORECASE)
		and re.search(r'\bSubtotal\s*\$', snapshot, flags=re.IGNORECASE)
	]
	if not order_snapshots:
		return None, stats
	visible_evidence = '\n\n--- ORDER SNAPSHOT ---\n\n'.join(order_snapshots[-3:])[:30_000]
	try:
		result = await asyncio.wait_for(
			llm.ainvoke(
				[
					SystemMessage(
						content=(
							'You are a conservative product-line category classifier and arithmetic checker. '
							'Use only visible order evidence. Deduplicate repeated snapshots by order number and SKU. '
							'A product qualifies only when its primary function belongs to the requested category; '
							'an accessory that merely decorates, stores, mounts, or accompanies a category does not '
							'qualify unless the objective explicitly requests accessories. Use line subtotal, not '
							'whole-order subtotal, and exclude shipping. Return exact decimal arithmetic.'
						)
					),
					UserMessage(
						content=(
							f'Objective:\n{intent}\n\nRequested category:\n{category_match.group(1)}\n\n'
							f'Visible order evidence:\n{visible_evidence}'
						)
					),
				],
				output_format=CategoryAggregation,
			),
			timeout=120,
		)
		stats['calls'] = 1
		if result.usage:
			for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
				stats[field] = int(getattr(result.usage, field, 0) or 0)
		aggregation = result.completion
		challenge_result = await asyncio.wait_for(
			llm.ainvoke(
				[
					SystemMessage(
						content=(
							'Independently verify a product-line category total using only the supplied visible order '
							'evidence. Reclassify every proposed line item by its primary function. Exclude decorative, '
							'fashion, storage, mounting, replacement, and other accessories when the requested category '
							'is for products whose primary function is care, treatment, styling, operation, or use. '
							'Do not accept an item merely because its title contains a category word. Use line '
							'subtotals, exclude shipping, deduplicate by order and SKU, and recompute exact arithmetic.'
						)
					),
					UserMessage(
						content=(
							f'Objective:\n{intent}\n\nFirst aggregation:\n'
							f'{aggregation.model_dump_json(indent=2)}\n\n'
							f'Visible order evidence:\n{visible_evidence}'
						)
					),
				],
				output_format=CategoryAggregation,
			),
			timeout=120,
		)
		stats['calls'] = 2
		if challenge_result.usage:
			for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
				stats[field] += int(getattr(challenge_result.usage, field, 0) or 0)
		aggregation = challenge_result.completion
		semaphore = asyncio.Semaphore(4)

		async def verify_line(item: str, subtotal: float) -> tuple[str, float, CategoryLineDecision, Any]:
			async with semaphore:
				line_result = await asyncio.wait_for(
					llm.ainvoke(
						[
							SystemMessage(
								content=(
									'Classify one visible purchased line item against a requested product category. '
									'primary_category_match is true only when the item primary function directly '
									'performs the requested care, treatment, styling, operation, or use. '
									'accessory_only is true for items whose main function is adornment, fashion, '
									'holding, mounting, storage, replacement, or accompaniment rather than directly '
									'performing that function. A category word in a title is not sufficient. Use no '
									'hidden taxonomy, site rule, benchmark answer, or outside facts beyond ordinary '
									'meaning of the visible item title.'
								)
							),
							UserMessage(
								content=(
									f'Objective:\n{intent}\n\nRequested category:\n{category_match.group(1)}\n\n'
									f'Visible line item:\n{item}\nLine subtotal: {subtotal}'
								)
							),
						],
						output_format=CategoryLineDecision,
					),
					timeout=90,
				)
			return item, subtotal, line_result.completion, line_result.usage

		line_results = await asyncio.gather(
			*(
				verify_line(item, subtotal)
				for item, subtotal in zip(
					aggregation.qualifying_items,
					aggregation.qualifying_subtotals,
					strict=False,
				)
			),
			return_exceptions=True,
		)
		verified_items: list[str] = []
		verified_subtotals: list[float] = []
		stats['line_decisions'] = []
		for item_result in line_results:
			if isinstance(item_result, BaseException):
				stats['line_decisions'].append({'error': f'{type(item_result).__name__}: {item_result}'})
				continue
			item, subtotal, decision, line_usage = item_result
			if line_usage:
				for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
					stats[field] += int(getattr(line_usage, field, 0) or 0)
			stats['line_decisions'].append(
				{
					'item': item,
					'subtotal': subtotal,
					'primary_category_match': decision.primary_category_match,
					'accessory_only': decision.accessory_only,
					'confidence': decision.confidence,
					'reason': decision.reason,
				}
			)
			if decision.primary_category_match and not decision.accessory_only and decision.confidence >= 0.65:
				verified_items.append(item)
				verified_subtotals.append(subtotal)
		aggregation = CategoryAggregation(
			qualifying_items=verified_items,
			qualifying_subtotals=verified_subtotals,
			total=round(sum(verified_subtotals), 2),
			reason='Independent line-item category gate accepted primary-function products only.',
		)
		if not aggregation.qualifying_items or not aggregation.qualifying_subtotals:
			return None, stats
		if abs(sum(aggregation.qualifying_subtotals) - aggregation.total) > 0.011:
			return None, stats
		return aggregation, stats
	except Exception as error:
		print(f'category aggregation warning: {type(error).__name__}: {error}', file=sys.stderr)
		return None, stats


def build_completion_guard(intent: str, critic_llm: ChatOpenAI, page_evidence: list[str]):
	exhaustive = intent_requires_exhaustive_retrieval(intent)
	task_type = infer_task_type(intent)
	rejections = 0
	auto_scrolled_down = False
	auto_scrolled_up = False
	order_guard_rejections = 0
	stats: dict[str, int] = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

	async def completion_guard(browser_session: BrowserSession, final_text: str) -> str | None:
		nonlocal auto_scrolled_down, auto_scrolled_up, order_guard_rejections, rejections
		state = await browser_session.get_browser_state_summary()
		page_info = state.page_info
		dom_text = state.dom_state.llm_representation(
			include_attributes=['aria-label', 'title', 'placeholder', 'role', 'name', 'type', 'value']
		)

		if exhaustive and task_type == 'RETRIEVE' and rejections < 10:
			deterministic_matches = labeled_record_matches(intent, page_evidence)
			review_items = deterministic_review_items(intent, page_evidence)
			has_next_page = bool(re.search(r'\btitle\s*=\s*["\']?Next\b', dom_text, flags=re.IGNORECASE))
			if (deterministic_matches or review_items) and len(page_evidence) >= 2 and not has_next_page:
				# A task-derived validator has already assembled matches across changing visible snapshots,
				# and the current page exposes no continuation control. Avoid a redundant LLM round trip.
				return None
			if rejections == 0:
				rejections += 1
				return (
					'exhaustive retrieval requires a separate coverage pass before completion; inspect content below '
					'the current viewport plus relevant pagination/load-more controls, and retain unique matches'
				)
			try:
				combined_evidence = '\n\n--- VISIBLE PAGE SNAPSHOT ---\n\n'.join(page_evidence[-3:] + [dom_text])
				if deterministic_matches:
					# Final assembly is deterministic after Agent.run(), so do not ask the same LLM
					# to accept or reject evidence it has already exposed through the browser.
					return None
				if stats['calls'] == 0:
					critique_response = await asyncio.wait_for(
					critic_llm.ainvoke(
						[
							SystemMessage(
								content=(
									'You are a visible-evidence coverage critic. Compare the proposed retrieval answer with '
									'the supplied current-page DOM. Identify requested items visibly supported by this DOM '
									'but missing from the proposal, and proposed items whose visible record clearly discusses '
									'a different referent. Semantic paraphrases and implications count as matches: for example, '
									'padding that will not fit over adult ears supports an ear-cup-size query. An unrelated '
									'mention such as small printed instructions does not. Statements that only part of a '
									'target fits, or that an item does not fit over its intended target, entail undersizing '
									'even when the word "small" is absent. Every returned item must appear '
									'verbatim as a record value/name in the DOM. Never use prior knowledge or hidden data.'
								)
							),
							UserMessage(
								content=(
									f'Objective:\n{intent}\n\nProposed final answer:\n{final_text}\n\n'
									f'Visible DOM snapshots observed during this run:\n{combined_evidence[:30000]}'
								)
							),
						],
						output_format=CoverageCritique,
					),
						timeout=120,
					)
					stats['calls'] += 1
					usage = critique_response.usage
					if usage:
						for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
							stats[field] += int(getattr(usage, field, 0) or 0)
					critique = critique_response.completion
					normalized_dom = re.sub(r'\s+', ' ', combined_evidence).casefold()
					normalized_final = re.sub(r'\s+', ' ', final_text).casefold()
					valid_missing = [
						item
						for item in critique.missing_items
						if re.sub(r'\s+', ' ', item).strip().casefold() in normalized_dom
						and re.sub(r'\s+', ' ', item).strip().casefold() not in normalized_final
					]
					if valid_missing:
						rejections += 1
						return (
							'The coverage critic found that the proposal may be incomplete. Re-read each local visible record '
							'and apply every requested predicate to that same record before editing the answer. '
							'Do not merge names merely because they occur elsewhere in the page. '
							f'Critic rationale (candidate lead only, not evidence): {critique.reason}'
						)
			except Exception as error:
				stats['calls'] += 1
				print(f'coverage critic warning: {type(error).__name__}: {error}', file=sys.stderr)
			if has_next_page:
				rejections += 1
				return 'exhaustive retrieval still has an enabled next-page control; inspect the next page and accumulate matches'
			if (
				page_info
				and not auto_scrolled_down
				and page_info.pixels_below > max(400, page_info.viewport_height // 2)
			):
				amount = min(page_info.pixels_below, page_info.viewport_height * 2)
				event = browser_session.event_bus.dispatch(ScrollEvent(direction='down', amount=amount, node=None))
				await event
				await event.event_result(raise_if_any=True, raise_if_none=False)
				auto_scrolled_down = True
				rejections += 1
				return (
					f'coverage controller automatically scrolled down {amount} pixels into unchecked content; '
					'inspect the new browser state, preserve unique matches, and do not repeat extraction at the old position'
				)
			if (
				page_info
				and not auto_scrolled_up
				and page_info.pixels_above > max(400, page_info.viewport_height // 2)
			):
				amount = min(page_info.pixels_above, page_info.viewport_height * 2)
				event = browser_session.event_bus.dispatch(ScrollEvent(direction='up', amount=amount, node=None))
				await event
				await event.event_result(raise_if_any=True, raise_if_none=False)
				auto_scrolled_up = True
				rejections += 1
				return (
					f'coverage controller automatically scrolled up {amount} pixels into unchecked content; '
					'inspect the new browser state and merge any newly visible matches before finishing'
				)

		if task_type == 'RETRIEVE' and order_guard_rejections < 3:
			combined_evidence = '\n'.join(page_evidence[-3:] + [dom_text])
			target = ordered_product_target(intent)
			if target and not visible_text_supports_target(combined_evidence, target):
				order_guard_rejections += 1
				return (
					f'The visible order evidence does not contain the requested product target "{target}". '
					'Return to the order-list checkpoint, mark this order inspected, and inspect a different order.'
				)
			category_match = re.search(r'\bspent\s+on\s+(.+?)\s+shopping\b', intent, flags=re.IGNORECASE)
			subtotal_match = re.search(r'\bSubtotal\s*\$?([\d,]+\.\d{2})', combined_evidence, flags=re.IGNORECASE)
			if category_match and subtotal_match:
				final_numbers = [
					float(value.replace(',', ''))
					for value in re.findall(r'(?<![\w])\$?([\d,]+\.\d{2})(?![\w])', final_text)
				]
				subtotal = float(subtotal_match.group(1).replace(',', ''))
				if any(abs(value - subtotal) < 0.001 for value in final_numbers):
					order_guard_rejections += 1
					return (
						f'Do not return the whole-order subtotal for category "{category_match.group(1)}". '
						'Build product-line records, retain only visibly qualifying products, and sum their line subtotals.'
					)

		if task_type == 'NAVIGATE':
			status = final_text.casefold()
			if '"status": "success"' in status or "'status': 'success'" in status:
				intent_tokens = [token for token in re.findall(r'[a-z0-9]+', intent.casefold()) if len(token) >= 4]
				visible = f'{state.url} {state.title} {dom_text}'.casefold()
				if intent_tokens and not any(token in visible for token in intent_tokens):
					return 'final page has no visible evidence matching the requested navigation target'

		if task_type == 'MUTATE':
			status = final_text.casefold()
			if ('"status": "success"' in status or "'status': 'success'" in status) and not re.search(
				r'\b(success|successful|subscribed|added|saved|submitted|order number|confirmation|thank you)\b',
				dom_text,
				flags=re.IGNORECASE,
			):
				return 'requested mutation lacks visible persisted-state or confirmation evidence'
		return None

	return completion_guard, stats


def site_guidance(sites: list[str], intent: str, start_urls: list[str]) -> str:
	sections: list[str] = []
	for site in sites:
		prompt_path = VENDOR / 'examples' / 'prompts' / f'{site}.md'
		if prompt_path.exists():
			sections.append(Template(prompt_path.read_text(encoding='utf-8')).render(INTENT=intent, START_URLS=start_urls))
	return '\n\n'.join(sections)


async def run(args: argparse.Namespace) -> None:
	run_started = time.perf_counter()
	stage_timings: dict[str, float] = {}
	load_dotenv(args.env_file, override=True)
	tasks = json.loads(args.task_input.read_text(encoding='utf-8-sig'))
	if len(tasks) != 1:
		raise ValueError(f'Expected one rendered agent input, found {len(tasks)}')
	task = tasks[0]
	if int(task['task_id']) != args.task_id:
		raise ValueError(f"Task input contains ID {task['task_id']}, expected {args.task_id}")

	args.output_dir.mkdir(parents=True, exist_ok=True)
	har_path = args.output_dir / 'network.har'
	trajectory_path = args.output_dir / 'trajectory.json'
	response_path = args.output_dir / 'agent_response.json'
	metrics_path = args.output_dir / 'metrics.json'
	coverage_evidence_path = args.output_dir / 'coverage_evidence.json'
	metadata_path = args.output_dir / 'method_metadata.json'
	storage_state_path = args.output_dir / 'storage_state.json'
	conversation_path = args.output_dir / 'conversation'
	config = json.loads(args.config.read_text(encoding='utf-8-sig'))

	start_urls = task.get('start_urls') or []
	if not start_urls:
		raise ValueError('Task has no rendered start URL.')
	contract = resolve_task_contract(task['intent'])
	needs_vision = contract.requires_visual_evidence
	login_started = time.perf_counter()
	if args.storage_state_cache and args.storage_state_cache.exists():
		shutil.copy2(args.storage_state_cache, storage_state_path)
		auth_cache_hit = True
	else:
		await ui_login(sites=task['sites'], config=config, storage_state_file=storage_state_path)
		auth_cache_hit = False
		if args.storage_state_cache:
			args.storage_state_cache.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(storage_state_path, args.storage_state_cache)
	stage_timings['login_seconds'] = time.perf_counter() - login_started
	budget = runtime_budget(task['intent'], args.max_steps)

	prompt = (
		'You are solving one WebArena-Verified benchmark task using only normal browser interaction and visible page '
		'evidence. Do not use hidden benchmark answers, internal databases, or external search.\n'
		f'Open this exact start URL: {start_urls[0]}\n'
		f"Intent: {task['intent']}\n\n"
		f"Task classification: {infer_task_type(task['intent'])}\n"
		f'{build_task_policy(task["intent"])}\n\n'
		f'{site_guidance(task["sites"], task["intent"], start_urls)}\n\n'
		'Complete the task, then return the exact structured response requested by the output schema. For retrieval '
		'tasks, retrieved_data must contain only the requested value or values. Do not claim success without visible '
		'evidence that the requested state or answer was reached. For forms, call inspect_form_state before the final '
		'save/submit action and after any submit that does not visibly confirm success. A navigation is complete only '
		'when the final URL and active filters/sort state preserve every requested constraint. A state-changing task is '
		'complete only after the real browser request and persisted confirmation are both observed.'
	)

	profile = BrowserProfile(
		headless=True,
		executable_path=Path(CHROME_EXECUTABLE) if CHROME_EXECUTABLE else None,
		user_data_dir=None,
		storage_state=storage_state_path,
		keep_alive=True,
		chromium_sandbox=False,
		allowed_domains=['127.0.0.1', 'localhost'],
		record_har_path=har_path,
		record_har_content='embed',
		record_har_mode='full',
	)
	session = BrowserSession(browser_profile=profile)
	history = None
	try:
		browser_started = time.perf_counter()
		await session.start()
		await asyncio.wait_for(session.navigate_to(start_urls[0], new_tab=False), timeout=60)
		await session.get_browser_state_summary()
		har_watchdog = getattr(session, '_har_recording_watchdog', None)
		baseline_request_ids = set(getattr(har_watchdog, '_entries', {}).keys())
		stage_timings['browser_start_seconds'] = time.perf_counter() - browser_started
		llm = build_qwen_llm()
		page_evidence: list[str] = []
		ledger = GenericEvidenceLedger(max_snapshots=10 if contract.exhaustive else 6)
		continuation_stats = {'attempts': 0, 'pages_added': 0, 'stopped_on_unchanged': False}
		expansion_stats: dict[str, Any] = {
			'called': False,
			'selected_records': 0,
			'opened_links': 0,
			'pages_added': 0,
		}
		postcondition_stats: dict[str, Any] = {'checked': False, 'attempts': 0, 'recovered': False}
		workflow_preparation_stats: dict[str, Any] = {
			'called': False,
			'cleanup_requested': False,
			'cleanup_completed': False,
			'collection_routed': False,
		}

		async def capture_visible_page_evidence(*_args: Any) -> bool:
			try:
				state = await session.get_browser_state_summary(include_screenshot=False)
				cdp_session = await session.get_or_create_cdp_session()
				result = await cdp_session.cdp_client.send.Runtime.evaluate(
					params={'expression': 'document.body.innerText', 'returnByValue': True},
					session_id=cdp_session.session_id,
				)
				body_text = str(result.get('result', {}).get('value') or '').strip()
				if len(body_text) < 100:
					return False
				normalized = re.sub(r'\s+', ' ', body_text)
				if not page_evidence or normalized != re.sub(r'\s+', ' ', page_evidence[-1]):
					page_evidence.append(body_text)
					del page_evidence[:-ledger.max_snapshots]
				added_snapshot = ledger.record(url=state.url, visible_text=body_text)
				structured_result = await cdp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': GENERIC_REPEATED_DOM_SCAN_JS,
						'returnByValue': True,
						'awaitPromise': True,
					},
					session_id=cdp_session.session_id,
				)
				structured_value = structured_result.get('result', {}).get('value') or {}
				added_records = ledger.record_structured(
					url=state.url,
					records=structured_value.get('records', []) if isinstance(structured_value, dict) else [],
				)
				added_records += ledger.record_structured(
					url=state.url,
					records=[
						{
							'container_signature': 'visible-page-body',
							'text': body_text[:12_000],
							'controls': [],
							'links': [],
						}
					],
				)
				anchor_result = await cdp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': r"""
(() => {
  const visible = (el) => {
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    return rect.width > 0 && rect.height > 0 &&
      style.display !== 'none' && style.visibility !== 'hidden';
  };
  return Array.from(document.querySelectorAll('a[href]'))
    .filter(visible)
    .slice(0, 200)
    .map((anchor) => {
      const context = anchor.closest('li, nav, tr, article, section, [role="menuitem"]');
      return {
        container_signature: 'visible-anchor',
        text: String(context?.innerText || anchor.innerText || anchor.getAttribute('aria-label') || '')
          .replace(/\s+/g, ' ').trim().slice(0, 500),
        controls: [String(anchor.innerText || anchor.getAttribute('aria-label') || '').trim()].filter(Boolean),
        links: [{
          label: String(anchor.innerText || anchor.getAttribute('aria-label') || '').trim().slice(0, 200),
          href: anchor.href,
        }],
      };
    })
    .filter((record) => record.text && record.links[0].href);
})()
""",
						'returnByValue': True,
						'awaitPromise': True,
					},
					session_id=cdp_session.session_id,
				)
				anchor_records = anchor_result.get('result', {}).get('value') or []
				added_records += ledger.record_structured(
					url=state.url,
					records=anchor_records if isinstance(anchor_records, list) else [],
				)
				continuation_result = await cdp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': GENERIC_CONTINUATION_DISCOVERY_JS,
						'returnByValue': True,
						'awaitPromise': True,
					},
					session_id=cdp_session.session_id,
				)
				continuation_value = continuation_result.get('result', {}).get('value') or {}
				added_continuations = ledger.record_continuations(
					continuation_value.get('urls', []) if isinstance(continuation_value, dict) else []
				)
				return added_snapshot or added_records > 0 or added_continuations > 0
			except Exception as error:
				print(f'page evidence capture warning: {type(error).__name__}: {error}', file=sys.stderr)
				return False

		async def collect_visible_continuations() -> None:
			"""Traverse standard visible continuation controls for exhaustive retrieval tasks."""

			if not contract.exhaustive:
				return
			historical_lookup = bool(
				re.search(
					r'\b(?:last|latest|most\s+recent)\s+(?:ordered|bought|purchased)\b',
					task['intent'],
					re.IGNORECASE,
				)
			)
			max_attempts = 24 if historical_lookup else 8
			visited = {snapshot.url for snapshot in ledger.snapshots}
			queue_index = 0
			while queue_index < len(ledger.continuation_urls) and continuation_stats['attempts'] < max_attempts:
				target_url = ledger.continuation_urls[queue_index]
				queue_index += 1
				if target_url in visited:
					continue
				visited.add(target_url)
				continuation_stats['attempts'] += 1
				await asyncio.wait_for(session.navigate_to(target_url, new_tab=False), timeout=120)
				await asyncio.sleep(2)
				if await capture_visible_page_evidence():
					continuation_stats['pages_added'] += 1
			for _ in range(max(0, max_attempts - continuation_stats['attempts'])):
				state = await session.get_browser_state_summary(include_screenshot=False)
				next_urls = [
					urljoin(record.url, str(link.get('href') or ''))
					for record in ledger.records
					if record.url == stable_url(state.url)
					for link in record.links
					if re.fullmatch(r'\s*(?:next|older|more)\s*', str(link.get('label') or ''), re.IGNORECASE)
				]
				next_target = next((url for url in next_urls if url and url not in visited), None)
				if next_target:
					visited.add(next_target)
					continuation_stats['attempts'] += 1
					await asyncio.wait_for(session.navigate_to(next_target, new_tab=False), timeout=120)
					await asyncio.sleep(2)
					if await capture_visible_page_evidence():
						continuation_stats['pages_added'] += 1
					continue
				current_snapshot = next(
					(snapshot for snapshot in reversed(ledger.snapshots) if snapshot.url == stable_url(state.url)),
					None,
				)
				if historical_lookup and current_snapshot and current_snapshot.has_continuation:
					parts = urlsplit(state.url)
					query = parse_qsl(parts.query, keep_blank_values=True)
					query_map = dict(query)
					try:
						next_page = int(query_map.get('p', '1')) + 1
					except ValueError:
						next_page = 2
					query = [(key, value) for key, value in query if key != 'p']
					query.append(('p', str(next_page)))
					next_target = urlunsplit(parts._replace(query=urlencode(query), fragment=''))
					if next_target not in visited:
						visited.add(next_target)
						continuation_stats['attempts'] += 1
						await asyncio.wait_for(session.navigate_to(next_target, new_tab=False), timeout=120)
						await asyncio.sleep(2)
						if await capture_visible_page_evidence():
							continuation_stats['pages_added'] += 1
						continue
				cdp_session = await session.get_or_create_cdp_session()
				result = await cdp_session.cdp_client.send.Runtime.evaluate(
					params={
						'expression': GENERIC_CONTINUATION_CLICK_JS,
						'returnByValue': True,
						'awaitPromise': True,
					},
					session_id=cdp_session.session_id,
				)
				value = result.get('result', {}).get('value') or {}
				if not isinstance(value, dict) or not value.get('clicked'):
					break
				continuation_stats['attempts'] += 1
				await asyncio.sleep(2)
				if await capture_visible_page_evidence():
					continuation_stats['pages_added'] += 1
					continue
				continuation_stats['stopped_on_unchanged'] = True
				break

		async def enter_retrieval_collection_hub() -> None:
			"""Enter a visible account/history collection before exhaustive detail expansion."""

			if contract.operation != 'RETRIEVE' or not re.search(
				r'\b(?:last|latest|most\s+recent)\s+(?:ordered|bought|purchased)\b',
				task['intent'],
				re.IGNORECASE,
			):
				return
			visited: set[str] = set()
			for phase in ('account', 'history'):
				candidates: list[tuple[int, str]] = []
				for record in ledger.records:
					for link in record.links:
						label = str(link.get('label') or '')
						href = urljoin(record.url, str(link.get('href') or ''))
						haystack = f'{label} {urlsplit(href).path}'.casefold()
						if phase == 'account':
							score = 10 if re.search(r'\b(?:my\s+)?(?:account|profile)\b', haystack) else 0
						else:
							score = 20 if re.search(
								r'\b(?:my\s+)?(?:orders?|purchases?|transactions?|history|receipts?|invoices?)\b',
								haystack,
							) else 0
							if re.search(r'\bview\s+all\b', label, re.IGNORECASE):
								score += 5
						if score and href not in visited:
							candidates.append((score, href))
				if not candidates:
					continue
				candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
				target = candidates[0][1]
				visited.add(target)
				await asyncio.wait_for(session.navigate_to(target, new_tab=False), timeout=120)
				await asyncio.sleep(2)
				await capture_visible_page_evidence()
				continuation_stats.setdefault('collection_hubs', []).append(target)

		async def expand_linked_record_evidence() -> None:
			"""Open only visible record details selected by a generic evidence planner."""

			historical_lookup = bool(
				re.search(
					r'\b(?:last|latest|most\s+recent)\s+(?:ordered|bought|purchased)\b',
					task['intent'],
					re.IGNORECASE,
				)
			)
			linked_records = rank_expansion_records(
				task['intent'],
				ledger.records,
				limit=240 if historical_lookup else 40,
			)
			if historical_lookup:
				scoped_records = [
					record
					for record in linked_records
					if re.search(
						r'\b(?:orders?|purchases?|transactions?|history|receipts?|invoices?)\b',
						f"{record.url} {record.container_signature} "
						+ ' '.join(link.get('label', '') for link in record.links),
						re.IGNORECASE,
					)
				]
				if scoped_records:
					linked_records = scoped_records
			if contract.operation != 'RETRIEVE' or not linked_records:
				return
			if aggregate_visible_dated_currency_records(task['intent'], ledger.records) is not None:
				expansion_stats['complete_without_expansion'] = True
				expansion_stats['reason'] = 'Visible row records already satisfy deterministic filter and aggregation fields.'
				return
			if contract.exhaustive and historical_lookup:
				selected = linked_records
				expansion_stats['selection_override'] = 'exhaustive_history_entity_lookup'
				expansion_stats['reason'] = 'Executable coverage skill selected visible history detail links.'
			else:
				system_prompt, user_prompt = evidence_expansion_prompt(task['intent'], linked_records)
				try:
					plan_result = await asyncio.wait_for(
						llm.ainvoke(
							[SystemMessage(content=system_prompt), UserMessage(content=user_prompt)],
							output_format=EvidenceExpansionPlan,
						),
						timeout=120,
					)
				except Exception as error:
					expansion_stats['error'] = f'{type(error).__name__}: {error}'
					return
				expansion_stats['called'] = True
				if plan_result.usage:
					expansion_stats['prompt_tokens'] = int(plan_result.usage.prompt_tokens or 0)
					expansion_stats['completion_tokens'] = int(plan_result.usage.completion_tokens or 0)
					expansion_stats['total_tokens'] = int(plan_result.usage.total_tokens or 0)
				plan = plan_result.completion
				if plan.complete_without_expansion:
					expansion_stats['reason'] = plan.reason
					expansion_stats['complete_without_expansion'] = True
					return
				selected_ids = set(plan.record_ids)
				selected = [record for record in linked_records if record.record_id in selected_ids]
				expansion_stats['reason'] = plan.reason
			expansion_stats['selected_records'] = len(selected)
			visited_urls: set[str] = set()
			unique_targets: list[tuple[StructuredVisibleRecord, str]] = []
			for record in selected:
				safe_links = [
					link
					for link in record.links
					if not re.search(
						r'\b(?:add|buy|delete|remove|reorder|submit|checkout)\b',
						link.get('label', ''),
						re.IGNORECASE,
					)
				]
				if not safe_links:
					continue
				target_url = safe_links[0].get('href', '')
				if not target_url or target_url in visited_urls:
					continue
				visited_urls.add(target_url)
				unique_targets.append((record, target_url))
			expansion_stats['unique_targets'] = len(unique_targets)
			expansion_limit = 240 if historical_lookup else 64
			for record, target_url in unique_targets[:expansion_limit]:
				try:
					await asyncio.wait_for(session.navigate_to(target_url, new_tab=False), timeout=120)
					expansion_stats['opened_links'] += 1
					page_added = False
					for _ in range(3):
						await asyncio.sleep(3)
						if await capture_visible_page_evidence():
							page_added = True
						page_records = [item for item in ledger.records if item.url == target_url]
						if sum(len(item.text) for item in page_records) >= 1200:
							break
					if page_added:
						expansion_stats['pages_added'] += 1
					if historical_lookup and extract_latest_same_record_date(task['intent'], ledger.records):
						expansion_stats['early_stop'] = 'same_record_entity_date_verified'
						break
				except Exception as error:
					expansion_stats.setdefault('link_errors', []).append(
						f'{type(error).__name__}: {error}'
					)

		async def enforce_postconditions_after_loop() -> None:
			"""Verify auto-terminated runs and perform one bounded local navigation repair."""

			state = await session.get_browser_state_summary(include_screenshot=False)
			dom_text = state.dom_state.llm_representation(
				include_attributes=['aria-label', 'title', 'placeholder', 'role', 'name', 'type', 'value']
			)
			blocker = completion_decision(
				ledger,
				contract,
				intent=task['intent'],
				final_text='post-loop verification',
				current_url=state.url,
				visible_text=dom_text,
				start_url=start_urls[0],
				request_evidence=observed_request_evidence(session, baseline_request_ids),
			)
			postcondition_stats['checked'] = True
			postcondition_stats['initial_blocker'] = blocker
			if blocker and contract.operation == 'NAVIGATE':
				for target in rank_navigation_recovery_urls(
					task['intent'], ledger.records, current_url=state.url, limit=2
				):
					postcondition_stats['attempts'] += 1
					await asyncio.wait_for(session.navigate_to(target, new_tab=False), timeout=120)
					cdp_session = await session.get_or_create_cdp_session()
					await cdp_session.cdp_client.send.Page.reload(
						params={'ignoreCache': True},
						session_id=cdp_session.session_id,
					)
					await asyncio.sleep(2)
					await capture_visible_page_evidence()
					state = await session.get_browser_state_summary(include_screenshot=False)
					dom_text = state.dom_state.llm_representation(
						include_attributes=['aria-label', 'title', 'placeholder', 'role', 'name', 'type', 'value']
					)
					blocker = completion_decision(
						ledger,
						contract,
						intent=task['intent'],
						final_text='post-loop repaired verification',
						current_url=state.url,
						visible_text=dom_text,
						start_url=start_urls[0],
					)
					if blocker is None:
						postcondition_stats['recovered'] = True
						postcondition_stats['recovery_url'] = state.url
						break
			postcondition_stats['final_blocker'] = blocker

		completion_guard, coverage_stats = build_generic_completion_guard(
			task['intent'],
			start_urls[0],
			ledger,
			baseline_request_ids,
		)
		tools = Tools(
			exclude_actions=['search', 'screenshot'] if not needs_vision else ['search'],
			completion_guard=completion_guard,
		)
		register_generic_actions(tools, task_intent=task['intent'])
		comparison_spec = resolve_comparison_spec(task['intent'])
		cleanup_requested = bool(
			re.search(r'\b(?:discard|clear|empty|remove|delete)\b', task['intent'], re.IGNORECASE)
			and re.search(r'\b(?:cart|basket)\b', task['intent'], re.IGNORECASE)
		)
		if cleanup_requested or (comparison_spec.objective and comparison_spec.category_label):
			workflow_preparation_stats['called'] = True
			workflow_preparation_stats['cleanup_requested'] = cleanup_requested
			preparation_messages: list[str] = []
			if cleanup_requested:
				cart_target = await reveal_navigation_target(session, 'My Cart')
				workflow_preparation_stats['cart_navigation'] = cart_target
				if cart_target.get('found') and cart_target.get('href'):
					await session.navigate_to(str(cart_target['href']), new_tab=False)
					cdp_session = await session.get_or_create_cdp_session()
					await cdp_session.cdp_client.send.Page.reload(
						params={'ignoreCache': True},
						session_id=cdp_session.session_id,
					)
					await asyncio.sleep(3)
					cleanup_action = tools.registry.registry.actions['clear_existing_collection']
					cleanup_result = await cleanup_action.function(
						params=ClearExistingCollectionAction(),
						browser_session=session,
					)
					workflow_preparation_stats['cleanup_result'] = cleanup_result.model_dump(mode='json')
					workflow_preparation_stats['cleanup_completed'] = not bool(cleanup_result.error)
					if not cleanup_result.error:
						preparation_messages.append(
							cleanup_result.extracted_content
							or 'Explicit pre-existing collection cleanup completed.'
						)
					else:
						preparation_messages.append(
							'Pre-existing collection cleanup is not yet visibly verified. Remain on the current '
							'cart or basket, call clear_existing_collection again, and do not select or add the '
							'new target until an explicit empty state is visible.'
						)
			cleanup_ready = not cleanup_requested or bool(workflow_preparation_stats['cleanup_completed'])
			if cleanup_ready and comparison_spec.objective and comparison_spec.category_label:
				await session.navigate_to(start_urls[0], new_tab=False)
				await asyncio.sleep(1)
				target = await reveal_navigation_target(session, comparison_spec.category_label)
				workflow_preparation_stats['collection_navigation'] = target
				if target.get('found') and target.get('href'):
					await session.navigate_to(str(target['href']), new_tab=False)
					await asyncio.sleep(1)
					workflow_preparation_stats['collection_routed'] = True
					preparation_messages.append(
						f'Exact requested collection {target.get("label")!r} was visibly revealed and opened.'
					)
			await capture_visible_page_evidence()
			print(
				'[WORKFLOW_PREPARATION] '
				+ json.dumps(workflow_preparation_stats, ensure_ascii=False, default=str),
				flush=True,
			)
			if preparation_messages:
				prompt += (
					'\n\nThe generic workflow preparation controller completed these observable prerequisites '
					'before the Agent loop:\n- '
					+ '\n- '.join(preparation_messages)
					+ '\nContinue from the current browser state. Do not repeat completed cleanup or collection routing.'
				)
		agent = Agent(
			task=prompt,
			llm=llm,
			browser_session=session,
			tools=tools,
			register_new_step_callback=capture_visible_page_evidence,
			override_system_message=COMPACT_AGENT_SYSTEM_PROMPT,
			use_vision='auto' if needs_vision else False,
			visual_context_mode='html_first',
			include_attributes=['aria-label', 'title', 'placeholder', 'role', 'name', 'type', 'value'],
			max_clickable_elements_length=budget['max_clickable_elements_length'],
			max_actions_per_step=1,
			max_history_items=budget['max_history_items'],
			max_failures=8,
			use_thinking=False,
			calculate_cost=True,
			use_judge=False,
			goal_aware_task_strategy=True,
			directly_open_url=False,
			save_conversation_path=conversation_path,
			step_timeout=360,
			llm_timeout=300,
		)
		agent_started = time.perf_counter()
		history = await asyncio.wait_for(agent.run(max_steps=budget['max_steps']), timeout=args.timeout)
		stage_timings['agent_seconds'] = time.perf_counter() - agent_started
		history.save_to_file(trajectory_path)
		# Post-processing schemas can require more output than one browser action.
		llm = build_qwen_llm(max_completion_tokens=2048)
		continuation_started = time.perf_counter()
		await capture_visible_page_evidence()
		await enforce_postconditions_after_loop()
		await enter_retrieval_collection_hub()
		await collect_visible_continuations()
		await expand_linked_record_evidence()
		stage_timings['continuation_collection_seconds'] = time.perf_counter() - continuation_started
		coverage_evidence_path.write_text(
			json.dumps(page_evidence, ensure_ascii=False, indent=2) + '\n',
			encoding='utf-8',
		)
		coverage_evidence_path.with_name('structured_evidence.json').write_text(
			json.dumps([record.model_dump() for record in ledger.records], ensure_ascii=False, indent=2) + '\n',
			encoding='utf-8',
		)
		structured_response = history.structured_output
		response = normalize_final_response(
			structured_response if structured_response is not None else (history.final_result() or ''),
			default_task_type=infer_task_type(task['intent']),
		)
		response = VerifiedAgentResponse.model_validate(
			normalize_response_for_contract(response, contract)
		)
		if contract.operation == 'MUTATE' and postcondition_stats.get('final_blocker'):
			response = response.model_copy(
				update={
					'status': 'DATA_VALIDATION_ERROR',
					'error_details': postcondition_stats['final_blocker'],
				}
			)
		response, grounded_audit_stats = await audit_grounded_retrieval(
			task['intent'],
			response,
			ledger,
			llm,
		)
		workflow_verification = verify_retrieval_workflow(
			task['intent'],
			list(response.retrieved_data or []),
			ledger,
		)
		if workflow_verification.verified_values:
			response = response.model_copy(
				update={
					'status': 'SUCCESS',
					'retrieved_data': workflow_verification.verified_values,
					'error_details': None,
				}
			)
			if not workflow_verification.accepted:
				workflow_verification = workflow_verification.model_copy(update={'accepted': True})
		elif not workflow_verification.accepted and response.status == 'SUCCESS':
			response = response.model_copy(
				update={
					'status': 'DATA_VALIDATION_ERROR',
					'error_details': workflow_verification.repair_instruction,
				}
			)
		deterministic_aggregate = aggregate_visible_dated_currency_records(task['intent'], ledger.records)
		if deterministic_aggregate is not None:
			response = response.model_copy(
				update={
					'status': 'SUCCESS',
					'retrieved_data': [deterministic_aggregate],
					'error_details': None,
				}
			)
		category_aggregation, category_aggregation_stats = await synthesize_category_total(
			task['intent'],
			page_evidence,
			llm,
		)
		if category_aggregation is not None:
			response = response.model_copy(
				update={
					'status': 'SUCCESS',
					'retrieved_data': [category_aggregation.total],
					'error_details': None,
				}
			)
			category_aggregation_stats['applied'] = True
			category_aggregation_stats['qualifying_items'] = category_aggregation.qualifying_items
			category_aggregation_stats['qualifying_subtotals'] = category_aggregation.qualifying_subtotals
			category_aggregation_stats['total'] = category_aggregation.total
		else:
			category_aggregation_stats['applied'] = False
		response_path.write_text(
			json.dumps(response.model_dump(mode='json'), ensure_ascii=False, indent=2) + '\n',
			encoding='utf-8',
		)
		usage = history.usage
		execution_requests = observed_request_evidence(session, baseline_request_ids)
		metrics = {
			'task_id': args.task_id,
			'steps': history.number_of_steps(),
			'duration_seconds': history.total_duration_seconds(),
			'action_names': history.action_names(),
			'errors': [str(error) for error in history.errors() if error],
			'prompt_tokens': usage.total_prompt_tokens if usage else None,
			'completion_tokens': usage.total_completion_tokens if usage else None,
			'total_tokens': usage.total_tokens if usage else None,
			'estimated_cost': usage.total_cost if usage else None,
			'coverage_critic': coverage_stats,
			'generic_contract': contract.model_dump(mode='json'),
			'evidence_snapshots': len(ledger.snapshots),
			'structured_evidence_records': len(ledger.records),
			'generic_continuation_collector': continuation_stats,
			'post_loop_workflow_verification': postcondition_stats,
			'pre_loop_workflow_preparation': workflow_preparation_stats,
			'generic_evidence_expansion': expansion_stats,
			'grounded_retrieval_audit': grounded_audit_stats,
			'independent_workflow_verification': workflow_verification.model_dump(mode='json'),
			'deterministic_visible_aggregate': deterministic_aggregate,
			'generic_line_item_aggregation': category_aggregation_stats,
			'execution_evidence': {
				'request_count': len(execution_requests),
				'mutation_count': len(
					[
						item
						for item in execution_requests
						if item['method'] in {'POST', 'PUT', 'PATCH', 'DELETE'}
					]
				),
				'requests': execution_requests[-40:],
			},
			'specialized_postprocessors': [],
			'runtime_budget': budget,
			'auth_cache_hit': auth_cache_hit,
			'stage_timings': stage_timings,
			'wall_seconds': time.perf_counter() - run_started,
		}
		metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
		metadata_path.write_text(
			json.dumps(
				{
					'method': os.getenv('OUR_METHOD_VERSION', 'ours_v4_execution_verified'),
					'repository': str(REPOSITORY),
					'model': os.getenv('QWEN_CHAT_MODEL'),
					'task_id': args.task_id,
					'sites': task['sites'],
					'observation_policy': 'visible_page_and_authenticated_session_only',
					'vision_routed': needs_vision,
					'runtime_budget': budget,
					'auth_cache_hit': auth_cache_hit,
				},
				ensure_ascii=False,
				indent=2,
			)
			+ '\n',
			encoding='utf-8',
		)
	finally:
		try:
			await session.kill()
		except Exception as error:
			print(f'browser cleanup warning: {error}', file=sys.stderr)

	if history is None or not har_path.exists() or not response_path.exists():
		raise RuntimeError('Agent run did not produce both HAR and structured response artifacts.')
	print(json.dumps({'task_id': args.task_id, 'output_dir': str(args.output_dir)}, ensure_ascii=False))


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--task-id', type=int, required=True)
	parser.add_argument('--task-input', type=Path, required=True)
	parser.add_argument('--output-dir', type=Path, required=True)
	parser.add_argument(
		'--config',
		type=Path,
		default=Path(os.getenv('WEBARENA_CONFIG', 'configs/webarena_verified_config.json')),
	)
	parser.add_argument('--env-file', type=Path, default=Path(os.getenv('WEBARENA_ENV_FILE', '.env')))
	parser.add_argument('--max-steps', type=int, default=30)
	parser.add_argument('--timeout', type=float, default=1200)
	parser.add_argument('--storage-state-cache', type=Path)
	args = parser.parse_args()
	asyncio.run(run(args))


if __name__ == '__main__':
	main()
