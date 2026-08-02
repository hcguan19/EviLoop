"""Dataset-agnostic task constraints, capability routing, and progress recovery."""

from __future__ import annotations

import hashlib
import re
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field


_STOP_WORDS = {
	'a',
	'an',
	'and',
	'are',
	'as',
	'at',
	'be',
	'before',
	'by',
	'complete',
	'current',
	'do',
	'for',
	'from',
	'go',
	'i',
	'in',
	'instruction',
	'is',
	'it',
	'looking',
	'like',
	'lower',
	'made',
	'me',
	'my',
	'need',
	'of',
	'on',
	'open',
	'or',
	'page',
	'please',
	'price',
	'prefer',
	'preferably',
	'shown',
	'should',
	'task',
	'than',
	'that',
	'the',
	'this',
	'to',
	'using',
	'want',
	'website',
	'with',
	'would',
	'dollar',
	'dollars',
}


def normalize_text(value: str) -> str:
	"""Normalize visible text for stable comparisons without changing semantics."""

	return re.sub(r'[^a-z0-9.$]+', ' ', value.casefold()).strip()


def meaningful_tokens(value: str) -> list[str]:
	"""Return ordered, de-duplicated task tokens suitable for visible evidence matching."""

	tokens: list[str] = []
	for token in re.findall(r'[a-z0-9]+(?:[.-][a-z0-9]+)*', value.casefold()):
		if len(token) < 2 or token in _STOP_WORDS or re.fullmatch(r'\d+(?:[.-]\d+)*', token):
			continue
		if token not in tokens:
			tokens.append(token)
	return tokens


def visible_value_matches_text(value: str, text: str) -> bool:
	"""Match a visible control value to task text, including conservative morphology."""

	value_normalized = normalize_text(value)
	text_normalized = normalize_text(text)
	if not value_normalized or not text_normalized:
		return False
	if re.search(rf'(?<![a-z0-9]){re.escape(value_normalized)}(?![a-z0-9])', text_normalized):
		return True

	value_tokens = meaningful_tokens(value_normalized)
	text_tokens = meaningful_tokens(text_normalized)
	if not value_tokens or not text_tokens:
		return False
	for value_token in value_tokens:
		if not any(
			value_token == text_token
			or (
				min(len(value_token), len(text_token)) >= 4
				and (value_token.startswith(text_token) or text_token.startswith(value_token))
			)
			for text_token in text_tokens
		):
			return False
	return True


def is_visible_product_detail(url: str, title: str) -> bool:
	"""Require both a detail-page route and a non-empty visible product title."""

	return '/item_page/' in urlsplit(url).path and bool(normalize_text(title))


class TaskConstraint(BaseModel):
	"""One requirement grounded in user text or currently visible controls."""

	name: str
	operator: Literal['contains', 'equals', 'at_most', 'at_least', 'selected', 'completed']
	expected: str | float | int | bool
	required: bool = True
	source: Literal['task', 'visible_option', 'visible_control'] = 'task'


class TaskRequirements(BaseModel):
	"""Compact task representation shared by browser, desktop, and replay adapters."""

	objective: str
	keywords: list[str] = Field(default_factory=list, max_length=40)
	constraints: list[TaskConstraint] = Field(default_factory=list, max_length=50)
	completion_markers: list[str] = Field(default_factory=list, max_length=12)
	irreversible_intent: bool = False
	max_price: float | None = None


class PageCapabilities(BaseModel):
	"""Observable interaction capabilities independent of websites and datasets."""

	form_controls: int = 0
	search_controls: int = 0
	candidate_count: int = 0
	selectable_options: int = 0
	table_rows: int = 0
	links: int = 0
	visual_regions: int = 0
	file_controls: int = 0
	dynamic_controls: int = 0
	commit_controls: int = 0
	terminal_evidence: bool = False

	def routes(self) -> list[str]:
		"""Choose only capabilities supported by the current observable state."""

		routes: list[str] = []
		if self.search_controls and self.candidate_count >= 2:
			routes.append('candidate_search')
		if self.candidate_count >= 2 or self.table_rows >= 2:
			routes.append('candidate_ranking')
		if self.form_controls:
			routes.append('form_interaction')
		if self.selectable_options:
			routes.append('option_selection')
		if self.file_controls:
			routes.append('file_interaction')
		if self.dynamic_controls:
			routes.append('predicate_sampling')
		if self.commit_controls:
			routes.append('commit_validation')
		if self.visual_regions and not (self.form_controls or self.links or self.candidate_count):
			routes.append('visual_grounding')
		if self.terminal_evidence:
			routes.append('completion_validation')
		return routes or ['general_navigation']


ObservationModality = Literal['dom', 'vision_dom', 'desktop_visual', 'offline_trajectory']


class TaskObservation(BaseModel):
	"""Normalized observation produced by any benchmark-specific environment adapter."""

	modality: ObservationModality
	location: str = ''
	visible_text: str = ''
	capabilities: PageCapabilities = Field(default_factory=PageCapabilities)
	candidates: list['VisibleCandidate'] = Field(default_factory=list, max_length=100)
	screenshot_available: bool = False
	action_history_available: bool = False
	provenance: Literal['live_visible_state', 'recorded_trajectory'] = 'live_visible_state'


class PolicyRoute(BaseModel):
	"""Dataset-independent route selected from observable capabilities and modality."""

	primary: str
	fallbacks: list[str] = Field(default_factory=list)
	requires_visual_grounding: bool = False
	reason: str


def route_observation(observation: TaskObservation) -> PolicyRoute:
	"""Select a policy route without inspecting a benchmark name or hidden evaluator state."""

	routes = observation.capabilities.routes()
	visual_required = observation.modality == 'desktop_visual' or (
		observation.modality == 'vision_dom' and routes == ['general_navigation'] and observation.screenshot_available
	)
	if visual_required:
		return PolicyRoute(
			primary='visual_grounding',
			fallbacks=['refresh_state', 'replan'],
			requires_visual_grounding=True,
			reason='The observation exposes no sufficient semantic controls, so screenshot grounding is required.',
		)
	if observation.modality == 'offline_trajectory':
		return PolicyRoute(
			primary=routes[0],
			fallbacks=routes[1:] + ['trajectory_consistency_check'],
			requires_visual_grounding=False,
			reason='Choose actions from recorded observable state and validate them against trajectory evidence.',
		)
	return PolicyRoute(
		primary=routes[0],
		fallbacks=routes[1:] + ['refresh_state', 'replan'],
		requires_visual_grounding=False,
		reason='The route is supported by controls and content exposed in the current live observation.',
	)


class VisibleCandidate(BaseModel):
	"""A candidate exposed by the current page, table, list, or search result DOM."""

	identifier: str
	label: str
	url: str = ''
	visible_text: str = ''
	price: float | None = None
	option_values: list[str] = Field(default_factory=list)
	metadata: dict[str, str | float | int | bool | None] = Field(default_factory=dict)


class RankedCandidate(BaseModel):
	"""A visible candidate plus an auditable task-constraint score."""

	candidate: VisibleCandidate
	score: float
	matched_keywords: list[str] = Field(default_factory=list)
	matched_options: list[str] = Field(default_factory=list)
	violations: list[str] = Field(default_factory=list)
	eligible: bool = True
	keyword_coverage: float = 0.0
	title_coverage: float = 0.0


class RankedCandidateAction(BaseModel):
	"""Select one candidate by its 1-based rank from visible evidence."""

	rank: int = Field(ge=1, le=20)


def extract_task_requirements(task: str, visible_option_values: list[str] | None = None) -> TaskRequirements:
	"""Parse task-level constraints without benchmark metadata or hidden environment state."""

	normalized = normalize_text(task)
	price_patterns = (
		r'(?:under|below|less than|no more than|at most|up to|max(?:imum)?(?: price)?)\s*\$?\s*(\d+(?:\.\d+)?)',
		r'\$\s*(\d+(?:\.\d+)?)\s*(?:or less|maximum|max)',
	)
	max_price = None
	for pattern in price_patterns:
		match = re.search(pattern, normalized, flags=re.IGNORECASE)
		if match:
			max_price = float(match.group(1))
			break

	constraints: list[TaskConstraint] = []
	if max_price is not None:
		constraints.append(TaskConstraint(name='price', operator='at_most', expected=max_price))

	for value in visible_option_values or []:
		option = normalize_text(value)
		if option and re.search(rf'(?<![a-z0-9]){re.escape(option)}(?![a-z0-9])', normalized):
			constraints.append(
				TaskConstraint(name='visible_option', operator='selected', expected=value, source='visible_option')
			)

	irreversible = bool(
		re.search(r'\b(?:buy|purchase|checkout|submit|send|delete|remove|publish|confirm|book|order|save)\b', normalized)
	)
	completion_markers = ['success', 'completed', 'confirmation']
	if re.search(r'\b(?:buy|checkout|order|purchase)\b', normalized):
		completion_markers.extend(['score', 'reward', 'order placed'])

	return TaskRequirements(
		objective=re.sub(r'\s+', ' ', task).strip(),
		keywords=meaningful_tokens(task)[:40],
		constraints=constraints,
		completion_markers=list(dict.fromkeys(completion_markers)),
		irreversible_intent=irreversible,
		max_price=max_price,
	)


def rank_visible_candidates(
	requirements: TaskRequirements,
	candidates: list[VisibleCandidate],
	*,
	rejected_identifiers: set[str] | None = None,
) -> list[RankedCandidate]:
	"""Rank only visible candidates using task constraints and auditable evidence."""

	rejected = rejected_identifiers or set()
	ranked: list[RankedCandidate] = []
	for candidate in candidates:
		label_tokens = set(meaningful_tokens(candidate.label))
		evidence_tokens = set(meaningful_tokens(f'{candidate.label} {candidate.visible_text}'))
		matched_keywords = [token for token in requirements.keywords if token in evidence_tokens]
		matched_title = [token for token in requirements.keywords if token in label_tokens]
		matched_options = [
			option
			for option in candidate.option_values
			if normalize_text(option) and normalize_text(option) in normalize_text(requirements.objective)
		]
		violations: list[str] = []
		eligible = candidate.identifier not in rejected
		if not eligible:
			violations.append('candidate_was_rejected')
		if requirements.max_price is not None and candidate.price is not None and candidate.price > requirements.max_price:
			eligible = False
			violations.append(f'price_above_{requirements.max_price:g}')
		if requirements.keywords and not matched_keywords:
			eligible = False
			violations.append('no_task_keyword_match')

		coverage = len(matched_keywords) / max(1, len(requirements.keywords))
		title_coverage = len(matched_title) / max(1, len(requirements.keywords))
		# Coverage is normalized so a verbose task cannot win merely by contributing
		# more weak token matches. Title evidence is weighted more heavily than
		# surrounding card text, while visible option matches remain explicit.
		score = 6.0 * coverage + 3.0 * title_coverage + 0.75 * len(matched_options)
		if requirements.max_price is not None and candidate.price is not None and candidate.price <= requirements.max_price:
			score += 0.75
		if not eligible:
			score -= 100.0
		ranked.append(
			RankedCandidate(
				candidate=candidate,
				score=round(score, 4),
				matched_keywords=matched_keywords,
				matched_options=matched_options,
				violations=violations,
				eligible=eligible,
				keyword_coverage=round(coverage, 4),
				title_coverage=round(title_coverage, 4),
			)
		)

	return sorted(
		ranked,
		key=lambda item: (
			item.eligible,
			item.score,
			len(item.matched_options),
			-item.candidate.price if item.candidate.price is not None else float('-inf'),
		),
		reverse=True,
	)


class RecoveryDirective(BaseModel):
	"""Program-level recovery instruction emitted when execution stops making progress."""

	triggered: bool = False
	reason: str = ''
	action: Literal[
		'continue',
		'refresh_state',
		'backtrack',
		'choose_new_candidate',
		'replan',
		'abort',
	] = 'continue'
	blocked_signature: str = ''


class ProgressTracker(BaseModel):
	"""Bounded execution memory for loops, revisits, and candidate exhaustion."""

	max_recent: int = Field(default=12, ge=4, le=50)
	max_same_state: int = Field(default=3, ge=2, le=10)
	max_same_action: int = Field(default=2, ge=2, le=12)
	max_candidate_visits: int = Field(default=2, ge=1, le=10)
	recent_states: list[str] = Field(default_factory=list)
	recent_actions: list[str] = Field(default_factory=list)
	candidate_visits: dict[str, int] = Field(default_factory=dict)
	rejected_candidates: set[str] = Field(default_factory=set)

	@staticmethod
	def stable_url(url: str) -> str:
		"""Remove fragments while preserving state-bearing path and query information."""

		parts = urlsplit(url)
		return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ''))

	@staticmethod
	def state_signature(url: str, visible_text: str = '', candidate_id: str = '') -> str:
		"""Build a compact signature without retaining full page content."""

		digest = hashlib.sha1(normalize_text(visible_text)[:2000].encode('utf-8')).hexdigest()[:10]
		return f'{ProgressTracker.stable_url(url)}|{candidate_id}|{digest}'

	def record(
		self,
		*,
		url: str,
		action: str,
		visible_text: str = '',
		candidate_id: str = '',
		progressed: bool = False,
	) -> RecoveryDirective:
		"""Record one transition and deterministically request recovery when it loops."""

		signature = self.state_signature(url, visible_text, candidate_id)
		self.recent_states.append(signature)
		self.recent_actions.append(action)
		self.recent_states = self.recent_states[-self.max_recent :]
		self.recent_actions = self.recent_actions[-self.max_recent :]
		if candidate_id:
			self.candidate_visits[candidate_id] = self.candidate_visits.get(candidate_id, 0) + 1

		if progressed:
			return RecoveryDirective()
		if candidate_id and self.candidate_visits[candidate_id] > self.max_candidate_visits:
			self.rejected_candidates.add(candidate_id)
			return RecoveryDirective(
				triggered=True,
				reason='The same candidate was revisited without satisfying the task constraints.',
				action='choose_new_candidate',
				blocked_signature=signature,
			)
		if len(self.recent_states) >= self.max_same_state and len(set(self.recent_states[-self.max_same_state :])) == 1:
			return RecoveryDirective(
				triggered=True,
				reason='The observable state did not change across multiple actions.',
				action='backtrack',
				blocked_signature=signature,
			)
		if len(self.recent_actions) >= self.max_same_action and len(set(self.recent_actions[-self.max_same_action :])) == 1:
			return RecoveryDirective(
				triggered=True,
				reason=f'The action {action!r} repeated without visible progress.',
				action='replan',
				blocked_signature=signature,
			)
		return RecoveryDirective()


class RetrievalCoverage(BaseModel):
	"""Evidence ledger used to prevent premature completion of exhaustive retrieval."""

	exhaustive: bool = False
	seen_items: list[str] = Field(default_factory=list, max_length=500)
	visited_page_signatures: list[str] = Field(default_factory=list, max_length=100)
	visible_total: int | None = None
	has_next_control: bool = False
	has_load_more_control: bool = False
	has_collapsed_relevant_section: bool = False

	def record_items(self, items: list[str]) -> None:
		for item in items:
			normalized = normalize_text(item)
			if normalized and normalized not in self.seen_items:
				self.seen_items.append(normalized)

	def completion_blockers(self) -> list[str]:
		"""Return only blockers observable from task text and current page state."""

		if not self.exhaustive:
			return []
		blockers: list[str] = []
		if self.has_next_control:
			blockers.append('relevant next-page control remains')
		if self.has_load_more_control:
			blockers.append('relevant load-more control remains')
		if self.has_collapsed_relevant_section:
			blockers.append('relevant collapsed section remains')
		if self.visible_total is not None and len(self.seen_items) < self.visible_total:
			blockers.append(f'visible total indicates {self.visible_total} records but only {len(self.seen_items)} were checked')
		return blockers

	def completion_allowed(self) -> bool:
		return not self.completion_blockers()


def intent_requires_exhaustive_retrieval(intent: str) -> bool:
	"""Detect plural/exhaustive retrieval without benchmark or website identities."""

	return bool(
		re.search(
			r'\b(?:all|each|every|name(?:s|\(s\))|item(?:s|\(s\))|order(?:s|\(s\))|'
			r'review(?:s|\(s\))|reviewer(?:s|\(s\))|result(?:s|\(s\))|comment(?:s|\(s\))|'
			r'post(?:s|\(s\))|issue(?:s|\(s\))|repositories)(?![a-z])',
			intent,
			flags=re.IGNORECASE,
		)
	)


def coverage_policy_directive(intent: str) -> str:
	"""Render concise control guidance consumed by live agents and replay harnesses."""

	lines = [
		'Prefer a visible link, tab, accordion, or button matching the requested section over blind scrolling.',
		'After two actions without new relevant evidence, change action type or navigation target.',
	]
	if intent_requires_exhaustive_retrieval(intent):
		lines.extend(
			[
				'Accumulate unique matching records across all relevant visible sections and pages.',
				'Before completion, exhaust relevant next/load-more controls, collapsed sections, and visible total counts.',
				'Do not treat the first visible block as complete when the page exposes unchecked records.',
			]
		)
	return '\n'.join(f'- {line}' for line in lines)


def completion_is_grounded(
	requirements: TaskRequirements,
	*,
	url: str,
	visible_text: str,
	unresolved_constraints: list[str] | None = None,
) -> bool:
	"""Require visible terminal evidence before accepting completion."""

	if unresolved_constraints:
		return False
	normalized_url = normalize_text(url)
	normalized_text = normalize_text(visible_text)
	url_terminal = bool(re.search(r'(?:^|\s)(?:done|success|confirmation|complete)(?:\s|$)', normalized_url))
	strong_terminal_patterns = (
		r'\btask (?:is )?(?:complete|completed|successful)\b',
		r'\b(?:successfully|has been) (?:submitted|sent|saved|placed|booked|completed)\b',
		r'\bconfirmation (?:number|id|code)\b',
		r'\byour (?:score|reward)\s*(?::|=)',
		r'\b(?:final )?(?:score|reward)\s*(?::|=)\s*[01](?:\.\d+)?\b',
	)
	text_terminal = any(re.search(pattern, normalized_text) for pattern in strong_terminal_patterns)
	return url_terminal or text_terminal
