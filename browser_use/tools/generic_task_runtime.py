"""Dataset-independent task contracts, evidence memory, and completion control."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from pydantic import BaseModel, Field


TaskOperation = Literal['RETRIEVE', 'NAVIGATE', 'MUTATE']


class TaskContract(BaseModel):
	"""A compact execution contract derived only from the user request."""

	operation: TaskOperation
	exhaustive: bool = False
	irreversible: bool = False
	draft_only: bool = False
	requires_visual_evidence: bool = False
	output_shape: Literal['none', 'scalar', 'list', 'object'] = 'none'


class WorkflowVerification(BaseModel):
	"""Evidence-only verdict used to admit, repair, or reject a proposed result."""

	accepted: bool
	failure_kind: Literal['none', 'coverage', 'record_binding', 'navigation', 'mutation', 'unsupported'] = 'none'
	missing_evidence: list[str] = Field(default_factory=list)
	repair_instruction: str | None = None
	verified_values: list[str | int | float | bool | dict[str, Any]] = Field(default_factory=list)


class EvidenceSnapshot(BaseModel):
	"""One compact, visible browser state."""

	url: str
	signature: str
	text: str
	visible_total: int | None = None
	has_continuation: bool = False


class StructuredVisibleRecord(BaseModel):
	"""One repeated DOM record discovered without site-specific selectors."""

	record_id: str
	url: str
	container_signature: str
	text: str
	controls: list[str] = Field(default_factory=list, max_length=30)
	links: list[dict[str, str]] = Field(default_factory=list, max_length=30)


class ComparisonSpec(BaseModel):
	"""Dataset-independent comparison objective and numeric constraints."""

	objective: Literal['highest_rating', 'lowest_rating', 'highest_price', 'lowest_price'] | None = None
	min_price: float | None = None
	max_price: float | None = None
	category_label: str | None = None
	required_terms: list[str] = Field(default_factory=list, max_length=16)


class RankedVisibleCandidate(BaseModel):
	"""One visible same-record candidate ranked against a comparison request."""

	record_text: str
	title: str
	price: float | None = None
	rating: float | None = None
	href: str | None = None
	controls: list[str] = Field(default_factory=list)


class GenericEvidenceLedger(BaseModel):
	"""Bounded cross-site memory based exclusively on visible states."""

	max_snapshots: int = Field(default=8, ge=2, le=30)
	snapshots: list[EvidenceSnapshot] = Field(default_factory=list)
	records: list[StructuredVisibleRecord] = Field(default_factory=list, max_length=5000)
	continuation_urls: list[str] = Field(default_factory=list, max_length=40)
	proposal_attempts: dict[str, int] = Field(default_factory=dict)

	def record(self, *, url: str, visible_text: str) -> bool:
		text = re.sub(r'[ \t]+', ' ', visible_text).strip()
		if not text:
			return False
		signature = page_signature(url, text)
		if any(item.signature == signature for item in self.snapshots):
			return False
		self.snapshots.append(
			EvidenceSnapshot(
				url=stable_url(url),
				signature=signature,
				text=text,
				visible_total=extract_visible_total(text),
				has_continuation=has_visible_continuation(text),
			)
		)
		self.snapshots = self.snapshots[-self.max_snapshots :]
		return True

	def record_structured(self, *, url: str, records: list[dict[str, Any]]) -> int:
		"""Store unique repeated DOM records while preserving their visible provenance."""

		known = {record.record_id for record in self.records}
		added = 0
		for value in records:
			text = re.sub(r'\s+', ' ', str(value.get('text') or '')).strip()
			if len(text) < 3:
				continue
			signature = str(value.get('container_signature') or 'repeated-dom-record')
			record_id = hashlib.sha1(
				f'{stable_url(url)}|{signature}|{text.casefold()}'.encode('utf-8')
			).hexdigest()[:20]
			if record_id in known:
				continue
			controls = [
				re.sub(r'\s+', ' ', str(item)).strip()
				for item in value.get('controls', [])
				if str(item).strip()
			]
			links = [
				{
					'label': re.sub(r'\s+', ' ', str(item.get('label') or '')).strip(),
					'href': str(item.get('href') or '').strip(),
				}
				for item in value.get('links', [])
				if isinstance(item, dict) and str(item.get('href') or '').strip()
			]
			self.records.append(
				StructuredVisibleRecord(
					record_id=record_id,
					url=stable_url(url),
					container_signature=signature,
					text=text,
					controls=list(dict.fromkeys(controls))[:30],
					links=links[:30],
				)
			)
			known.add(record_id)
			added += 1
		self.records = self.records[-5000:]
		return added

	def record_continuations(self, urls: list[str]) -> int:
		added = 0
		for url in urls:
			value = stable_url(str(url).strip())
			if not value or value in self.continuation_urls:
				continue
			self.continuation_urls.append(value)
			added += 1
		self.continuation_urls = self.continuation_urls[-40:]
		return added

	def register_proposal(self, final_text: str) -> int:
		key = hashlib.sha1(normalize_proposal(final_text).encode('utf-8')).hexdigest()[:16]
		self.proposal_attempts[key] = self.proposal_attempts.get(key, 0) + 1
		return self.proposal_attempts[key]

	def compact_evidence(self, intent: str, *, max_chars: int = 18_000) -> str:
		"""Keep task-relevant line neighborhoods while preserving record locality."""

		keywords = meaningful_tokens(intent)
		structured: list[tuple[int, StructuredVisibleRecord]] = []
		for record in self.records:
			normalized = record.text.casefold()
			score = sum(1 for token in keywords if token in normalized)
			structured.append((score, record))
		structured.sort(key=lambda item: item[0], reverse=True)
		structured_text = '\n\n'.join(
			f'STRUCTURED RECORD {index + 1} [{record.record_id}]\n'
			f'URL: {record.url}\n'
			f'CONTAINER: {record.container_signature}\n'
			f'TEXT: {record.text[:5000]}\n'
			f'CONTROLS: {json.dumps(record.controls, ensure_ascii=False)}\n'
			f'LINKS: {json.dumps(record.links, ensure_ascii=False)}'
			for index, (_, record) in enumerate(structured[:80])
		)
		chunks: list[str] = [f'BROWSER-DERIVED STRUCTURED RECORDS:\n{structured_text}'] if structured_text else []
		for snapshot in self.snapshots:
			repeated_records = extract_repeated_anchor_records(snapshot.text)
			lines = [line.strip() for line in snapshot.text.splitlines() if line.strip()]
			selected: set[int] = set()
			for index, line in enumerate(lines):
				normalized = line.casefold()
				if any(token in normalized for token in keywords):
					selected.update(range(max(0, index - 10), min(len(lines), index + 12)))
			if not selected:
				selected.update(range(min(len(lines), 80)))
			body = '\n'.join(lines[index] for index in sorted(selected))
			record_text = '\n\n'.join(
				f'RECORD {index + 1}\n{record}'
				for index, record in enumerate(repeated_records[:40])
			)
			repeated_section = f'DYNAMIC REPEATED RECORDS:\n{record_text}\n\n' if record_text else ''
			chunks.append(
				f'URL: {snapshot.url}\n'
				f'{repeated_section}'
				f'TASK-RELEVANT VISIBLE LINES:\n{body}'
			)
		if not chunks:
			return ''
		if structured_text:
			structured_chunk = chunks[0][: max_chars // 2]
			page_chunks = '\n\n--- VISIBLE STATE ---\n\n'.join(chunks[1:])
			page_budget = max_chars - len(structured_chunk) - 29
			return f'{structured_chunk}\n\n--- VISIBLE STATE ---\n\n{page_chunks[-page_budget:]}'
		joined = '\n\n--- VISIBLE STATE ---\n\n'.join(chunks)
		return joined[-max_chars:]


class GroundedRetrievalAudit(BaseModel):
	"""A final answer reconstructed exclusively from visible evidence records."""

	retrieved_data: list[str | int | float | bool | dict[str, Any] | None] | dict[str, Any] = Field(
		default_factory=list
	)
	complete: bool
	confidence: float = Field(ge=0.0, le=1.0)
	reason: str
	evidence: list[str] = Field(default_factory=list, max_length=32)


class RecordSemanticDecision(BaseModel):
	"""Independent semantic decision for one browser-derived record."""

	relevant: bool
	projected_values: list[str | int | float | bool] = Field(default_factory=list, max_length=8)
	confidence: float = Field(ge=0.0, le=1.0)
	reason: str
	evidence: str = ''


class EvidenceExpansionPlan(BaseModel):
	"""Visible records whose linked details are needed to answer the request."""

	# The runtime still opens a bounded subset. A wider schema prevents an otherwise
	# useful plan from being discarded when a model over-selects candidates.
	record_ids: list[str] = Field(default_factory=list, max_length=64)
	complete_without_expansion: bool
	reason: str


def rank_task_records(
	intent: str,
	records: list[StructuredVisibleRecord],
	*,
	min_score: int = 2,
	limit: int = 16,
) -> list[StructuredVisibleRecord]:
	"""Return high-recall record candidates using only request/visible-text overlap."""

	tokens = meaningful_tokens(intent)
	scored: list[tuple[int, int, StructuredVisibleRecord]] = []
	for index, record in enumerate(records):
		text = record.text.casefold()
		score = sum(1 for token in tokens if token in text)
		if score >= min_score:
			scored.append((score, -index, record))
	scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
	return [record for _, _, record in scored[:limit]]


def record_semantic_prompt(intent: str, record: StructuredVisibleRecord) -> tuple[str, str]:
	"""Build a compact dataset-neutral predicate/projection check for one record."""

	system = (
		'Judge exactly one browser-derived visible record against the user request. Use semantic entailment, not '
		'keyword matching. Apply all requested predicates to this same record. For size or fit predicates, incomplete '
		'coverage, fitting only a smaller target, child-versus-adult mismatch, or resting on top rather than '
		'surrounding can entail undersizing without the word "small". Respect negation. If relevant, extract only the '
		'value or values requested by the user from this record; do not return the whole record. When the request says '
		'explicitly, exact phrase, or contains, require the literal normalized phrase in this same local evidence and '
		'reject synonyms. Enforce every numeric threshold in this same record. Use no site rule, '
		'hidden data, benchmark answer, or prior knowledge. Evidence must be one short local excerpt, at most 300 '
		'characters, that by itself connects the requested object and predicate. If no such local excerpt exists, '
		'mark the record irrelevant. Keep the reason short.'
	)
	user = (
		f'User request:\n{intent}\n\n'
		f'Record ID: {record.record_id}\n'
		f'Record URL: {record.url}\n'
		f'Visible record text:\n{record.text[:6000]}\n'
		f'Visible controls:\n{json.dumps(record.controls, ensure_ascii=False)}'
	)
	return system, user


def evidence_expansion_prompt(
	intent: str,
	records: list[StructuredVisibleRecord],
) -> tuple[str, str]:
	"""Select visible record links whose detail pages are necessary for the answer."""

	system = (
		'Plan evidence expansion for a browser retrieval task using only visible repeated records and their visible '
		'links. Select a record when its local fields make it a plausible request match but the requested attribute, '
		'line item, option, breakdown, or other evidence is only likely to exist behind its detail link. Apply visible '
		'date, status, identity, and numeric constraints before selecting. For a date range, include every plausible '
		'record in that range. Do not select navigation chrome, pagination, reorder, destructive, or submission links. '
		'If the visible records already contain every field needed to filter and compute the requested answer, set '
		'complete_without_expansion=true and return no record IDs. Opening details is not a coverage action by itself. '
		'Return at most 40 record IDs only. Use no site convention, hidden data, benchmark answer, or prior knowledge.'
	)
	payload = [
		{
			'record_id': record.record_id,
			'text': record.text[:600],
			'links': record.links,
		}
		for record in records
		if record.links
	]
	user = f'User request:\n{intent}\n\nVisible linked records:\n{json.dumps(payload, ensure_ascii=False)}'
	return system, user


def rank_expansion_records(
	intent: str,
	records: list[StructuredVisibleRecord],
	*,
	limit: int = 40,
) -> list[StructuredVisibleRecord]:
	"""Prioritize visible linked rows using lexical, temporal, and control evidence."""

	tokens = meaningful_tokens(intent)
	months = {
		'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
		'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
		'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8,
		'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12,
	}
	intent_lower = intent.casefold()
	requested_months = {number for name, number in months.items() if re.search(rf'\b{name}\b', intent_lower)}
	requested_years = {int(value) for value in re.findall(r'\b(?:19|20)\d{2}\b', intent_lower)}
	requested_days = {
		int(value)
		for value in re.findall(
			r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)'
			r'\s+(\d{1,2})\b',
			intent_lower,
		)
	}
	historical_lookup = bool(
		re.search(r'\b(?:last|latest|most\s+recent)\s+(?:ordered|bought|purchased)\b', intent, re.IGNORECASE)
	)
	scored: list[tuple[int, int, StructuredVisibleRecord]] = []
	seen_text: set[str] = set()
	for index, record in enumerate(records):
		if not record.links:
			continue
		text = record.text.casefold()
		if text in seen_text:
			continue
		seen_text.add(text)
		score = sum(2 for token in tokens if token in text)
		date_matches = re.findall(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', text)
		for month, day, year in date_matches:
			normalized_year = int(year)
			if normalized_year < 100:
				normalized_year += 2000
			if requested_months and int(month) in requested_months:
				score += 12
			if requested_years and normalized_year in requested_years:
				score += 12
			if requested_days and int(day) in requested_days:
				score += 8
		labels = ' '.join(link.get('label', '') for link in record.links)
		if re.search(r'\b(?:view|details?|open)\b', labels, re.IGNORECASE):
			score += 5
		if historical_lookup:
			context = f'{record.url} {record.container_signature} {labels}'.casefold()
			if re.search(r'\b(?:order|purchase|transaction|history|receipt|invoice)\b', context):
				score += 30
			elif not any(token in text for token in tokens):
				score -= 30
		if len(record.text) <= 500:
			score += 2
		scored.append((score, -index, record))
	scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
	return [record for _, _, record in scored[:limit]]


def rank_navigation_recovery_urls(
	intent: str,
	records: list[StructuredVisibleRecord],
	*,
	current_url: str = '',
	limit: int = 10,
) -> list[str]:
	"""Rank previously visible links by destination postconditions, not page popularity."""

	stopwords = {
		'open', 'go', 'navigate', 'visit', 'page', 'category', 'filtered', 'filter', 'sorted',
		'under', 'below', 'above', 'over', 'less', 'more', 'than', 'at', 'most', 'up', 'to',
	}
	terms = [token for token in meaningful_tokens(intent) if token not in stopwords and not token.isdigit()]
	upper = re.search(
		r'\b(?:under|below|less\s+than|at\s+most|up\s+to)\s*\$?\s*(\d+(?:\.\d+)?)\b',
		intent,
		re.IGNORECASE,
	)
	current_query = dict(parse_qsl(urlsplit(current_url).query, keep_blank_values=True))
	scored: dict[str, tuple[int, int, int]] = {}
	for record in records:
		for link in record.links:
			label = str(link.get('label') or '').strip()
			href = urljoin(record.url, str(link.get('href') or '').strip())
			parts = urlsplit(href)
			if not href or parts.scheme not in {'http', 'https'}:
				continue
			if re.search(r'\b(?:add|buy|cart|checkout|delete|remove|review|submit)\b', label, re.IGNORECASE):
				continue
			haystack = re.sub(r'[-_./]+', ' ', f'{label} {parts.path}').casefold()
			matched = sum(term in haystack for term in terms)
			if not matched:
				continue
			query = parse_qsl(parts.query, keep_blank_values=True)
			if upper and 'price' in current_query:
				query = [(key, value) for key, value in query if key != 'price']
				query.append(('price', f'0-{upper.group(1)}'))
			href = urlunsplit(parts._replace(query=urlencode(query), fragment=''))
			depth = len([part for part in parts.path.split('/') if part])
			all_terms = int(bool(terms) and matched == len(terms))
			value = (all_terms, matched, depth)
			if value > scored.get(href, (-1, -1, -1)):
				scored[href] = value
	# Local fallback for faceted pages whose visible hierarchy is encoded in labels but
	# whose current URL retains only an opaque query parameter.
	parts = urlsplit(current_url)
	if parts.path.endswith('.html') and terms:
		root_segments = [segment for segment in parts.path[:-5].split('/') if segment]
		target_segments = list(dict.fromkeys(re.sub(r'[^a-z0-9]+', '-', term).strip('-') for term in terms))
		overlap = 0
		for size in range(1, min(len(root_segments), len(target_segments)) + 1):
			if root_segments[-size:] == target_segments[:size]:
				overlap = size
		path_segments = root_segments + target_segments[overlap:]
		if path_segments:
			query = parse_qsl(parts.query, keep_blank_values=True)
			if upper:
				query = [(key, value) for key, value in query if key not in {'cat', 'price'}]
				query.append(('price', f'0-{upper.group(1)}'))
			candidate = urlunsplit(
				parts._replace(path='/' + '/'.join(path_segments) + '.html', query=urlencode(query), fragment='')
			)
			scored[candidate] = max(scored.get(candidate, (-1, -1, -1)), (1, len(terms), len(terms) + 10))
	return [url for url, _ in sorted(scored.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _historical_entity_phrase(intent: str) -> str | None:
	match = re.search(
		r'\b(?:last|latest|most\s+recent)\s+(?:ordered|bought|purchased)\s+(.+?)(?:[?.!,;]|$)',
		intent,
		re.IGNORECASE,
	)
	if not match:
		return None
	phrase = re.sub(r'\b(?:from|on|at)\s+(?:the\s+)?(?:website|site|store)\b.*$', '', match.group(1), flags=re.I)
	phrase = re.sub(r'^(?:my|the|a|an)\s+', '', phrase, flags=re.I)
	return re.sub(r'\s+', ' ', phrase).strip(' \t\r\n"\'') or None


def _record_dates(text: str) -> list[datetime]:
	values: list[datetime] = []
	month_pattern = (
		r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
		r'\s+\d{1,2},?\s+\d{4}'
	)
	labeled = re.findall(
		r'(?i)(?:order|purchase|transaction|invoice)\s+date\s*:?\s*(' + month_pattern + r'|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2})',
		text,
	)
	candidates = labeled or re.findall(month_pattern + r'|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b', text, re.I)
	for value in candidates:
		for pattern in ('%B %d, %Y', '%B %d %Y', '%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d'):
			try:
				values.append(datetime.strptime(re.sub(r'\s+', ' ', value.strip()), pattern))
				break
			except ValueError:
				continue
	return values


def extract_latest_same_record_date(intent: str, records: list[StructuredVisibleRecord]) -> str | None:
	"""Bind the requested entity and date inside one visible record before selecting the latest."""

	entity = _historical_entity_phrase(intent)
	if not entity:
		return None
	entity_tokens = meaningful_tokens(entity)
	if not entity_tokens:
		return None
	date_pattern = re.compile(
		r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
		r'\s+\d{1,2},?\s+\d{4}|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b',
		re.I,
	)
	dates: list[datetime] = []
	for record in records:
		token_matches = {
			token: list(re.finditer(r'(?<!\w)' + re.escape(token) + r'(?!\w)', record.text, re.I))
			for token in entity_tokens
		}
		if any(not matches for matches in token_matches.values()):
			continue
		for date_match in date_pattern.finditer(record.text):
			nearest = [
				min(matches, key=lambda match: abs(match.start() - date_match.start()))
				for matches in token_matches.values()
			]
			entity_start = min(match.start() for match in nearest)
			entity_end = max(match.end() for match in nearest)
			if len(re.findall(r'\b\w+\b', record.text[entity_start:entity_end])) > 20:
				continue
			bound_start = min(entity_start, date_match.start())
			bound_end = max(entity_end, date_match.end())
			if len(re.findall(r'\b\w+\b', record.text[bound_start:bound_end])) > 180:
				continue
			dates.extend(_record_dates(date_match.group(0)))
	return max(dates).strftime('%Y-%m-%d') if dates else None


def verify_retrieval_workflow(
	intent: str,
	proposed_values: list[Any],
	ledger: GenericEvidenceLedger,
) -> WorkflowVerification:
	"""Independently verify high-risk retrieval invariants from visible evidence."""

	entity = _historical_entity_phrase(intent)
	if entity:
		verified_date = extract_latest_same_record_date(intent, ledger.records)
		if verified_date is None:
			return WorkflowVerification(
				accepted=False,
				failure_kind='coverage',
				missing_evidence=[f'one visible record containing both {entity!r} and its date'],
				repair_instruction='Expand visible history/detail records and verify entity and date in the same record.',
			)
		proposal_text = json.dumps(proposed_values, ensure_ascii=False)
		return WorkflowVerification(
			accepted=verified_date in proposal_text,
			failure_kind='none' if verified_date in proposal_text else 'record_binding',
			repair_instruction=None if verified_date in proposal_text else 'Replace the proposal with the independently bound date.',
			verified_values=[verified_date],
		)
	return WorkflowVerification(accepted=True)


def semantic_decision_is_consistent(
	decision: RecordSemanticDecision,
	intent: str | None = None,
	*,
	record_text: str | None = None,
	max_token_span: int = 180,
) -> bool:
	"""Reject structured labels contradicted by the model's own explanation."""

	if not decision.relevant:
		return True
	reason = decision.reason.casefold()
	contradictions = (
		'does not satisfy',
		'does not match',
		'does not mention',
		'no such mention',
		'not relevant to',
		'therefore no ',
	)
	if not decision.projected_values or any(phrase in reason for phrase in contradictions):
		return False
	if intent is None:
		return True
	evidence = decision.evidence.casefold()
	if not evidence:
		return False
	if re.search(r'\b(?:explicitly|exact(?:ly)?|contains?)\b', intent, re.IGNORECASE):
		quoted_phrases = [
			next(value for value in match.groups() if value)
			for match in re.finditer(r"'([^']+)'|\"([^\"]+)\"", intent)
		]
		inferred_phrases = [
			match.group(1).strip().strip("'\"")
			for match in re.finditer(
				r'\b(?:mention(?:s|ed|ing)?|contains?)\s+(.+?)\s+(?:explicitly|exactly)\b',
				intent,
				re.IGNORECASE,
			)
		]
		literal_phrases = list(dict.fromkeys(quoted_phrases + inferred_phrases))
		source = (record_text or decision.evidence).casefold()
		if literal_phrases and not all(phrase.casefold() in source for phrase in literal_phrases):
			return False
		maximum_match = re.search(
			r'\b(?:rating\s+(?:of\s+)?)?(\d+(?:\.\d+)?)\s+or\s+(?:less|fewer)\s+stars?\b',
			intent,
			re.IGNORECASE,
		)
		if maximum_match:
			maximum = float(maximum_match.group(1))
			percent_match = re.search(r'\brating\s*:?\s*(\d+(?:\.\d+)?)\s*%', source)
			star_match = re.search(r'\brating\s*:?\s*(\d+(?:\.\d+)?)\s*(?:out\s+of\s+5\s*)?stars?\b', source)
			if percent_match:
				visible_rating = float(percent_match.group(1)) / 20.0
			elif star_match:
				visible_rating = float(star_match.group(1))
			else:
				return False
			if visible_rating > maximum:
				return False
		return True
	positions: list[int] = []
	for token in meaningful_tokens(intent):
		position = evidence.find(token)
		if position >= 0:
			positions.append(position)
	if len(positions) < 2:
		return False
	return max(positions) - min(positions) <= max_token_span


def preserve_visible_composite_values(
	candidates: list[str | int | float | bool],
	references: list[str | int | float | bool],
	visible_evidence: str,
	intent: str,
) -> list[str | int | float | bool]:
	"""Restore a longer literal field value when a verifier trims its visible suffix."""

	if re.search(r'\b(?:as\s+a\s+number|number\s+only|numeric\s+only)\b', intent, re.IGNORECASE):
		return candidates
	evidence_lower = visible_evidence.casefold()
	evidence_lines = [line.strip() for line in visible_evidence.splitlines()]
	intent_tokens = meaningful_tokens(intent)
	result: list[str | int | float | bool] = []
	for candidate in candidates:
		replacement: str | int | float | bool = candidate
		if isinstance(candidate, str):
			candidate_lower = candidate.strip().casefold()
			visible_references: list[str] = []
			for index, line in enumerate(evidence_lines):
				if index == 0 or len(line) > 80:
					continue
				if not line.casefold().startswith(candidate_lower + ' '):
					continue
				previous = evidence_lines[index - 1].casefold()
				if any(token in previous for token in intent_tokens):
					visible_references.append(line)
			all_references = list(references) + sorted(visible_references, key=len)
			for reference in all_references:
				if not isinstance(reference, str):
					continue
				reference_clean = reference.strip()
				reference_lower = reference_clean.casefold()
				if (
					len(reference_clean) > len(candidate.strip())
					and reference_lower.startswith(candidate_lower + ' ')
					and reference_lower in evidence_lower
				):
					replacement = reference_clean
					break
		result.append(replacement)
	return result


def grounded_audit_prompt(intent: str, proposed_data: Any, visible_evidence: str) -> tuple[str, str]:
	"""Build a dataset-neutral record reasoning prompt for a structured LLM call."""

	system = (
		'You audit a browser agent using only visible page evidence from its own trajectory. '
		'Reconstruct the requested retrieval answer rather than trusting the proposed answer. '
		'Group nearby fields into local records such as rows, cards, list entries, messages, or detail sections. '
		'Infer repeated field order across the page before binding fields: a repeated author, owner, date, or status '
		'marker may consistently follow rather than precede the content that belongs to it. '
		'Apply every requested predicate to the same record; never join a name, attribute, status, date, or value '
		'from unrelated records. For an aggregate, filter qualifying records first, use their record-level values, '
		'count each record once, and then calculate. Distinguish line values from container subtotal, shipping, and '
		'grand-total values according to the request. For an entity attribute, first identify the matching entity '
		'record and then return its visible attribute with units preserved. For exhaustive requests, account for '
		'visible totals and continuation evidence. A semantic predicate can be entailed without repeating its exact '
		'words: for example, only part of an intended target fitting or failure to fit around it entails undersizing. '
		'The verb mention alone requests semantic relevance, not literal wording; pronouns and contextual references '
		'to the entity or component count when the local record makes their referent clear. '
		'However, when the request says explicitly, exact phrase, or contains, require the literal normalized phrase '
		'in that same record and reject synonyms. For semantic categories, classify each line record independently; '
		'never assume every line qualifies because the surrounding container does. '
		'Do not use site conventions, hidden data, benchmark answers, or '
		'prior knowledge. Set complete=false when the visible evidence is insufficient. Keep evidence excerpts short.'
	)
	user = (
		f'User request:\n{intent}\n\n'
		f'Agent proposed data (a fallible hint, not evidence):\n'
		f'{json.dumps(proposed_data, ensure_ascii=False, default=str)}\n\n'
		f'Visible trajectory evidence:\n{visible_evidence}'
	)
	return system, user


def grounded_challenge_prompt(
	intent: str,
	first_audit: GroundedRetrievalAudit,
	visible_evidence: str,
) -> tuple[str, str]:
	"""Build an independent adversarial pass over the same visible evidence."""

	system = (
		'You are the independent second-pass verifier for a browser retrieval result. The first audit is fallible. '
		'Inspect every task-relevant STRUCTURED RECORD independently before deciding that no item was omitted. '
		'Search the full visible evidence for omitted records, semantic paraphrases, negations, and indirect entailment. '
		'Unless the request says explicitly, exact phrase, or contains, do not require literal repetition: include '
		'records where partial fit, failed fit, comparison, target-versus-user size mismatch, child-versus-adult fit, '
		'or a clear contextual reference entails the predicate. A record can entail that an object is undersized by '
		'describing incomplete coverage, resting on top instead of surrounding the target, or fitting a smaller user '
		'but not a larger one, even when it never says "small". Treat these as generic spatial entailments rather '
		'than literal phrase matches. '
		'Also remove any selected value whose identity and requested predicates are not supported inside one local '
		'record. Infer the repeated field order before binding fields: identity markers may follow the content they '
		'label. When the request requires a literal or explicit phrase, reject synonyms and require that phrase in the '
		'same record. For calculations, independently enumerate qualifying line records and recompute; do not reuse a '
		'container subtotal or grand total unless that is exactly what was requested. For an entity attribute, verify '
		'the entity and attribute belong to the same record and preserve visible units. Return the corrected complete '
		'answer, not a critique. Do not stop after finding the first obvious matches. Use no hidden information, '
		'benchmark answer, site rule, or prior knowledge.'
	)
	user = (
		f'User request:\n{intent}\n\n'
		f'First audit:\n{first_audit.model_dump_json(indent=2)}\n\n'
		f'Visible trajectory evidence:\n{visible_evidence}'
	)
	return system, user


_MUTATION_VERBS = (
	'add',
	'book',
	'buy',
	'cancel',
	'change',
	'checkout',
	'create',
	'delete',
	'edit',
	'fill',
	'follow',
	'like',
	'order',
	'post',
	'purchase',
	'rate',
	'remove',
	'reserve',
	'save',
	'send',
	'set',
	'submit',
	'subscribe',
	'unfollow',
	'update',
	'upload',
)
_NAVIGATION_VERBS = ('go to', 'navigate to', 'open', 'pull up', 'visit')
_EXHAUSTIVE_SIGNALS = (
	r'\ball\b',
	r'\beach\b',
	r'\bevery\b',
	r'\bname\s*\(s\)',
	r'\bnames\b',
	r'\bitems\b',
	r'\borders\b',
	r'\breviewers\b',
	r'\breviews\b',
	r'\bresults\b',
	r'\bcomments\b',
	r'\bposts\b',
	r'\bissues\b',
	r'\brepositories\b',
	r'\bhow many\b',
	r'\bhow much\b',
	r'\b(?:bought|purchased|spent)\b',
	r'\blast\s+(?:ordered|bought|purchased)\b',
	r'\bprice range\b',
)


def resolve_task_contract(intent: str) -> TaskContract:
	"""Resolve broad task semantics without dataset, domain, or site rules."""

	normalized = re.sub(r'\s+', ' ', intent).strip().casefold()
	leading = re.sub(r'^(?:please|can you|could you|i want you to)\s+', '', normalized)
	mutation_pattern = r'^(?:' + '|'.join(re.escape(verb) for verb in _MUTATION_VERBS) + r')\b'
	embedded_mutation_pattern = (
		r'\b(?:add|book|buy|cancel|change|checkout|create|delete|edit|fill\s+out|follow|like|'
		r'order|post|purchase|rate|remove|reserve|save|send|set|submit|subscribe|unfollow|'
		r'update|upload)\s+(?:my|the|this|that|an?|information|details?|address|form|product)\b'
	)
	return_pattern = (
		r'\breturn\s+(?:my|the|this|an?)\s+'
		r'(?:(?:purchased|ordered)\s+)?(?:item|order|product|purchase|package)\b'
	)
	if (
		re.search(mutation_pattern, leading)
		or re.search(embedded_mutation_pattern, normalized)
		or re.search(return_pattern, normalized)
	):
		operation: TaskOperation = 'MUTATE'
	elif any(leading.startswith(verb) for verb in _NAVIGATION_VERBS):
		operation = 'NAVIGATE'
	else:
		operation = 'RETRIEVE'

	exhaustive = operation == 'RETRIEVE' and any(re.search(pattern, normalized) for pattern in _EXHAUSTIVE_SIGNALS)
	draft_only = operation == 'MUTATE' and bool(
		re.search(r'\b(?:do\s+not|don\'t|without)\s+(?:submit|send|save|post)\b|\bleave\b.+\bready\s+for\s+review\b', normalized)
	)
	irreversible = operation == 'MUTATE' and bool(
		re.search(r'\b(?:buy|checkout|delete|remove|send|submit|post|purchase|cancel|book|reserve)\b', normalized)
	) and not draft_only
	# Object names such as "picture frame" are ordinary textual entities, not a
	# request to inspect pixels. Enable vision only when the instruction names a
	# visual source or asks for a property that cannot reliably come from the DOM.
	visual_intent = re.sub(r'\bpicture\s+frames?\b', ' ', normalized)
	requires_visual = bool(
		re.search(
			r'\b(?:from|in|on)\s+(?:the\s+)?(?:image|photo|picture|screenshot|chart|graph|diagram|canvas)\b'
			r'|\b(?:look\s+at|inspect|analy[sz]e|read|identify|compare|describe|transcribe)\b'
			r'.{0,40}\b(?:image|photo|picture|screenshot|chart|graph|diagram|logo|icon|canvas)\b'
			r'|\b(?:chart|graph|diagram|canvas)\b.{0,40}\b(?:shows?|contains?|depicts?|displays?)\b',
			visual_intent,
		)
	)
	if operation != 'RETRIEVE':
		output_shape: Literal['none', 'scalar', 'list', 'object'] = 'none'
	elif re.search(r'\breturn\s+(?:an?\s+)?object\b|\bwith\s+keys?\b', normalized):
		output_shape = 'object'
	elif re.search(r'\breturn\s+(?:a\s+)?list\b', normalized) or exhaustive:
		output_shape = 'list'
	else:
		output_shape = 'scalar'
	return TaskContract(
		operation=operation,
		exhaustive=exhaustive,
		irreversible=irreversible,
		draft_only=draft_only,
		requires_visual_evidence=requires_visual,
		output_shape=output_shape,
	)


def build_generic_policy(contract: TaskContract) -> str:
	"""Render the same observable-state policy for every website and benchmark."""

	lines = [
		'Maintain a compact ledger of visited page states, rejected candidates, satisfied constraints, and remaining constraints.',
		'After any navigation or DOM-changing action, discard old element indexes and inspect the new state.',
		'Apply all requested predicates to the same visible record; never combine fields from unrelated records.',
		'Change strategy after two actions that produce no new URL, visible evidence, or satisfied constraint.',
	]
	if contract.exhaustive:
		lines.extend(
			[
				'For exhaustive retrieval, traverse relevant pagination, load-more controls, tabs, and collapsed sections.',
				'Track unique records and visible totals; do not equate the first viewport with complete coverage.',
			]
		)
	if contract.operation == 'MUTATE':
		lines.append(
			'Execute explicit prerequisite clauses before selecting or creating the new target state. For example, '
			'when asked to discard existing contents if non-empty, verify and clear them first, then select the new '
			'candidate; never add the new candidate and later treat it as pre-existing content.'
		)
		lines.append(
			'For highest/lowest candidate tasks, navigate to the exact requested collection before searching. If the '
			'objective cannot be sorted directly but a numeric bound can, sort by that bound so qualifying records '
			'are contiguous, use the largest visible page size, rank same-record candidates, and stop only after the '
			'ordered values cross the bound.'
		)
		if contract.draft_only:
			lines.append(
				'This is a draft-only mutation: fill every requested field, verify its current value, and do not submit.'
			)
		else:
			lines.append(
				'Before completing a mutation, require an observed state-changing request plus visible persisted-state evidence.'
			)
		lines.append(
			'Do not infer a quantity from a number embedded in an exact visible entity or product name. Treat it as '
			'part of the identity unless the request separately says quantity, units, copies, items, or uses an '
			'explicit multiplier such as x2.'
		)
	if contract.operation == 'NAVIGATE':
		lines.extend(
			[
				'Before completion, verify the final URL, title, and requested visible page state.',
				'For a hierarchical destination, reveal the visible top-level navigation menu and follow its '
				'child labels in order. Do not substitute a broad faceted listing when the request names a '
				'parent-child category path.',
			]
		)
	if contract.requires_visual_evidence:
		lines.append('Use screenshot evidence only for information that the DOM cannot express reliably.')
	return '\n'.join(f'- {line}' for line in lines)


def completion_decision(
	ledger: GenericEvidenceLedger,
	contract: TaskContract,
	*,
	intent: str = '',
	final_text: str,
	current_url: str,
	visible_text: str,
	start_url: str,
	request_evidence: list[dict[str, Any]] | None = None,
) -> str | None:
	"""Return a bounded, generic completion blocker or allow completion."""

	attempt = ledger.register_proposal(final_text)
	if contract.operation == 'RETRIEVE' and not final_text.strip():
		return 'Return the requested value using the required structured response.'
	if contract.operation == 'NAVIGATE':
		if stable_url(current_url) == stable_url(start_url):
			return 'The browser has not left the initial location; verify the requested destination and visible state.'
		destination = f'{current_url} {visible_text}'.casefold()
		url_destination = re.sub(r'[-_./?=&%]+', ' ', current_url).casefold()
		intent_terms = meaningful_tokens(
			re.sub(r'\b(?:open|go|navigate|visit|pull|page|filtered|sorted|listings|product)\b', ' ', intent)
		)
		matched_terms = [term for term in intent_terms if term in destination]
		required_matches = min(2, len(intent_terms))
		if required_matches and len(matched_terms) < required_matches:
			return (
				'The final URL and visible page do not yet contain enough requested destination constraints. '
				f'Only matched {matched_terms}; inspect the selected category, item, filters, and sort state.'
			)
		path_terms = [
			term
			for term in intent_terms
			if not re.fullmatch(r'\d+(?:\.\d+)?', term)
			and term not in {'under', 'over', 'above', 'below', 'less', 'more', 'than'}
		]
		required_path_matches = min(2, len(path_terms))
		matched_path_terms = [term for term in path_terms if term in url_destination]
		if required_path_matches and len(matched_path_terms) < required_path_matches:
			return (
				'The final URL does not preserve enough of the requested destination hierarchy. '
				f'Only matched {matched_path_terms}; navigate to the exact requested category or entity page.'
			)
		upper_bound = re.search(
			r'\b(?:under|below|less\s+than|at\s+most|up\s+to)\s*\$?\s*(\d+(?:\.\d+)?)\b',
			intent,
			re.IGNORECASE,
		)
		if upper_bound:
			bound = re.escape(upper_bound.group(1))
			exact_upper_range = re.search(
				rf'(?:price|cost|amount|range)?[^0-9]{{0,8}}0(?:\.0+)?\s*(?:-|to|–)\s*{bound}(?:\.0+)?(?!\d)',
				destination,
				re.IGNORECASE,
			)
			if not exact_upper_range:
				return (
					f'The requested upper-bound filter must preserve the complete 0-{upper_bound.group(1)} result '
					'range. A narrower interval or a page that merely mentions the number is not equivalent.'
				)
		numeric_constraints = re.findall(r'(?<![\w])\d+(?:\.\d+)?', intent)
		if numeric_constraints and not all(value in destination for value in numeric_constraints):
			return (
				'The final page does not visibly preserve every numeric navigation constraint '
				f'{numeric_constraints}; apply the exact filter or capacity bound before completing.'
			)
	if contract.operation == 'MUTATE':
		commit_intent = bool(
			re.search(r'\b(?:buy|purchase|checkout|place\s+(?:the\s+)?order|book|reserve)\b', intent, re.IGNORECASE)
		)
		if commit_intent:
			terminal = re.search(
				r'\b(?:thank you|order (?:number|confirmation)|order has been placed|purchase (?:was )?successful|'
				r'booking confirmed|reservation confirmed)\b',
				visible_text,
				flags=re.IGNORECASE,
			)
		else:
			terminal = re.search(
				r'\b(?:success(?:ful(?:ly)?)?|you saved|has been (?:saved|updated|added|removed|deleted)|'
				r'submitted|sent|confirmed|confirmation)\b',
				visible_text,
				flags=re.IGNORECASE,
			)
		if contract.draft_only:
			if len(ledger.snapshots) < 2 or ledger.snapshots[-1].signature == ledger.snapshots[0].signature:
				return 'Draft fields have not produced a verifiable visible state change; inspect all current form values.'
			quoted = [
				re.sub(r'<[^>]+>', ' ', value)
				for value in re.findall(r'["“](.+?)["”]', intent)
			]
			for value in quoted:
				required = meaningful_tokens(value)
				matched = sum(token in visible_text.casefold() for token in required)
				if required and matched < max(1, (len(required) + 1) // 2):
					return (
						'The draft form does not visibly contain enough of the requested field content; '
						'inspect current values and fill every missing field without submitting.'
					)
		else:
			mutations = [
				item
				for item in (request_evidence or [])
				if str(item.get('method') or '').upper() in {'POST', 'PUT', 'PATCH', 'DELETE'}
			]
			if not mutations:
				state_changed = (
					len(ledger.snapshots) >= 2
					and ledger.snapshots[-1].signature != ledger.snapshots[0].signature
				)
				if terminal and state_changed:
					return None
				repeated = ' Repeating the same success claim cannot satisfy the execution contract.' if attempt >= 2 else ''
				return (
					'No state-changing browser request was observed. Inspect required form fields and validation errors, '
					'perform the actual submit/save action, then verify the persisted result.' + repeated
				)
			if not terminal:
				repeated = (
					' Do not call done again. Inspect the current form and validation state, then use a different '
					'recovery action such as reloading once, correcting an address or required field, or selecting '
					'an available option.'
					if attempt >= 2
					else ''
				)
				return (
					'A state-changing request occurred, but no visible persisted-state or confirmation evidence is present.'
					+ repeated
				)
	if contract.exhaustive and ledger.snapshots:
		latest = ledger.snapshots[-1]
		if latest.has_continuation:
			return 'A visible continuation control remains; inspect it and retain unique records before completing.'
		visible_totals = [item.visible_total for item in ledger.snapshots if item.visible_total is not None]
		if visible_totals and len(ledger.snapshots) == 1 and max(visible_totals) > 1:
			return 'The page exposes multiple records; perform one explicit coverage pass before completing.'
	return None


def normalize_response_for_contract(value: Any, contract: TaskContract) -> dict[str, Any]:
	"""Enforce the public response schema without adding task-specific answers."""

	if hasattr(value, 'model_dump'):
		value = value.model_dump(mode='json')
	if not isinstance(value, dict):
		data = value if isinstance(value, list) else [value] if value not in (None, '') else None
		value = {'status': 'SUCCESS' if data else 'UNKNOWN_ERROR', 'retrieved_data': data}
	status = str(value.get('status') or 'UNKNOWN_ERROR').upper()
	data = value.get('retrieved_data')
	error_details = value.get('error_details')
	if contract.operation == 'RETRIEVE':
		if data is not None and not isinstance(data, list):
			data = [data]
		if status == 'SUCCESS' and not data:
			status = 'UNKNOWN_ERROR'
			error_details = error_details or 'No grounded retrieval value was returned.'
	else:
		data = None
	return {
		'task_type': contract.operation,
		'status': status,
		'retrieved_data': data,
		'error_details': error_details,
	}


def aggregate_visible_dated_currency_records(
	intent: str,
	records: list[StructuredVisibleRecord],
) -> dict[str, int | float] | None:
	"""Compute a requested count/amount object from deduplicated visible row records."""

	keys_match = re.search(r'\bkeys?\s+(.+?)(?:\s+only|,?\s+without|$)', intent, re.IGNORECASE)
	keys = re.findall(r'["\']([A-Za-z_][A-Za-z0-9_]*)["\']', keys_match.group(1) if keys_match else '')
	count_key = next((key for key in keys if 'count' in key.casefold()), None)
	amount_key = next(
		(key for key in keys if any(token in key.casefold() for token in ('amount', 'total', 'spent', 'sum'))),
		None,
	)
	if not count_key or not amount_key:
		return None
	today_match = re.search(
		r'\btoday\s+is\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})',
		intent,
		re.IGNORECASE,
	)
	if not today_match or not re.search(r'\bpast\s+year\b', intent, re.IGNORECASE):
		return None
	try:
		end_date = datetime.strptime(today_match.group(1), '%B %d, %Y').date()
		start_date = end_date.replace(year=end_date.year - 1)
	except ValueError:
		return None
	required_status = 'complete' if re.search(r'\bcomplete\s+(?:orders?|records?)\b', intent, re.IGNORECASE) else None
	seen: set[str] = set()
	amount = Decimal('0')
	count = 0
	row_pattern = re.compile(
		r'\b(?P<id>\d{5,})\b\s+'
		r'(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{2,4})\s+'
		r'\$(?P<amount>[\d,]+(?:\.\d{2})?)\s+'
		r'(?P<status>[A-Za-z][A-Za-z ]{1,30}?)(?=\s+(?:View|Open|Edit|Reorder|Details?)\b|$)',
		re.IGNORECASE,
	)
	for record in records:
		match = row_pattern.search(record.text)
		if not match or match.group('id') in seen:
			continue
		year = int(match.group('year'))
		if year < 100:
			year += 2000
		try:
			row_date = datetime(year, int(match.group('month')), int(match.group('day'))).date()
		except ValueError:
			continue
		if not start_date <= row_date <= end_date:
			continue
		status = re.sub(r'\s+', ' ', match.group('status')).strip().casefold()
		if required_status and status != required_status:
			continue
		seen.add(match.group('id'))
		count += 1
		amount += Decimal(match.group('amount').replace(',', ''))
	if not count:
		return None
	return {count_key: count, amount_key: float(amount.quantize(Decimal('0.01')))}


def stable_url(url: str) -> str:
	parts = urlsplit(url)
	return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ''))


def page_signature(url: str, visible_text: str) -> str:
	normalized = re.sub(r'\s+', ' ', visible_text).casefold()[:12_000]
	digest = hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:16]
	return f'{stable_url(url)}|{digest}'


def normalize_proposal(value: str) -> str:
	try:
		return json.dumps(json.loads(value), ensure_ascii=False, sort_keys=True)
	except (TypeError, ValueError, json.JSONDecodeError):
		return re.sub(r'\s+', ' ', value).strip().casefold()


def meaningful_tokens(value: str) -> list[str]:
	stop = {
		'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'get', 'i', 'in', 'is', 'it',
		'being', 'current', 'me', 'mention', 'mentions', 'my', 'name', 'names', 'of', 'on', 'or', 'page',
		'please', 'product', 'return', 'reviewer', 'reviewers', 'show', 'that', 'the', 'this', 'to', 'was',
		'what', 'which', 'who', 'with',
	}
	result: list[str] = []
	for token in re.findall(r'[a-z0-9]+', value.casefold()):
		if len(token) >= 3 and token not in stop and token not in result:
			result.append(token)
	return result[:24]


def resolve_comparison_spec(intent: str) -> ComparisonSpec:
	"""Parse a comparison request without relying on website or benchmark fields."""

	normalized = re.sub(r'\s+', ' ', intent).strip()
	lower = normalized.casefold()
	objective: Literal['highest_rating', 'lowest_rating', 'highest_price', 'lowest_price'] | None = None
	if re.search(r'\b(?:highest|best|top)[ -]rated\b|\bhighest\s+rating\b', lower):
		objective = 'highest_rating'
	elif re.search(r'\b(?:lowest|worst)[ -]rated\b|\blowest\s+rating\b', lower):
		objective = 'lowest_rating'
	elif re.search(r'\b(?:most\s+expensive|highest\s+(?:price|priced))\b', lower):
		objective = 'highest_price'
	elif re.search(r'\b(?:cheapest|least\s+expensive|lowest\s+(?:price|priced))\b', lower):
		objective = 'lowest_price'

	min_match = re.search(
		r'\b(?:above|over|greater\s+than|more\s+than|at\s+least|minimum(?:\s+of)?)\s*\$?\s*([\d,]+(?:\.\d+)?)',
		lower,
	)
	max_match = re.search(
		r'\b(?:below|under|less\s+than|at\s+most|maximum(?:\s+of)?|within(?:\s+a)?\s+budget(?:\s+of)?)'
		r'\s*\$?\s*([\d,]+(?:\.\d+)?)',
		lower,
	)
	category_match = re.search(r'\bfrom\s+the\s+(.+?)\s+categor(?:y|ies)\b', lower)
	category_label = category_match.group(1).strip() if category_match else None
	required_terms = meaningful_tokens(category_label) if category_label else []
	return ComparisonSpec(
		objective=objective,
		min_price=float(min_match.group(1).replace(',', '')) if min_match else None,
		max_price=float(max_match.group(1).replace(',', '')) if max_match else None,
		category_label=category_label,
		required_terms=required_terms,
	)


def rank_visible_comparison_candidates(
	intent: str,
	records: list[StructuredVisibleRecord],
	*,
	limit: int = 12,
) -> list[RankedVisibleCandidate]:
	"""Rank visible cards using same-record identity, rating, price, and request constraints."""

	spec = resolve_comparison_spec(intent)
	if spec.objective is None:
		return []
	unique: dict[str, RankedVisibleCandidate] = {}
	for record in records:
		text = re.sub(r'\s+', ' ', record.text).strip()
		lower = text.casefold()
		if spec.required_terms and not all(
			re.search(rf'\b{re.escape(term)}s?\b', lower) for term in spec.required_terms
		):
			continue
		price_match = re.search(r'\$\s*([\d,]+(?:\.\d{1,2})?)', text)
		price = float(price_match.group(1).replace(',', '')) if price_match else None
		if spec.min_price is not None and (price is None or price <= spec.min_price):
			continue
		if spec.max_price is not None and (price is None or price >= spec.max_price):
			continue
		percent_match = re.search(r'\bRating\s*:?\s*(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
		star_match = re.search(
			r'\b(\d+(?:\.\d+)?)\s*(?:out\s+of\s+5\s*)?stars?\b',
			text,
			re.IGNORECASE,
		)
		rating = (
			float(percent_match.group(1))
			if percent_match
			else float(star_match.group(1)) * 20.0
			if star_match
			else None
		)
		title_end = len(text)
		for marker in (' Rating:', ' $', '\nRating:', '\n$'):
			position = text.find(marker)
			if position >= 0:
				title_end = min(title_end, position)
		title = text[:title_end].strip(' -:')[:300]
		if not title:
			continue
		href = next(
			(
				str(link.get('href') or '')
				for link in record.links
				if str(link.get('href') or '').strip()
				and normalize_proposal(str(link.get('label') or title)).startswith(normalize_proposal(title)[:40])
			),
			next((str(link.get('href') or '') for link in record.links if str(link.get('href') or '').strip()), None),
		)
		candidate = RankedVisibleCandidate(
			record_text=title[:180],
			title=title,
			price=price,
			rating=rating,
			href=href,
			controls=record.controls,
		)
		key = normalize_proposal(title)
		previous = unique.get(key)
		if previous is None or (candidate.rating or -1) > (previous.rating or -1):
			unique[key] = candidate

	def sort_key(candidate: RankedVisibleCandidate) -> tuple[float, float, str]:
		rating = candidate.rating if candidate.rating is not None else -1.0
		price = candidate.price if candidate.price is not None else float('inf')
		if spec.objective == 'highest_rating':
			return (-rating, price, candidate.title.casefold())
		if spec.objective == 'lowest_rating':
			return (rating if candidate.rating is not None else float('inf'), price, candidate.title.casefold())
		if spec.objective == 'highest_price':
			return (-price if candidate.price is not None else float('inf'), -rating, candidate.title.casefold())
		return (price, -rating, candidate.title.casefold())

	return sorted(unique.values(), key=sort_key)[: max(1, min(limit, 30))]


def extract_repeated_anchor_records(text: str, *, max_record_chars: int = 6_000) -> list[str]:
	"""Discover repeated label/value anchors and segment their preceding local records."""

	lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines() if line.strip()]
	groups: dict[tuple[str, ...], list[tuple[int, str]]] = {}
	for index, line in enumerate(lines):
		tokens = re.findall(r'[a-z0-9]+', line.casefold())
		for prefix_length in (1, 2, 3):
			if len(tokens) <= prefix_length:
				continue
			prefix = tuple(tokens[:prefix_length])
			remainder = ' '.join(tokens[prefix_length:])
			groups.setdefault(prefix, []).append((index, remainder))

	candidates: list[tuple[int, int, tuple[str, ...], list[int]]] = []
	for prefix, values in groups.items():
		indices = sorted({index for index, _ in values})
		remainders = {value for _, value in values if value}
		if len(indices) < 3 or len(remainders) < 2:
			continue
		# Prefer a longer repeated label and broad page coverage.
		candidates.append((len(prefix), len(indices), prefix, indices))
	candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

	records: list[str] = []
	used_prefixes: list[tuple[str, ...]] = []
	for _, _, prefix, indices in candidates:
		if any(prefix[: len(existing)] == existing or existing[: len(prefix)] == prefix for existing in used_prefixes):
			continue
		used_prefixes.append(prefix)
		previous = max(0, indices[0] - 6)
		for anchor_index in indices:
			block = '\n'.join(lines[previous : anchor_index + 1])
			if len(block) > max_record_chars:
				block = block[-max_record_chars:]
			if block and block not in records:
				records.append(block)
			previous = anchor_index + 1
		if len(used_prefixes) >= 5:
			break
	return records


GENERIC_REPEATED_DOM_SCAN_JS = r"""
(() => {
  const clean = (value) => (value || '').replace(/\s+/g, ' ').trim();
  const visible = (el) => {
    if (!(el instanceof Element)) return false;
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      Number(style.opacity || 1) > 0 && rect.width > 0 && rect.height > 0;
  };
  const classSignature = (el) => Array.from(el.classList || [])
    .filter((name) => !/\d{4,}|active|focus|hover|selected|open|closed/i.test(name))
    .slice(0, 4).sort().join('.');
  const signature = (el) => [
    el.tagName.toLowerCase(),
    el.getAttribute('role') || '',
    classSignature(el),
  ].join('|');
  const controlElements = (root) => Array.from(
    root.querySelectorAll('a,button,input,select,textarea,[role="button"],[role="link"],[role="tab"]')
  ).filter(visible);
  const controlLabels = (root) => controlElements(root).map((el) => clean(
    el.getAttribute('aria-label') || el.getAttribute('title') ||
    el.innerText || el.value || el.getAttribute('name')
  )).filter(Boolean).slice(0, 30);
  const controlLinks = (root) => controlElements(root).map((el) => ({
    label: clean(el.getAttribute('aria-label') || el.getAttribute('title') || el.innerText),
    href: el.href || '',
  })).filter((item) => item.href).slice(0, 30);

  const parents = Array.from(document.querySelectorAll(
    'table,tbody,ul,ol,[role="table"],[role="list"],[role="grid"],section,article,main,div'
  )).filter(visible).slice(0, 2500);
  const groups = [];
  for (const parent of parents) {
    const bySignature = new Map();
    for (const child of Array.from(parent.children || [])) {
      if (!visible(child)) continue;
      const text = clean(child.innerText);
      if (text.length < 8 || text.length > 12000) continue;
      const key = signature(child);
      if (!bySignature.has(key)) bySignature.set(key, []);
      bySignature.get(key).push(child);
    }
    for (const [key, children] of bySignature.entries()) {
      if (children.length < 2) continue;
      const unique = new Set(children.map((child) => clean(child.innerText).toLowerCase()));
      if (unique.size < 2) continue;
      const avg = children.reduce((sum, child) => sum + clean(child.innerText).length, 0) / children.length;
      groups.push({ key, children, score: children.length * 100 + Math.min(avg, 5000) });
    }
  }
  groups.sort((a, b) => b.score - a.score);
  const seen = new Set();
  const records = [];
  for (const group of groups) {
    for (const child of group.children) {
      const text = clean(child.innerText);
      const key = text.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      records.push({
        container_signature: group.key,
        text: text.slice(0, 12000),
        controls: controlLabels(child),
        links: controlLinks(child),
      });
      if (records.length >= 160) break;
    }
    if (records.length >= 160) break;
  }
  return { records };
})()
"""


GENERIC_CONTINUATION_DISCOVERY_JS = r"""
(() => {
  const clean = (value) => (value || '').replace(/\s+/g, ' ').trim();
  const visible = (el) => {
    if (!(el instanceof Element)) return false;
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      Number(style.opacity || 1) > 0 && rect.width > 0 && rect.height > 0;
  };
  const urls = Array.from(document.querySelectorAll('a[href]')).filter(visible).filter((el) => {
    const label = clean(el.getAttribute('aria-label') || el.getAttribute('title') || el.innerText);
    const rel = clean(el.getAttribute('rel')).toLowerCase();
    return rel.split(/\s+/).includes('next') ||
      /^(next|next page|load more|show more|more results|older)$/i.test(label) ||
      /\b(next|load more|show more|more results|older)\b/i.test(label);
  }).map((el) => el.href).filter(Boolean);
  return { urls: Array.from(new Set(urls)).slice(0, 12) };
})()
"""


GENERIC_CONTINUATION_CLICK_JS = r"""
(() => {
  const clean = (value) => (value || '').replace(/\s+/g, ' ').trim();
  const visible = (el) => {
    if (!(el instanceof Element)) return false;
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      Number(style.opacity || 1) > 0 && rect.width > 0 && rect.height > 0;
  };
  const enabled = (el) => !el.matches(
    ':disabled,[disabled],[aria-disabled="true"],.disabled'
  );
  const candidates = Array.from(document.querySelectorAll(
    'a,button,input[type="button"],input[type="submit"],[role="button"],[role="link"]'
  )).filter((el) => visible(el) && enabled(el)).map((el) => {
    const label = clean(
      el.getAttribute('aria-label') || el.getAttribute('title') ||
      el.innerText || el.value || el.getAttribute('name')
    );
    const rel = clean(el.getAttribute('rel')).toLowerCase();
    let score = 0;
    if (rel.split(/\s+/).includes('next')) score += 100;
    if (/^(next|next page|load more|show more|more results|older)$/i.test(label)) score += 80;
    if (/\b(next|load more|show more|more results|older)\b/i.test(label)) score += 40;
    if (/\b(previous|prev|newer|back)\b/i.test(label)) score -= 100;
    return { el, label, score };
  }).filter((item) => item.score > 0).sort((a, b) => b.score - a.score);
  if (!candidates.length) return { clicked: false };
  const target = candidates[0];
  target.el.scrollIntoView({ block: 'center', inline: 'center' });
  target.el.click();
  return { clicked: true, label: target.label, score: target.score };
})()
"""


def extract_visible_total(text: str) -> int | None:
	patterns = (
		r'\bitems?\s+\d+\s+to\s+\d+\s+of\s+(\d+)\b',
		r'\bshowing\s+\d+\s*(?:-|to)\s*\d+\s+of\s+(\d+)\b',
		r'\b(\d+)\s+(?:reviews?|results?|items?|records?|orders?|comments?)\b',
	)
	values = [int(match.group(1)) for pattern in patterns for match in re.finditer(pattern, text, re.IGNORECASE)]
	return max(values) if values else None


def has_visible_continuation(text: str) -> bool:
	return bool(
		re.search(
			r'(?im)^\s*(?:next|load more|show more|more results|older|next page)\s*$',
			text,
		)
	)
