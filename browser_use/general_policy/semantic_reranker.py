"""Confidence-gated semantic reranking over a small visible candidate set."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import SystemMessage, UserMessage
from browser_use.tools.task_policy import RankedCandidate


class SemanticRerankConfig(BaseModel):
	"""Runtime and ambiguity thresholds for optional semantic reranking."""

	enabled: bool = False
	top_k: int = Field(default=5, ge=2, le=10)
	margin_threshold: float = Field(default=1.0, ge=0.0, le=10.0)
	max_evidence_chars: int = Field(default=700, ge=100, le=2000)
	timeout_seconds: float = Field(default=60.0, ge=5.0, le=180.0)


class SemanticRerankSelection(BaseModel):
	"""Validated model response referring only to a presented 1-based rank."""

	selected_rank: int = Field(ge=1, le=10)
	confidence: float = Field(ge=0.0, le=1.0)
	reason: str = Field(default='', max_length=500)


class SemanticRerankDecision(BaseModel):
	"""Auditable outcome of the optional reranking stage."""

	triggered: bool = False
	trigger_reasons: list[str] = Field(default_factory=list)
	selected_rank: int | None = None
	selected_identifier: str | None = None
	confidence: float | None = None
	reason: str = ''
	error: str | None = None
	prompt_tokens: int | None = None
	completion_tokens: int | None = None
	total_tokens: int | None = None


def assess_semantic_rerank(ranked: list[RankedCandidate], config: SemanticRerankConfig) -> tuple[bool, list[str]]:
	"""Trigger only when observable ranking evidence is ambiguous or low-information."""

	if not config.enabled or len(ranked) < 2:
		return False, []
	reasons: list[str] = []
	margin = ranked[0].score - ranked[1].score
	if margin <= config.margin_threshold:
		reasons.append('small_top_score_margin')
	top = ranked[0].candidate
	tag = str(top.metadata.get('tag') or '').casefold()
	attributes = str(top.metadata.get('attributes') or '').casefold()
	semantic_text = f'{top.label} {top.visible_text}'.strip()
	if len(semantic_text) < 24:
		reasons.append('sparse_top_candidate_evidence')
	if tag in {'div', 'span', 'svg'} and not any(
		marker in attributes for marker in ('contenteditable', 'is_clickable', 'onclick', 'role=')
	):
		reasons.append('ambiguous_structural_top_candidate')
	return bool(reasons), reasons


def apply_semantic_selection(ranked: list[RankedCandidate], selected_rank: int, top_k: int) -> list[RankedCandidate]:
	"""Move one validated Top-K selection to rank one while preserving deterministic order."""

	bounded = min(top_k, len(ranked))
	if selected_rank < 1 or selected_rank > bounded:
		raise ValueError(f'selected_rank must be between 1 and {bounded}, got {selected_rank}')
	selected = ranked[selected_rank - 1]
	return [selected, *ranked[: selected_rank - 1], *ranked[selected_rank:]]


def _candidate_evidence(candidate: RankedCandidate, index: int, max_chars: int) -> str:
	visible = candidate.candidate.visible_text[:max_chars]
	metadata = {key: value for key, value in candidate.candidate.metadata.items() if key in {'tag', 'attributes', 'role'}}
	return (
		f'Candidate {index}\n'
		f'label: {candidate.candidate.label[:300]}\n'
		f'visible_evidence: {visible}\n'
		f'metadata: {metadata}\n'
		f'retrieval_score: {candidate.score}'
	)


async def semantic_rerank(
	*,
	objective: str,
	operation: str,
	ranked: list[RankedCandidate],
	llm: BaseChatModel,
	config: SemanticRerankConfig,
) -> tuple[list[RankedCandidate], SemanticRerankDecision]:
	"""Use an LLM only after the deterministic ambiguity gate requests semantic evidence."""

	triggered, reasons = assess_semantic_rerank(ranked, config)
	if not triggered:
		return ranked, SemanticRerankDecision(triggered=False, trigger_reasons=reasons)
	top_candidates = ranked[: config.top_k]
	candidate_text = '\n\n'.join(
		_candidate_evidence(candidate, index, config.max_evidence_chars)
		for index, candidate in enumerate(top_candidates, start=1)
	)
	system = SystemMessage(
		content=(
			'You select the single best visible web element for the current user instruction. '
			'Use only the supplied candidate evidence. Do not invent selectors or choose an element outside the list. '
			'Prefer semantic purpose and control role over incidental repeated page text.'
		)
	)
	user = UserMessage(
		content=(
			f'Current instruction: {objective}\n'
			f'Expected operation: {operation or "infer from instruction"}\n\n'
			f'{candidate_text}\n\nReturn the 1-based rank of the best candidate.'
		)
	)
	try:
		response = await asyncio.wait_for(
			llm.ainvoke([system, user], output_format=SemanticRerankSelection),
			timeout=config.timeout_seconds,
		)
		selection = response.completion
		if selection.selected_rank > len(top_candidates):
			raise ValueError(f'model selected rank {selection.selected_rank} outside Top-{len(top_candidates)}')
		reranked = apply_semantic_selection(ranked, selection.selected_rank, config.top_k)
		usage: Any = response.usage
		return reranked, SemanticRerankDecision(
			triggered=True,
			trigger_reasons=reasons,
			selected_rank=selection.selected_rank,
			selected_identifier=reranked[0].candidate.identifier,
			confidence=selection.confidence,
			reason=selection.reason,
			prompt_tokens=getattr(usage, 'prompt_tokens', None),
			completion_tokens=getattr(usage, 'completion_tokens', None),
			total_tokens=getattr(usage, 'total_tokens', None),
		)
	except Exception as exc:
		return ranked, SemanticRerankDecision(
			triggered=True,
			trigger_reasons=reasons,
			error=f'{type(exc).__name__}: {exc}',
		)
