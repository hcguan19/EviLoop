"""Benchmark-specific environment boundaries that never participate in policy decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class BenchmarkAdapterContract(BaseModel):
	"""Declare how a benchmark is observed, executed, and scored."""

	benchmark_family: str
	observation_modality: Literal['dom', 'vision_dom', 'desktop_visual', 'offline_trajectory']
	execution_adapter: str
	scoring_adapter: str
	evaluation_ready: bool
	status: str


def benchmark_adapter_contract(task_data: dict, task_file: str) -> BenchmarkAdapterContract:
	"""Resolve environment plumbing from task metadata without changing Agent behavior."""

	identity = ' '.join(
		str(value)
		for value in (
			task_file,
			task_data.get('benchmark'),
			task_data.get('dataset'),
			task_data.get('source'),
		)
		if value
	).casefold()
	if task_data.get('miniwob_reward_threshold') is not None:
		return BenchmarkAdapterContract(
			benchmark_family='miniwob++',
			observation_modality='dom',
			execution_adapter='live_browser',
			scoring_adapter='official_runtime_reward',
			evaluation_ready=True,
			status='native',
		)
	if task_data.get('webshop_reward_threshold') is not None:
		return BenchmarkAdapterContract(
			benchmark_family='webshop',
			observation_modality='dom',
			execution_adapter='live_browser',
			scoring_adapter='official_environment_reward',
			evaluation_ready=True,
			status='native',
		)
	if 'visualwebarena' in identity or 'visual_web_arena' in identity:
		return BenchmarkAdapterContract(
			benchmark_family='visualwebarena',
			observation_modality='vision_dom',
			execution_adapter='live_browser',
			scoring_adapter='official_evaluator_required',
			evaluation_ready=False,
			status='environment_and_scorer_pending',
		)
	if 'workarena' in identity:
		return BenchmarkAdapterContract(
			benchmark_family='workarena',
			observation_modality='dom',
			execution_adapter='live_authenticated_browser',
			scoring_adapter='official_evaluator_required',
			evaluation_ready=False,
			status='environment_auth_and_scorer_pending',
		)
	if 'webarena' in identity:
		return BenchmarkAdapterContract(
			benchmark_family='webarena',
			observation_modality='dom',
			execution_adapter='live_browser',
			scoring_adapter='official_evaluator_required',
			evaluation_ready=False,
			status='environment_and_scorer_pending',
		)
	if 'osworld' in identity:
		return BenchmarkAdapterContract(
			benchmark_family='osworld',
			observation_modality='desktop_visual',
			execution_adapter='desktop_vm_required',
			scoring_adapter='official_state_evaluator_required',
			evaluation_ready=False,
			status='desktop_environment_and_scorer_pending',
		)
	if 'mind2web' in identity:
		return BenchmarkAdapterContract(
			benchmark_family='mind2web',
			observation_modality='offline_trajectory',
			execution_adapter='recorded_trajectory_replay',
			scoring_adapter='official_action_metrics_required',
			evaluation_ready=False,
			status='offline_observations_required',
		)
	if 'weblinx' in identity:
		return BenchmarkAdapterContract(
			benchmark_family='weblinx-1.1',
			observation_modality='offline_trajectory',
			execution_adapter='browsergym_replay',
			scoring_adapter='official_action_metrics_required',
			evaluation_ready=False,
			status='offline_observations_required',
		)
	return BenchmarkAdapterContract(
		benchmark_family='generic_web',
		observation_modality='vision_dom',
		execution_adapter='live_browser',
		scoring_adapter='llm_judge_non_official',
		evaluation_ready=False,
		status='exploratory_only',
	)
