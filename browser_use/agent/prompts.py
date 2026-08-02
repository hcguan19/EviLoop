import importlib.resources
import re
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

from browser_use.browser.views import PLACEHOLDER_4PX_SCREENSHOT
from browser_use.dom.views import NodeType, SimplifiedNode
from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, SystemMessage, UserMessage
from browser_use.observability import observe_debug
from browser_use.utils import is_new_tab_page, sanitize_surrogates

if TYPE_CHECKING:
	from browser_use.agent.views import AgentStepInfo
	from browser_use.browser.views import BrowserStateSummary
	from browser_use.filesystem.file_system import FileSystem


def _is_anthropic_4_5_model(model_name: str | None) -> bool:
	"""Check if the model is Claude Opus 4.5 or Haiku 4.5 (requires 4096+ token prompts for caching)."""
	if not model_name:
		return False
	model_lower = model_name.lower()
	# Check for Opus 4.5 or Haiku 4.5 variants
	is_opus_4_5 = 'opus' in model_lower and ('4.5' in model_lower or '4-5' in model_lower)
	is_haiku_4_5 = 'haiku' in model_lower and ('4.5' in model_lower or '4-5' in model_lower)
	return is_opus_4_5 or is_haiku_4_5


class SystemPrompt:
	def __init__(
		self,
		max_actions_per_step: int = 3,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
		use_thinking: bool = True,
		flash_mode: bool = False,
		is_anthropic: bool = False,
		is_browser_use_model: bool = False,
		model_name: str | None = None,
	):
		self.max_actions_per_step = max_actions_per_step
		self.use_thinking = use_thinking
		self.flash_mode = flash_mode
		self.is_anthropic = is_anthropic
		self.is_browser_use_model = is_browser_use_model
		self.model_name = model_name
		# Check if this is an Anthropic 4.5 model that needs longer prompts for caching
		self.is_anthropic_4_5 = _is_anthropic_4_5_model(model_name)
		prompt = ''
		if override_system_message is not None:
			prompt = override_system_message
		else:
			self._load_prompt_template()
			prompt = self.prompt_template.format(max_actions=self.max_actions_per_step)

		if extend_system_message:
			prompt += f'\n{extend_system_message}'

		self.system_message = SystemMessage(content=prompt, cache=True)

	def _load_prompt_template(self) -> None:
		"""Load the prompt template from the markdown file."""
		try:
			# Choose the appropriate template based on model type and mode
			# Browser-use models use simplified prompts optimized for fine-tuned models
			if self.is_browser_use_model:
				if self.flash_mode:
					template_filename = 'system_prompt_browser_use_flash.md'
				elif self.use_thinking:
					template_filename = 'system_prompt_browser_use.md'
				else:
					template_filename = 'system_prompt_browser_use_no_thinking.md'
			# Anthropic 4.5 models (Opus 4.5, Haiku 4.5) need 4096+ token prompts for caching
			elif self.is_anthropic_4_5 and self.flash_mode:
				template_filename = 'system_prompt_anthropic_flash.md'
			elif self.flash_mode and self.is_anthropic:
				template_filename = 'system_prompt_flash_anthropic.md'
			elif self.flash_mode:
				template_filename = 'system_prompt_flash.md'
			elif self.use_thinking:
				template_filename = 'system_prompt.md'
			else:
				template_filename = 'system_prompt_no_thinking.md'

			# This works both in development and when installed as a package
			with (
				importlib.resources.files('browser_use.agent.system_prompts')
				.joinpath(template_filename)
				.open('r', encoding='utf-8') as f
			):
				self.prompt_template = f.read()
		except Exception as e:
			raise RuntimeError(f'Failed to load system prompt template: {e}')

	def get_system_message(self) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
		    SystemMessage: Formatted system prompt
		"""
		return self.system_message


class AgentMessagePrompt:
	vision_detail_level: Literal['auto', 'low', 'high']

	def __init__(
		self,
		browser_state_summary: 'BrowserStateSummary',
		file_system: 'FileSystem',
		agent_history_description: str | None = None,
		read_state_description: str | None = None,
		task: str | None = None,
		include_attributes: list[str] | None = None,
		step_info: Optional['AgentStepInfo'] = None,
		page_filtered_actions: str | None = None,
		max_clickable_elements_length: int = 40000,
		sensitive_data: str | None = None,
		available_file_paths: list[str] | None = None,
		screenshots: list[str] | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		include_recent_events: bool = False,
		sample_images: list[ContentPartTextParam | ContentPartImageParam] | None = None,
		read_state_images: list[dict] | None = None,
		llm_screenshot_size: tuple[int, int] | None = None,
		unavailable_skills_info: str | None = None,
		plan_description: str | None = None,
		visual_context_note: str | None = None,
	):
		self.browser_state: 'BrowserStateSummary' = browser_state_summary
		self.file_system: 'FileSystem | None' = file_system
		self.agent_history_description: str | None = agent_history_description
		self.read_state_description: str | None = read_state_description
		self.task: str | None = task
		self.include_attributes = include_attributes
		self.step_info = step_info
		self.page_filtered_actions: str | None = page_filtered_actions
		self.max_clickable_elements_length: int = max_clickable_elements_length
		self.sensitive_data: str | None = sensitive_data
		self.available_file_paths: list[str] | None = available_file_paths
		self.screenshots = screenshots or []
		self.vision_detail_level = vision_detail_level
		self.include_recent_events = include_recent_events
		self.sample_images = sample_images or []
		self.read_state_images = read_state_images or []
		self.unavailable_skills_info: str | None = unavailable_skills_info
		self.plan_description: str | None = plan_description
		self.visual_context_note: str | None = visual_context_note
		self.llm_screenshot_size = llm_screenshot_size
		assert self.browser_state

	def _extract_page_statistics(self) -> dict[str, int]:
		"""Extract high-level page statistics from DOM tree for LLM context"""
		stats = {
			'links': 0,
			'iframes': 0,
			'shadow_open': 0,
			'shadow_closed': 0,
			'scroll_containers': 0,
			'images': 0,
			'interactive_elements': 0,
			'total_elements': 0,
			'text_chars': 0,
		}

		if not self.browser_state.dom_state or not self.browser_state.dom_state._root:
			return stats

		def traverse_node(node: SimplifiedNode) -> None:
			"""Recursively traverse simplified DOM tree to count elements"""
			if not node or not node.original_node:
				return

			original = node.original_node
			stats['total_elements'] += 1

			# Count by node type and tag
			if original.node_type == NodeType.ELEMENT_NODE:
				tag = original.tag_name.lower() if original.tag_name else ''

				if tag == 'a':
					stats['links'] += 1
				elif tag in ('iframe', 'frame'):
					stats['iframes'] += 1
				elif tag == 'img':
					stats['images'] += 1

				# Check if scrollable
				if original.is_actually_scrollable:
					stats['scroll_containers'] += 1

				# Check if interactive
				if node.is_interactive:
					stats['interactive_elements'] += 1

				# Check if this element hosts shadow DOM
				if node.is_shadow_host:
					# Check if any shadow children are closed
					has_closed_shadow = any(
						child.original_node.node_type == NodeType.DOCUMENT_FRAGMENT_NODE
						and child.original_node.shadow_root_type
						and child.original_node.shadow_root_type.lower() == 'closed'
						for child in node.children
					)
					if has_closed_shadow:
						stats['shadow_closed'] += 1
					else:
						stats['shadow_open'] += 1

			elif original.node_type == NodeType.TEXT_NODE:
				stats['text_chars'] += len(original.node_value.strip())

			elif original.node_type == NodeType.DOCUMENT_FRAGMENT_NODE:
				# Shadow DOM fragment - these are the actual shadow roots
				# But don't double-count since we count them at the host level above
				pass

			# Traverse children
			for child in node.children:
				traverse_node(child)

		traverse_node(self.browser_state.dom_state._root)
		return stats

	def _task_keywords(self) -> set[str]:
		if not self.task:
			return set()
		words = re.findall(r'[a-zA-Z0-9]+', self.task.lower())
		stopwords = {
			'open',
			'http',
			'https',
			'www',
			'127',
			'0',
			'1',
			'3003',
			'browseruse',
			'fixed',
			'complete',
			'webshop',
			'shopping',
			'instruction',
			'shown',
			'page',
			'search',
			'using',
			'key',
			'product',
			'words',
			'required',
			'attributes',
			'options',
			'values',
			'before',
			'clicking',
			'after',
			'checkout',
			'read',
			'include',
			'final',
			'response',
			'under',
			'price',
		}
		return {word for word in words if len(word) >= 3 and word not in stopwords}

	def _should_use_autowebglm_pruner(self) -> bool:
		if not self.task:
			return False
		task_lower = self.task.lower()
		url_lower = (self.browser_state.url or '').lower()
		return 'webshop' in task_lower or 'browseruse_fixed_' in task_lower or 'browseruse_fixed_' in url_lower

	def _autowebglm_node_text(self, node: SimplifiedNode, max_length: int = 500) -> str:
		original = node.original_node
		text_parts: list[str] = []
		if original.node_type == NodeType.TEXT_NODE:
			text_parts.append(original.node_value or '')
		elif original.node_type == NodeType.ELEMENT_NODE:
			attrs = original.attributes or {}
			for attr in (
				'id',
				'name',
				'type',
				'role',
				'aria-label',
				'placeholder',
				'title',
				'value',
				'alt',
				'href',
				'class',
			):
				value = attrs.get(attr)
				if value:
					text_parts.append(str(value))
			try:
				text_parts.append(original.get_all_children_text(max_depth=2))
			except Exception:
				text_parts.append(original.get_meaningful_text_for_llm())
		text = ' '.join(part.strip() for part in text_parts if part and part.strip())
		text = re.sub(r'\s+', ' ', text).strip()
		return text[:max_length]

	def _autowebglm_node_score(self, node: SimplifiedNode, keywords: set[str]) -> int:
		original = node.original_node
		tag = (original.tag_name or '').lower()
		text = self._autowebglm_node_text(node).lower()
		attrs = original.attributes or {}
		role = attrs.get('role', '').lower()

		score = 0
		strong_terms = {
			'buy',
			'buy now',
			'add to cart',
			'cart',
			'checkout',
			'score',
			'reward',
			'search',
			'results',
			'price',
			'$',
		}
		control_terms = {
			'input',
			'button',
			'select',
			'option',
			'checkbox',
			'radio',
			'combobox',
			'textbox',
			'quantity',
			'color',
			'size',
			'style',
			'flavor',
			'scent',
			'model',
			'capacity',
		}
		if any(term in text for term in strong_terms):
			score += 8
		if any(term in text for term in control_terms):
			score += 4
		if tag in {'input', 'select', 'button', 'textarea', 'a', 'option'}:
			score += 5
		if role in {'button', 'link', 'checkbox', 'radio', 'tab', 'menuitem', 'option', 'combobox', 'textbox'}:
			score += 4
		if node.is_interactive:
			score += 5
		if original.is_actually_scrollable or original.is_scrollable:
			score += 2
		for keyword in keywords:
			if keyword in text:
				score += 3
		if tag in {'nav', 'footer', 'header', 'script', 'style', 'noscript'}:
			score -= 4
		return score

	def _autowebglm_prune_dom_elements(self, elements_text: str) -> tuple[str, str]:
		"""Prune via DOM structure: keep relevant nodes plus ancestors, descendants, and nearby siblings."""
		if not elements_text or not self._should_use_autowebglm_pruner():
			return elements_text, ''
		if len(elements_text) <= self.max_clickable_elements_length and len(elements_text.splitlines()) <= 120:
			return elements_text, ''
		if not self.browser_state.dom_state or not self.browser_state.dom_state._root:
			return elements_text, ''

		try:
			from browser_use.dom.serializer.serializer import DOMTreeSerializer

			root = self.browser_state.dom_state._root
			keywords = self._task_keywords()
			nodes: list[tuple[int, SimplifiedNode, int, int | None]] = []
			parent_by_id: dict[int, int | None] = {}
			children_by_id: dict[int, list[int]] = {}
			node_by_id: dict[int, SimplifiedNode] = {}

			def traverse(node: SimplifiedNode, depth: int, parent_id: int | None) -> None:
				node_id = id(node)
				nodes.append((node_id, node, depth, parent_id))
				parent_by_id[node_id] = parent_id
				children_by_id.setdefault(node_id, [])
				node_by_id[node_id] = node
				if parent_id is not None:
					children_by_id.setdefault(parent_id, []).append(node_id)
				for child in node.children:
					traverse(child, depth + 1, node_id)

			traverse(root, 0, None)
			if len(nodes) <= 80:
				return elements_text, ''

			scored_nodes: list[tuple[int, int]] = []
			for node_id, node, _depth, _parent_id in nodes:
				score = self._autowebglm_node_score(node, keywords)
				if score > 0:
					scored_nodes.append((score, node_id))
			if not scored_nodes:
				return elements_text, ''

			scored_nodes.sort(reverse=True)
			anchor_ids = [node_id for score, node_id in scored_nodes if score >= 5][:90]
			if not anchor_ids:
				anchor_ids = [node_id for _score, node_id in scored_nodes[:40]]

			keep_ids: set[int] = set()

			def keep_ancestors(node_id: int) -> None:
				current_id = node_id
				while current_id is not None:
					keep_ids.add(current_id)
					current_id = parent_by_id.get(current_id)

			def keep_descendants(node_id: int, remaining_depth: int) -> None:
				if remaining_depth < 0:
					return
				for child_id in children_by_id.get(node_id, [])[:12]:
					keep_ids.add(child_id)
					keep_descendants(child_id, remaining_depth - 1)

			def keep_nearby_siblings(node_id: int, radius: int = 2) -> None:
				parent_id = parent_by_id.get(node_id)
				if parent_id is None:
					return
				siblings = children_by_id.get(parent_id, [])
				try:
					index = siblings.index(node_id)
				except ValueError:
					return
				for sibling_id in siblings[max(0, index - radius) : index + radius + 1]:
					keep_ids.add(sibling_id)

			for anchor_id in anchor_ids:
				keep_ancestors(anchor_id)
				keep_descendants(anchor_id, remaining_depth=2)
				keep_nearby_siblings(anchor_id)

			# Keep all visible interactive controls in compact WebShop pages, because missing one option can ruin reward.
			for node_id, node, _depth, _parent_id in nodes:
				if node.is_interactive and len(keep_ids) < 320:
					keep_ancestors(node_id)
					keep_ids.add(node_id)

			max_kept_nodes = 360
			if len(keep_ids) > max_kept_nodes:
				ordered_keep_ids = [node_id for _score, node_id in scored_nodes if node_id in keep_ids]
				ordered_keep_ids.extend(node_id for node_id, _node, _depth, _parent_id in nodes if node_id in keep_ids)
				keep_ids = set(ordered_keep_ids[:max_kept_nodes])
				for node_id in list(keep_ids):
					keep_ancestors(node_id)

			def render(node_id: int, depth: int) -> list[str]:
				if node_id not in keep_ids:
					return []
				node = node_by_id[node_id]
				original = node.original_node
				lines: list[str] = []
				depth_str = '\t' * depth
				if original.node_type == NodeType.TEXT_NODE:
					text = re.sub(r'\s+', ' ', (original.node_value or '').strip())
					if text and len(text) > 1:
						lines.append(f'{depth_str}{text[:220]}')
				elif original.node_type == NodeType.DOCUMENT_FRAGMENT_NODE:
					lines.append(f'{depth_str}Open Shadow')
				elif original.node_type == NodeType.ELEMENT_NODE:
					tag = original.tag_name or 'element'
					prefix = ''
					if node.is_shadow_host:
						prefix = '|SHADOW(open)|'
					if node.is_interactive:
						new_prefix = '*' if node.is_new else ''
						line = f'{depth_str}{prefix}{new_prefix}[{original.backend_node_id}]<{tag}'
					elif original.is_actually_scrollable or original.is_scrollable:
						line = f'{depth_str}{prefix}|scroll element|<{tag}'
					elif tag.upper() == 'IFRAME':
						line = f'{depth_str}{prefix}|IFRAME|<{tag}'
					elif tag.upper() == 'FRAME':
						line = f'{depth_str}{prefix}|FRAME|<{tag}'
					else:
						line = f'{depth_str}{prefix}<{tag}'
					attributes_html_str = DOMTreeSerializer._build_attributes_string(
						original, self.include_attributes or [], ''
					)
					text = self._autowebglm_node_text(node, max_length=180)
					text = text.replace('"', "'")
					if text and tag.lower() not in {'html', 'body'} and 'text=' not in attributes_html_str:
						attributes_html_str = f'{attributes_html_str} text="{text}"'.strip()
					if attributes_html_str:
						line += f' {attributes_html_str[:260]}'
					line += ' />'
					lines.append(line)

				child_depth = depth + 1 if original.node_type in {NodeType.ELEMENT_NODE, NodeType.DOCUMENT_FRAGMENT_NODE} else depth
				for child_id in children_by_id.get(node_id, []):
					lines.extend(render(child_id, child_depth))
				return lines

			rendered_lines = render(id(root), 0)
			pruned_text = '\n'.join(rendered_lines)
			if not pruned_text or len(pruned_text) >= len(elements_text) * 0.92:
				return elements_text, ''
			note = (
				f' (DOM-aware AutoWebGLM pruner kept {len(keep_ids)}/{len(nodes)} nodes, '
				f'{len(anchor_ids)} relevance anchors, preserving task matches plus ancestors/descendants/siblings)'
			)
			return pruned_text, note
		except Exception:
			return elements_text, ''

	def _autowebglm_prune_elements_text(self, elements_text: str) -> tuple[str, str]:
		"""Prune noisy page representation while preserving task-relevant context."""
		if not elements_text or not self._should_use_autowebglm_pruner():
			return elements_text, ''

		lines = elements_text.splitlines()
		if len(lines) <= 120 and len(elements_text) <= self.max_clickable_elements_length:
			return elements_text, ''

		keywords = self._task_keywords()
		always_keep_terms = {
			'search',
			'input',
			'button',
			'buy',
			'buy now',
			'add to cart',
			'checkout',
			'price',
			'$',
			'option',
			'select',
			'color',
			'size',
			'style',
			'flavor',
			'scent',
			'model',
			'capacity',
			'quantity',
			'score',
			'reward',
		}
		keep_indexes: set[int] = set()
		for index, line in enumerate(lines):
			line_lower = line.lower()
			if any(term in line_lower for term in always_keep_terms) or any(keyword in line_lower for keyword in keywords):
				for neighbor in range(max(0, index - 1), min(len(lines), index + 2)):
					keep_indexes.add(neighbor)

		if not keep_indexes:
			return elements_text, ''

		kept_lines = [lines[index] for index in sorted(keep_indexes)]
		pruned_text = '\n'.join(kept_lines)
		if len(pruned_text) >= len(elements_text) * 0.9:
			return elements_text, ''

		note = (
			f' (AutoWebGLM-style HTML pruner kept {len(kept_lines)}/{len(lines)} lines around task keywords, '
			'controls, options, prices, and final actions)'
		)
		return pruned_text, note

	@observe_debug(ignore_input=True, ignore_output=True, name='_get_browser_state_description')
	def _get_browser_state_description(self) -> str:
		# Extract page statistics first
		page_stats = self._extract_page_statistics()

		# Format statistics
		stats_text = '<page_stats>'
		if page_stats['total_elements'] < 10:
			stats_text += 'Page appears empty (SPA not loaded?) - '
		# Skeleton screen: many elements but almost no text = loading placeholders
		elif page_stats['total_elements'] > 20 and page_stats['text_chars'] < page_stats['total_elements'] * 5:
			stats_text += 'Page appears to show skeleton/placeholder content (still loading?) - '
		stats_text += f'{page_stats["links"]} links, {page_stats["interactive_elements"]} interactive, '
		stats_text += f'{page_stats["iframes"]} iframes'
		if page_stats['shadow_open'] > 0 or page_stats['shadow_closed'] > 0:
			stats_text += f', {page_stats["shadow_open"]} shadow(open), {page_stats["shadow_closed"]} shadow(closed)'
		if page_stats['images'] > 0:
			stats_text += f', {page_stats["images"]} images'
		stats_text += f', {page_stats["total_elements"]} total elements'
		stats_text += '</page_stats>\n'

		elements_text = self.browser_state.dom_state.llm_representation(include_attributes=self.include_attributes)
		elements_text, pruned_text = self._autowebglm_prune_dom_elements(elements_text)
		if not pruned_text:
			elements_text, pruned_text = self._autowebglm_prune_elements_text(elements_text)

		if len(elements_text) > self.max_clickable_elements_length:
			elements_text = elements_text[: self.max_clickable_elements_length]
			truncated_text = f' (truncated to {self.max_clickable_elements_length} characters)'
		else:
			truncated_text = ''
		truncated_text += pruned_text

		has_content_above = False
		has_content_below = False
		# Enhanced page information for the model
		page_info_text = ''
		if self.browser_state.page_info:
			pi = self.browser_state.page_info
			# Compute page statistics dynamically
			pages_above = pi.pixels_above / pi.viewport_height if pi.viewport_height > 0 else 0
			pages_below = pi.pixels_below / pi.viewport_height if pi.viewport_height > 0 else 0
			has_content_above = pages_above > 0
			has_content_below = pages_below > 0
			page_info_text = '<page_info>'
			page_info_text += f'{pages_above:.1f} pages above, {pages_below:.1f} pages below'
			if pages_below > 0.2:
				page_info_text += ' — scroll down to reveal more content'
			page_info_text += '</page_info>\n'
		if elements_text != '':
			if not has_content_above:
				elements_text = f'[Start of page]\n{elements_text}'
			if not has_content_below:
				elements_text = f'{elements_text}\n[End of page]'
		else:
			elements_text = 'empty page'

		tabs_text = ''
		current_tab_candidates = []

		# Find tabs that match both URL and title to identify current tab more reliably
		for tab in self.browser_state.tabs:
			if tab.url == self.browser_state.url and tab.title == self.browser_state.title:
				current_tab_candidates.append(tab.target_id)

		# If we have exactly one match, mark it as current
		# Otherwise, don't mark any tab as current to avoid confusion
		current_target_id = current_tab_candidates[0] if len(current_tab_candidates) == 1 else None

		for tab in self.browser_state.tabs:
			tabs_text += f'Tab {tab.target_id[-4:]}: {tab.url} - {tab.title[:30]}\n'

		current_tab_text = f'Current tab: {current_target_id[-4:]}' if current_target_id is not None else ''

		# Check if current page is a PDF viewer and add appropriate message
		pdf_message = ''
		if self.browser_state.is_pdf_viewer:
			pdf_message = (
				'PDF viewer cannot be rendered. In this page, DO NOT use the extract action as PDF content cannot be rendered. '
			)
			pdf_message += (
				'Use the read_file action on the downloaded PDF in available_file_paths to read the full text content.\n\n'
			)

		# Add recent events if available and requested
		recent_events_text = ''
		if self.include_recent_events and self.browser_state.recent_events:
			recent_events_text = f'Recent browser events: {self.browser_state.recent_events}\n'

		# Add closed popup messages if any
		closed_popups_text = ''
		if self.browser_state.closed_popup_messages:
			closed_popups_text = 'Auto-closed JavaScript dialogs:\n'
			for popup_msg in self.browser_state.closed_popup_messages:
				closed_popups_text += f'  - {popup_msg}\n'
			closed_popups_text += '\n'

		browser_state = f"""{stats_text}{current_tab_text}
Available tabs:
{tabs_text}
{page_info_text}
{recent_events_text}{closed_popups_text}{pdf_message}Interactive elements{truncated_text}:
{elements_text}
"""
		return browser_state

	def _get_agent_state_description(self) -> str:
		_todo_contents = self.file_system.get_todo_contents() if self.file_system else ''
		if not len(_todo_contents):
			_todo_contents = '[empty todo.md, fill it when applicable]'

		agent_state = f"""
<file_system>
{self.file_system.describe() if self.file_system else 'No file system available'}
</file_system>
<todo_contents>
{_todo_contents}
</todo_contents>
"""
		if self.plan_description:
			agent_state += f'<plan>\n{self.plan_description}\n</plan>\n'

		if self.sensitive_data:
			agent_state += f'<sensitive_data>{self.sensitive_data}</sensitive_data>\n'

		if self.available_file_paths:
			available_file_paths_text = '\n'.join(self.available_file_paths)
			agent_state += f'<available_file_paths>{available_file_paths_text}\nUse with absolute paths</available_file_paths>\n'
		return agent_state

	def _get_user_request_description(self) -> str:
		return f'<user_request>\n{self.task}\n</user_request>\n\n'

	def _get_step_meta_description(self) -> str:
		# Per-step varying metadata (step counter, wall-clock date). Kept out of <agent_state> so it
		# lives at the tail of the user message — anything before this block can in principle be
		# treated as the cacheable prefix.
		if self.step_info:
			step_info_description = f'Step{self.step_info.step_number + 1} maximum:{self.step_info.max_steps}\n'
		else:
			step_info_description = ''
		step_info_description += f'Today:{datetime.now().strftime("%Y-%m-%d")}'
		return f'<step_info>{step_info_description}</step_info>\n'

	def _resize_screenshot(self, screenshot_b64: str) -> str:
		"""Resize screenshot to llm_screenshot_size if configured."""
		if not self.llm_screenshot_size:
			return screenshot_b64

		try:
			import base64
			import logging
			from io import BytesIO

			from PIL import Image

			img = Image.open(BytesIO(base64.b64decode(screenshot_b64)))
			if img.size == self.llm_screenshot_size:
				return screenshot_b64

			logging.getLogger(__name__).info(
				f'🔄 Resizing screenshot from {img.size[0]}x{img.size[1]} to {self.llm_screenshot_size[0]}x{self.llm_screenshot_size[1]} for LLM'
			)

			img_resized = img.resize(self.llm_screenshot_size, Image.Resampling.LANCZOS)
			buffer = BytesIO()
			img_resized.save(buffer, format='PNG')
			return base64.b64encode(buffer.getvalue()).decode('utf-8')
		except Exception as e:
			logging.getLogger(__name__).warning(f'Failed to resize screenshot: {e}, using original')
			return screenshot_b64

	@observe_debug(ignore_input=True, ignore_output=True, name='get_user_message')
	def get_user_message(self, use_vision: bool = True) -> UserMessage:
		"""Get complete state as a single cached message"""
		# New-tab pages only carry placeholder screenshots, even later in a multi-tab session.
		if is_new_tab_page(self.browser_state.url):
			use_vision = False

		# Build complete state description
		state_description = (
			self._get_user_request_description()
			+ '<agent_history>\n'
			+ (self.agent_history_description.strip('\n') if self.agent_history_description else '')
			+ '\n</agent_history>\n\n'
		)
		state_description += '<agent_state>\n' + self._get_agent_state_description().strip('\n') + '\n</agent_state>\n'
		state_description += '<browser_state>\n' + self._get_browser_state_description().strip('\n') + '\n</browser_state>\n'
		if self.visual_context_note:
			state_description += '<visual_context>\n' + self.visual_context_note.strip('\n') + '\n</visual_context>\n'
		# Only add read_state if it has content
		read_state_description = self.read_state_description.strip('\n').strip() if self.read_state_description else ''
		if read_state_description:
			state_description += '<read_state>\n' + read_state_description + '\n</read_state>\n'

		if self.page_filtered_actions:
			state_description += '<page_specific_actions>\n'
			state_description += self.page_filtered_actions + '\n'
			state_description += '</page_specific_actions>\n'

		# Add unavailable skills information if any
		if self.unavailable_skills_info:
			state_description += '\n' + self.unavailable_skills_info + '\n'

		# Per-step varying metadata (step counter, date) lives at the tail of the message so that
		# everything above can in principle be treated as a cacheable prefix.
		state_description += self._get_step_meta_description()

		# Sanitize surrogates from all text content
		state_description = sanitize_surrogates(state_description)

		# Check if we have images to include (from read_file action)
		has_images = bool(self.read_state_images)
		screenshots = [screenshot for screenshot in self.screenshots if screenshot != PLACEHOLDER_4PX_SCREENSHOT]

		if (use_vision is True and screenshots) or has_images:
			# Start with text description
			content_parts: list[ContentPartTextParam | ContentPartImageParam] = [ContentPartTextParam(text=state_description)]

			# Add sample images
			content_parts.extend(self.sample_images)

			# Add screenshots with labels
			for i, screenshot in enumerate(screenshots):
				if i == len(screenshots) - 1:
					label = 'Current screenshot:'
				else:
					# Use simple, accurate labeling since we don't have actual step timing info
					label = 'Previous screenshot:'

				# Add label as text content
				content_parts.append(ContentPartTextParam(text=label))

				# Resize screenshot if llm_screenshot_size is configured
				processed_screenshot = self._resize_screenshot(screenshot)

				# Add the screenshot
				content_parts.append(
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:image/png;base64,{processed_screenshot}',
							media_type='image/png',
							detail=self.vision_detail_level,
						),
					)
				)

			# Add read_state images (from read_file action) before screenshots
			for img_data in self.read_state_images:
				img_name = img_data.get('name', 'unknown')
				img_base64 = img_data.get('data', '')

				if not img_base64:
					continue

				# Detect image format from name
				if img_name.lower().endswith('.png'):
					media_type = 'image/png'
				else:
					media_type = 'image/jpeg'

				# Add label
				content_parts.append(ContentPartTextParam(text=f'Image from file: {img_name}'))

				# Add the image
				content_parts.append(
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:{media_type};base64,{img_base64}',
							media_type=media_type,
							detail=self.vision_detail_level,
						),
					)
				)

			return UserMessage(content=content_parts, cache=True)

		return UserMessage(content=state_description, cache=True)


def get_rerun_summary_prompt(original_task: str, total_steps: int, success_count: int, error_count: int) -> str:
	return f'''You are analyzing the completion of a rerun task. Based on the screenshot and execution info, provide a summary.

Original task: {original_task}

Execution statistics:
- Total steps: {total_steps}
- Successful steps: {success_count}
- Failed steps: {error_count}

Analyze the screenshot to determine:
1. Whether the task completed successfully
2. What the final state shows
3. Overall completion status (complete/partial/failed)

Respond with:
- summary: A clear, concise summary of what happened during the rerun
- success: Whether the task completed successfully (true/false)
- completion_status: One of "complete", "partial", or "failed"'''


def get_rerun_summary_message(prompt: str, screenshot_b64: str | None = None) -> UserMessage:
	"""
	Build a UserMessage for rerun summary generation.

	Args:
		prompt: The prompt text
		screenshot_b64: Optional base64-encoded screenshot

	Returns:
		UserMessage with prompt and optional screenshot
	"""
	if screenshot_b64:
		# With screenshot: use multi-part content
		content_parts: list[ContentPartTextParam | ContentPartImageParam] = [
			ContentPartTextParam(type='text', text=prompt),
			ContentPartImageParam(
				type='image_url',
				image_url=ImageURL(url=f'data:image/png;base64,{screenshot_b64}'),
			),
		]
		return UserMessage(content=content_parts)
	else:
		# Without screenshot: use simple string content
		return UserMessage(content=prompt)


def get_ai_step_system_prompt() -> str:
	"""
	Get system prompt for AI step action used during rerun.

	Returns:
		System prompt string for AI step
	"""
	return """
You are an expert at extracting data from webpages.

<input>
You will be given:
1. A query describing what to extract
2. The markdown of the webpage (filtered to remove noise)
3. Optionally, a screenshot of the current page state
</input>

<instructions>
- Extract information from the webpage that is relevant to the query
- ONLY use the information available in the webpage - do not make up information
- If the information is not available, mention that clearly
- If the query asks for all items, list all of them
</instructions>

<output>
- Present ALL relevant information in a concise way
- Do not use conversational format - directly output the relevant information
- If information is unavailable, state that clearly
</output>
""".strip()


def get_ai_step_user_prompt(query: str, stats_summary: str, content: str) -> str:
	"""
	Build user prompt for AI step action.

	Args:
		query: What to extract or analyze
		stats_summary: Content statistics summary
		content: Page markdown content

	Returns:
		Formatted prompt string
	"""
	return f'<query>\n{query}\n</query>\n\n<content_stats>\n{stats_summary}\n</content_stats>\n\n<webpage_content>\n{content}\n</webpage_content>'
