from types import SimpleNamespace

import pytest
from browser_use import Tools
from pydantic import ValidationError

from browser_use.tools.generic_actions import (
	SetCurrentUrlQueryAction,
	rank_labeled_elements,
	rank_record_controls,
	register_generic_actions,
)


def node(name: str, *, tag: str = 'a', role: str = '') -> SimpleNamespace:
	return SimpleNamespace(
		node_name=tag,
		ax_node=SimpleNamespace(name=name),
		attributes={'role': role},
	)


def test_exact_accessible_label_beats_nested_container() -> None:
	selector_map = {
		10: node('Reviews 12', tag='div'),
		20: node('Reviews 12', tag='a', role='tab'),
	}
	ranked = rank_labeled_elements('Reviews 12', selector_map)
	assert ranked[0][1] == 20


def test_resolution_uses_current_selector_map() -> None:
	first = {100: node('Next', tag='a')}
	second = {900: node('Next', tag='a')}
	assert rank_labeled_elements('Next', first)[0][1] == 100
	assert rank_labeled_elements('Next', second)[0][1] == 900


def test_action_registers_with_tools() -> None:
	tools = Tools()
	register_generic_actions(tools)
	assert 'activate_visible_label' in tools.registry.registry.actions
	assert 'activate_record_control' in tools.registry.registry.actions
	assert 'inspect_form_state' in tools.registry.registry.actions
	assert 'select_only_visible_choice' in tools.registry.registry.actions
	assert 'inspect_ranked_candidates' in tools.registry.registry.actions
	assert 'clear_existing_collection' in tools.registry.registry.actions
	assert 'navigate_visible_link' in tools.registry.registry.actions
	assert 'set_current_url_query' in tools.registry.registry.actions
	assert 'submit_site_search' in tools.registry.registry.actions
	assert 'fill_visible_form_fields' in tools.registry.registry.actions
	assert 'hover_visible_label' in tools.registry.registry.actions


def test_query_constraint_action_validates_parameter_name() -> None:
	assert SetCurrentUrlQueryAction(parameter='price', value='0-25').parameter == 'price'
	with pytest.raises(ValidationError):
		SetCurrentUrlQueryAction(parameter='price&unsafe', value='0-25')


def test_repeated_control_is_resolved_inside_matching_record() -> None:
	row_a = SimpleNamespace(
		parent_node=None,
		get_all_children_text=lambda max_depth=6: 'Order 100 Complete View Order',
	)
	row_b = SimpleNamespace(
		parent_node=None,
		get_all_children_text=lambda max_depth=6: 'Order 200 Complete View Order',
	)
	control_a = node('View Order', tag='a')
	control_a.parent_node = row_a
	control_b = node('View Order', tag='a')
	control_b.parent_node = row_b
	ranked = rank_record_controls('Order 200', 'View Order', {10: control_a, 20: control_b})
	assert ranked[0][1] == 20
