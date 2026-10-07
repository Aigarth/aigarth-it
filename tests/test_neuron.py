"""Tests for aigarth_it.neuron_cl.AITClNeuron.

Covers the 'Circle' neuron's dot-product feedforward, its handling of feed/weight
length mismatches, the backward/forward input split, and state commit semantics.
All assertions hold on current ``main``.
"""

import pytest

from aigarth_it.exceptions import AigarthITUError
from aigarth_it.neuron_cl import AITClNeuron


def test_weights_are_clamped_to_trits():
    n = AITClNeuron(input_weights=[5, -3, 0, 2])
    assert n.input_weights == [1, -1, 0, 1]


def test_empty_or_none_weights_default_to_single_unit_weight():
    assert AITClNeuron(input_weights=None).input_weights == [1]
    assert AITClNeuron(input_weights=[]).input_weights == [1]


def _forward(neuron, feed):
    neuron.feedforward(feed)
    state, _ = neuron.commit_state()
    return state


def test_feedforward_dot_product_then_clamp():
    n = AITClNeuron(input_weights=[1, -1, 1])
    assert _forward(n, (1, 1, 1)) == 1      # 1 - 1 + 1 = 1
    assert _forward(n, (1, 1, -1)) == -1     # 1 - 1 - 1 = -1
    assert _forward(n, (0, 0, 0)) == 0       # 0


def test_feedforward_longer_feed_is_truncated_to_weights():
    n = AITClNeuron(input_weights=[1, -1, 1])
    # Only the first three feed values are used; the trailing ones are ignored.
    assert _forward(n, (1, 1, 1, 1, 1)) == 1


def test_feedforward_shorter_feed_is_zero_padded():
    n = AITClNeuron(input_weights=[1, -1, 1])
    # Missing inputs contribute 0: 1*1 = 1.
    assert _forward(n, (1,)) == 1


def test_feedforward_accepts_none_feed():
    n = AITClNeuron(input_weights=[1, 1])
    assert _forward(n, None) == 0


def test_commit_without_feedforward_raises():
    n = AITClNeuron(input_weights=[1])
    with pytest.raises(AigarthITUError):
        n.commit_state()


def test_commit_reports_state_change():
    n = AITClNeuron(input_weights=[1])
    n.feedforward((1,))
    state, changed = n.commit_state()
    assert state == 1 and changed is True
    # Committing the same value again reports no change.
    n.feedforward((1,))
    state, changed = n.commit_state()
    assert state == 1 and changed is False


@pytest.mark.parametrize(
    "count,skew",
    [(1, 0), (4, 0), (5, 0), (4, 2), (4, -2), (4, 10), (4, -10), (2, 1)],
)
def test_input_split_index_stays_in_bounds(count, skew):
    idx = AITClNeuron._get_input_split_index(count, skew)
    assert 0 <= idx <= count


def test_input_split_index_rejects_non_positive_count():
    with pytest.raises(ValueError):
        AITClNeuron._get_input_split_index(0, 0)
