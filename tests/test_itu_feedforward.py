"""Tests for AigarthITUCl.feedforward.

Includes a regression test for the ``None`` feed: the method signature declares
``feed: tuple[int, ...] | None = None`` (so ``feedforward()`` is a documented
call), but the body dereferenced ``len(feed)`` without guarding ``None``, which
raised ``TypeError``. The neuron-level ``feedforward`` already guarded this; the
ITU-level one now matches.
"""

import pytest

from aigarth_it.itu_cl import AigarthITUCl


@pytest.fixture
def itu():
    return AigarthITUCl(itu_group="test", input_bitwidth=4, output_bitwidth=2)


def test_feedforward_with_explicit_feed(itu):
    output, stats = itu.feedforward((1, -1, 0, 1))
    assert len(output) == itu.meta.output_bitwidth
    assert set(output) <= {-1, 0, 1}


def test_feedforward_with_no_argument_does_not_crash(itu):
    """Regression: feedforward() with the default None feed must not raise."""
    output, _ = itu.feedforward()
    assert len(output) == itu.meta.output_bitwidth


def test_feedforward_with_explicit_none_does_not_crash(itu):
    output, _ = itu.feedforward(None)
    assert len(output) == itu.meta.output_bitwidth


def test_feedforward_with_empty_feed(itu):
    output, _ = itu.feedforward(())
    assert len(output) == itu.meta.output_bitwidth


def test_feedforward_none_and_empty_feed_agree(itu):
    """A None feed is treated as an empty feed, so the two must behave alike."""
    out_none, _ = itu.feedforward(None)
    out_empty, _ = itu.feedforward(())
    assert out_none == out_empty


def test_feedforward_longer_feed_is_accepted(itu):
    # More feed values than input neurons: the surplus is ignored, no crash.
    output, _ = itu.feedforward((1, -1, 1, -1, 1, -1, 1))
    assert len(output) == itu.meta.output_bitwidth
