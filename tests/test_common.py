"""Tests for aigarth_it.common — ternary clamp and signed bit-string conversion.

These pin behaviour that the rest of the tissue relies on: every neuron state
and weight passes through ``ternary_clamp``, and the bit-string helpers are the
on-the-wire integer encoding. All assertions below hold on current ``main``.
"""

import pytest

from aigarth_it.common import (
    bitstring64_to_int,
    int_to_bitstring,
    ternary_clamp,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (-5, -1),
        (-1, -1),
        (0, 0),
        (1, 1),
        (5, 1),
        (1_000_000, 1),
        (-1_000_000, -1),
    ],
)
def test_ternary_clamp(value, expected):
    assert ternary_clamp(value) == expected


def test_ternary_clamp_only_returns_trits():
    assert {ternary_clamp(v) for v in range(-20, 21)} == {-1, 0, 1}


@pytest.mark.parametrize("const_len", [True, False])
@pytest.mark.parametrize(
    "number",
    [0, 1, -1, 2, -2, 4, -4, 5, -5, 127, -128, 255, -256, 2**31, -(2**31), 2**62, -(2**63), 2**63 - 1],
)
def test_bitstring_round_trip(number, const_len):
    """int -> signed bit string -> int is lossless in both length modes."""
    encoded = int_to_bitstring(number, max_bits=64, const_len=const_len)
    assert bitstring64_to_int(encoded) == number


def test_int_to_bitstring_const_len_width():
    assert int_to_bitstring(5, max_bits=8, const_len=True) == "00000101"


def test_int_to_bitstring_rejects_out_of_range():
    with pytest.raises(ValueError):
        int_to_bitstring(2**63, max_bits=64)
    with pytest.raises(ValueError):
        int_to_bitstring(-(2**63) - 1, max_bits=64)


def test_bitstring64_to_int_two_complement():
    # 4-bit two's complement: '1100' == -4, '0101' == 5
    assert bitstring64_to_int("1100") == -4
    assert bitstring64_to_int("0101") == 5


def test_bitstring64_to_int_rejects_signed_and_oversized():
    with pytest.raises(ValueError):
        bitstring64_to_int("-5")
    with pytest.raises(ValueError):
        bitstring64_to_int("1" * 65)
