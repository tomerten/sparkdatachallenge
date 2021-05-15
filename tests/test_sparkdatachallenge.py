# -*- coding: utf-8 -*-

"""Tests for sparkdatachallenge package."""

from re import I

import numpy as np
import pytest
import sparkdatachallenge

incheck_pass = [
    (np.array([1]), np.array([2]), True),
    (np.array([1]), np.array([1, 2]), False),
    (np.array([1002]), np.array([1, 2]), False),
    (np.array([-1]), np.array([1, 2]), False),
    (np.array([-1]), np.array([1]), False),
    (np.array([1]), np.array([-1]), False),
    (np.array([]), np.array([]), False),
    (np.array([1]), np.array([-1]), False),
    (np.array([1]), np.array([1_000_000]), False),
    (np.array([1]), np.array([0]), True),
]

incheck_fail = [
    (None, None),
    (None, np.array([1])),
    (np.array([1]), None),
]

simple = [
    (np.array([0, 1, 2, 2, 3, 5]), np.array([500_000, 500_000, 0, 0, 0, 20_000]), 8),
    (np.array([0, 0, 1, 2, 2, 3, 5]), np.array([0, 500_000, 500_000, 0, 0, 0, 20_000]), 8),
    (np.array([0, 0, 0, 1, 2, 2, 3, 5]), np.array([0, 0, 500_000, 500_000, 0, 0, 0, 20_000]), 9),
    (np.array([1, 3] * int(10 ** 0 / 2)), np.array([500_000, 0] * int(10 ** 0 / 2)), 0),
    (np.array([1, 3] * int(10 ** 1 / 2)), np.array([500_000, 0] * int(10 ** 1 / 2)), 35),
    (np.array([1, 3] * int(10 ** 2 / 2)), np.array([500_000, 0] * int(10 ** 2 / 2)), 3725),
    (np.array([1, 3] * int(10 ** 3 / 2)), np.array([500_000, 0] * int(10 ** 3 / 2)), 374750),
    (np.array([1, 3] * int(10 ** 4 / 2)), np.array([500_000, 0] * int(10 ** 4 / 2)), 37497500),
    (np.array([1, 3] * int(10 ** 5 / 2)), np.array([500_000, 0] * int(10 ** 5 / 2)), 1000000000),
]


@pytest.mark.parametrize("ina, inb, res", incheck_pass)
def test_input_check_pass(ina, inb, res):
    assert res == sparkdatachallenge.check_input(ina, inb)


@pytest.mark.parametrize("ina, inb", incheck_fail)
def test_input_check_fail(ina, inb):
    with pytest.raises(TypeError):
        sparkdatachallenge.check_input(ina, inb)


@pytest.mark.parametrize("ina, inb, res", simple)
def test_simple(ina, inb, res):
    # assert res == sparkdatachallenge.solution_brute1(ina, inb, verbose=False) - fails on memory allocation
    # assert res == sparkdatachallenge.solution_brute2(ina, inb, verbose=False) - takes a bit
    assert res == sparkdatachallenge.solution_math(ina, inb)


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (otherwise all tests are normally run with pytest)
# Make sure that you run this code with the project directory as CWD, and
# that the source directory is on the path
# ==============================================================================
# if __name__ == "__main__":
#    the_test_you_want_to_debug = test_hello_noargs
#
#    print("__main__ running", the_test_you_want_to_debug)
#    the_test_you_want_to_debug()
#    print('-*# finished #*-')

# eof
