# -*- coding: utf-8 -*-
"""
Package sparkdatachallenge
==========================
Arrays A and B consisting of N non-negative integers are given. 
Together, they represent N real numbers, denoted as C[0], ..., C[N−1]. 
Elements of A represent the integer parts and the corresponding elements of
B (divided by 1,000,000) represent the fractional parts of the elements of C. 

A[I] and B[I] represent 
C[I] = A[I] + B[I] / 1,000,000.

##
A pair of indices (P, Q) is multiplicative if 0 ≤ P < Q < N and C[P] * C[Q] ≥ C[P] + C[Q].
##

The package contains several methods to find the number of multiplicative pairs in C.

"""
__version__ = "0.1.0"

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def check_input(inA, inB, scale=1_000_000) -> bool:
    """Method to check input.

    Parameters
    ----------
    inA : np,array
        input array integer part
    inB : np.array
        input array fractional part

    Returns
    -------
    bool
        True or False if input is valid.
    """
    checkA = (
        (inA >= 0).all() and (inA <= 1000).all() and not np.isnan(inA).any() and not inA.size == 0
    )
    checkB = (
        (inB >= 0).all() and (inB < scale).all() and not np.isnan(inB).any() and not inB.size == 0
    )
    checkC = inA.shape == inB.shape

    return checkA and checkB and checkC


def generate_mul_triu(C: np.array) -> np.array:
    """Method to return an upper triangular array,
    containing the element by element products of a
    given input array C. The upper triangular part
    comes from the fact we only want products
    where col_idx > row_idx (hence k=-1) as C
    is assumed to be an non-decreasing array of decimal numbers
    adn where are looking for multiplicative pairs.

    Parameters:
    -----------
    C : np.array
        non-decreasing array of decimal numbers

    Returns
    -------
    np.array
        upper triangular array of element by element products
    """
    out = C[:, np.newaxis] * C
    return np.tril(out, k=-1).T


def generate_add_triu(C):
    """Method to return an upper triangular array,
    containing the element by element sums of a
    given input array C. The upper triangular part
    comes from the fact we only want products
    where col_idx > row_idx (hence k=-1) as C
    is assumed to be an non-decreasing array of decimal numbers
    adn where are looking for multiplicative pairs.

    Parameters
    ----------
    C : np.array
        non-decreasing array of decimal numbers

    Returns
    -------
    np.array
        upper triangular array of element by element sums
    """
    out = C[:, np.newaxis] + C
    return np.tril(out, k=-1).T


def pairs(M: np.array) -> List[tuple]:
    """Method to generate the multiplicative pairs.add()

    Parameters
    ----------
    M : np.array
        Array containing inequality values.

    Returns
    -------
    List[tuple]
        List of pairs as tuples.
    """
    # list of indices that obey criterium
    _sel_idx = np.argwhere(M >= 0)

    # indices from upper triangle
    triuidx = np.vstack(np.triu_indices_from(M, 1)).T

    # indices that are multiplicative
    _pairs = _sel_idx[(_sel_idx[:, None] == triuidx).all(-1).any(-1)]
    _pairs = [tuple(t) for t in _pairs.tolist()]

    return _pairs


def solution_brute1(A: np.array, B: np.array, verbose: bool = True) -> int:
    """Brute force method one - using upper triangular matrices. Expected
    to fail with large arrays and it does due to memory issues !!!!

    Parameters
    ----------
    A : np.array
        Integer part array
    B : np.array
        Decimal part array
    verbose : bool, optional
        to print out of pairs, by default True

    Returns
    -------
    int
        number of multiplicative pairs
    """
    C: np.array = A + B / 1_000_000

    _mul: np.array = generate_mul_triu(C)
    _add: np.array = generate_add_triu(C)
    _test: np.array = _mul - _add

    if verbose:
        print(pairs(_test))

    n_mul_pairs: int = np.where(_test[np.triu_indices_from(_test, 1)] >= 0)[0].shape[0]
    if n_mul_pairs > 1_000_000_000:
        return 1_000_000_000
    return n_mul_pairs


def solution_brute2(A, B, verbose=True, threshold=1_000_000_000, scale=1_000_000):

    # generate the floats and sort
    C: np.array = np.sort(A + B / scale)

    # size
    N = A.shape[0]

    # init
    tups = []
    counter = 0

    # double for loop - not very efficient
    for P in range(N):
        # use the sorting to reduce second for loop
        for Q in range(P + 1, N):
            if (C[P] * C[Q]) >= (C[P] + C[Q]):
                counter += 1
                if counter == threshold:
                    return threshold
                if verbose:
                    tups.append((P, Q))

    # if verbose print the mul pairs
    if verbose:
        print(tups)

    return counter


def solution_math(A, B, threshold=1_000_000_000, scale=1_000_000):
    C: np.array = np.sort(A + B / scale)

    # init count
    count = 0
    # x == 0 => y ==0 => count  C[i] = 0.0
    nzero = C[C == 0.0].shape[0]

    # calculate the number of zero - zero pairs and update count
    if nzero > 1:
        count = nzero * (nzero - 1) / 2

    if count > threshold:
        return threshold
    # 0 < x < 1 => no solution

    # x==1 => no solution

    #  1 < x < 2 => y >= x / (x-1)
    # inequality is always satisfied for x, y >= 2
    # for 1<x<2  we need y>=2
    for el in C[(1 < C) & (C < 2)]:
        f = el / (el - 1)
        count += C[C >= f].shape[0]

    if count > threshold:
        return threshold

    # combine case 4 and 5 x>=2 and y>=2
    k = C[C >= 2.0].shape[0]

    count += k * (k - 1) / 2

    return int(count)


def main(A: np.array, B: np.array):

    pass


def hello(who="world"):
    """'Hello world' method.

    :param str who: whom to say hello to
    :returns: a string
    """
    result = "Hello " + who
    return result


if __name__ == "__main__":
    main()

# eof
