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
__version__ = "0.1.1"

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def check_input(inA: np.array, inB: np.array, scale: int = 1_000_000) -> bool:
    """Check input method.

    Parameters
    ----------
    inA : np.array
        array containing the integer part
    inB : np.array
        array containing the decimal part
    scale : int, optional
        scale factor for the decimal parts, by default 1_000_000

    Returns
    -------
    bool
        Check if input is valid.
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

    IMPORTANT:
    ==========
    FAILS FOR LARGE ARRAYS!!!!

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
    # generate the decimal numbers
    C: np.array = A + B / 1_000_000

    # generate the upper triangular arrays
    _mul: np.array = generate_mul_triu(C)
    _add: np.array = generate_add_triu(C)
    _test: np.array = _mul - _add

    # if verbose print mul pairs
    if verbose:
        print(pairs(_test))

    # test where pairs are mul
    n_mul_pairs: int = np.where(_test[np.triu_indices_from(_test, 1)] >= 0)[0].shape[0]

    # if large number return threshold
    if n_mul_pairs > 1_000_000_000:
        return 1_000_000_000

    return n_mul_pairs


def solution_brute2(
    A: np.array,
    B: np.array,
    verbose: bool = True,
    threshold: int = 1_000_000_000,
    scale: int = 1_000_000,
) -> int:
    """Brute force method based on double for-loop.add()

    Parameters
    ----------
    A : np.array
        integer part of the decimal numbers
    B : np.array
        decimal part of the decimal numbers
    verbose : bool, optional
        Print the mul pairs, by default True
    threshold : int, optional
        Threshold for breaking the for looop, by default 1_000_000_000
    scale : int, optional
        scale factor for the decimals, by default 1_000_000

    Returns
    -------
    int
        returns the number of mul pairs of lower than threshold otherwise return threshold value
    """
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


def solution_math(
    A: np.array, B: np.array, threshold: int = 1_000_000_000, scale: int = 1_000_000
) -> int:
    """Math based method. See tutorial/examples in docs for more details.add()

    Parameters
    ----------
    A : np.array
        integer part of the decimal numbers
    B : np.array
        decimal part of the decimal numbers
    threshold : int, optional
        threshold value for the number of pairs, by default 1_000_000_000
    scale : int, optional
        scale factor for the decimals, by default 1_000_000

    Returns
    -------
    int
        returns number of mul pairs or the threshold value
    """
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
    # for 1<x<2  we need y>=x/(x-1)

    for el in C[(1 < C) & (C < 2)]:
        # I ASSUME HERE THE A AND B INDICES ARE THE SAME AS IN THE ORDERED C ARRAY
        f = el / (el - 1)

        count += C[C >= f].shape[0]

    if count > threshold:
        return threshold

    # case x>=2 and y>=2
    k = C[C >= 2.0].shape[0]

    count += k * (k - 1) / 2

    return int(count)


def compare(A: np.array, B: np.array, P: int, Q: int, scale: int = 1_000_000) -> bool:
    """Comparing composed numbers using there original integer and decimal
    values as integers.

    Parameters
    ----------
    A : np.array
        integer parts
    B : np.array
        decimal parts
    P : int
        index
    Q : int
        index
    scale : int, optional
        scale for decimals, by default 1_000_000

    Returns
    -------
    bool
        return true if multiplicative
    """
    # use associativity on integer and decimal part
    prodi = A[P] * A[Q] + (A[P] * B[Q]) // scale + (A[Q] * B[P]) // scale
    prodd = ((A[P] * B[Q]) % scale) * scale + ((A[Q] * B[P]) % scale) * scale + (B[P] * B[Q])
    print(prodi, prodd)

    sumi = A[P] + A[Q] + (B[P] + B[Q]) // scale
    sumd = ((B[P] + B[Q]) % scale) * scale

    print(sumi, sumd)

    if prodi > sumi:
        return True
    elif prodi == sumi:
        if prodd >= sumd:
            return True
    return False


def solution_math2(
    A: np.array, B: np.array, threshold: int = 1_000_000_000, scale: int = 1_000_000
) -> int:
    """Math based method. See tutorial/examples in docs for more details.add()

    Parameters
    ----------
    A : np.array
        integer part of the decimal numbers
    B : np.array
        decimal part of the decimal numbers
    threshold : int, optional
        threshold value for the number of pairs, by default 1_000_000_000
    scale : int, optional
        scale factor for the decimals, by default 1_000_000

    Returns
    -------
    int
        returns number of mul pairs or the threshold value
    """
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
    # for 1<x<2  we need y>=x/(x-1)
    # get indices of A == 1
    one_idx = np.argwhere(divmod(C, 1)[0].astype(int) == 1)[:, 0]

    for idx in one_idx:
        # A == 1
        # A.B / (A.B - 1) = 1.B / (1.B - 1) = 1.B / 0.B = 1 / 0.B + 1
        f = scale / B[idx] * scale  # max accuracy in B

        fscaled = int(f) + scale
        fscaled += 0 if np.ceil(f) else 1  # taking into account the last digit

        count += C[scale * C >= fscaled].shape[0]

    if count > threshold:
        return threshold

    # case x>=2 and y>=2
    k = C[C >= 2.0].shape[0]

    count += k * (k - 1) / 2

    return int(count)


if __name__ == "__main__":
    tup = (np.array([0, 1, 3]), np.array([0, 400_000, 500_000]), 1)
    print(solution_brute1(tup[0], tup[1], verbose=True))
    print(solution_brute2(tup[0], tup[1], verbose=True))
    print(solution_math(tup[0], tup[1]))
    print(solution_math2(tup[0], tup[1]))

# eof
