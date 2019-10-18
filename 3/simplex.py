"""Collection of functions used to implement the simplex algorithm

Obviously not finished, but might get extended depending on the exercises.
"""

import numpy as np


def basis_exchange(T, v):
    """Perform basis exchange on tableau T to obtain a tableau T' with basis \
    specified by the indices in v.

    Don't modify T and return a new tableau T'"""

    # make sure we have the right number of basis variables
    assert T.shape[0] == len(v), "need m basis variables"

    # for division to work
    T_prime = np.array(T, dtype=float)

    for i, k in enumerate(v):
        make_basis_element(T_prime, k, i)

    return T_prime


def make_basis_element(T, k, i):
    """Make the k-th variable of tableau T the i-th basis element."""
    assert i >= 0 and i < T.shape[1], "Variable index needs to be valid."
    if is_standard_basis_element(T[:, k]) and T[i, k] != 1:
        raise ValueError(f"Variable {k} is already in the basis, "
                         f"but not the {i}-th basis element")
    # maybe this should check for equality with 0 instead
    assert not np.isclose(T[i,k], 0), "Need nonzero pivot element."

    T[i, :] /= T[i,k]

    for row in range(T.shape[0]):
        if row == i:
            continue
        T[row, :] -= T[row, k] * T[i, :]


def is_standard_basis_element(v):
    """Return True if v is an element of the standard basis."""
    return v.ndim == 1 and np.count_nonzero(v) == 1 and np.sum(v) == 1
