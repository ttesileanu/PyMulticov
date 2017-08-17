""" This file defines the binary alignment class. """

from multicov.alignment import ReferenceMapping

from scipy import sparse

import numpy as np
import pandas as pd


class BinaryAlignment(object):
    def __init__(self, data=None, alphabet=None):
        self.data = sparse.coo_matrix([])
        self.reference = ReferenceMapping()
        self.alphabets = []
        self.annotations = pd.DataFrame({'seqw': []})

        if data is not None:
            self.add(data, alphabet)

    def add(self, data, alphabet=None):
        """ Add a new chunk of data, with the given alphabet. `data` can be another binary alignment, or a matrix. """
        if hasattr(data, 'annotations'):
            if len(data) == 0:
                return self
        elif np.size(data) == 0:
            return self

        # length-check, unless this alignment is empty
        old_n_rows = len(self)
        # noinspection PyUnresolvedReferences
        new_n_rows = len(data) if hasattr(data, 'annotations') else np.shape(data)[0]
        if old_n_rows != 0 and old_n_rows != new_n_rows:
            raise ValueError('Combining binary alignments with different sizes.')

        if alphabet is None:
            # adding another alignment
            if old_n_rows == 0:
                self.data = data.data.copy()
                self.annotations = data.annotations.copy()
            else:
                self.data = sparse.coo_matrix(sparse.hstack((self.data, data.data)))

            self.reference.extend(data.reference)
            self.alphabets.extend(data.alphabets)
        else:
            # adding a data matrix
            if old_n_rows == 0:
                self.data = sparse.coo_matrix(data, copy=True)
                self.annotations['seqw'] = np.ones(len(self))
            else:
                self.data = sparse.coo_matrix(sparse.hstack((self.data, data)))

            n_letters = alphabet.size(no_gap=True)
            # noinspection PyUnresolvedReferences
            n_binpos = np.shape(data)[1]
            if n_binpos % n_letters != 0:
                raise ValueError('Binary matrix size is not divisible by alphabet size.')
            n_sites = n_binpos // n_letters
            self.alphabets.append((alphabet, n_sites))
            self.reference.append(list(range(n_sites)))

        return self

    def __len__(self):
        # first dimension shape is 1 for empty matrices, so we need to fix it a bit...
        if np.size(self.data) == 0:
            return 0
        else:
            return self.data.shape[0]

    @classmethod
    def from_alignment(cls, align):
        result = cls()

        if len(align) == 0 or len(align.alphabets) == 0:
            return result

        start_idx = 0
        bin_blocks = []
        for alpha, alpha_width in align.alphabets:
            subdata = align.data[:, start_idx:start_idx+alpha_width]
            bin_subdata = _make_binary_from_mat(subdata, alpha)
            bin_blocks.append(bin_subdata)
            start_idx += alpha_width

        result.data = sparse.hstack(bin_blocks)
        result.reference = align.reference
        result.alphabets = align.alphabets
        result.annotations = align.annotations

        return result

    def __eq__(self, other):
        # compare _data, reference, annotations, alphabets
        if self is other:
            return True
        if not isinstance(other, BinaryAlignment):
            return False

        if np.shape(self.data) != np.shape(other.data):
            return False
        if np.size(self.data) > 0:
            my_data = sparse.csr_matrix(self.data)
            other_data = sparse.csr_matrix(other.data)
            diff_data = np.abs(my_data - other_data)
            if diff_data.max() > 1e-8:
                return False

        if not self.annotations.equals(other.annotations):
            return False

        if len(self.alphabets) != len(other.alphabets):
            return False
        for (alpha1, alpha2) in zip(self.alphabets, other.alphabets):
            if not alpha1 == alpha2:
                return False

        if self.reference != other.reference:
            return False

        return True

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "(" + repr(self.alphabets) + " x " + str(len(self)) + " seqs,\n" + repr(self.data) + ")"

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.annotations[item]
        else:
            raise IndexError('Trying to index BinaryAlignment by non-string argument.')

    def index_map(self, idx=None):
        """ Map indices from character alignments to the corresponding ranges in the binary alignment.

        If `idx` is not given, a full map is returned as an Nx2 array, such that `map[i]` is the binary index range for
        position `i`.
        If `idx` is given, only the contents of `map[idx]` are returned. Generating the full map is likely faster than
        looping over the index. """
        return binary_index_map(self, idx)


def _make_binary_from_mat(m, alphabet):
    """ Create a sparse binary alignment matrix from the given dense matrix, using the given alphabet. Gaps are
    represented as all zeros. """

    # this is about 30% slower than the Matlab version; maybe it's better to do it like in Matlab, where the binary
    # alignment is first created as a dense matrix, and then sparsified?

    letters = alphabet.letters(no_gap=True)
    nletts = len(letters)
    # XXX maybe try to use normal lists here instead of arrays? they should be more efficient at resizing
    i_list = np.array([])
    j_list = np.array([])
    for lett_idx, letter in enumerate(letters):
        # noinspection PyUnresolvedReferences
        idxs = (m == letter).nonzero()
        i_list = np.hstack((i_list, idxs[0]))
        j_list = np.hstack((j_list, idxs[1]*nletts + lett_idx))

    n, l = m.shape
    data = np.ones(len(i_list), float)
    return sparse.coo_matrix((data, (i_list, j_list)), shape=(n, l*nletts))


def binary_index_map(align, idx=None):
    """ Map indices from character alignments to the corresponding slices in the binary alignment.

    If `idx` is not given, a full map is returned as an Nx2 array, such that `map[i]` is the binary index range for
    position `i`.
    If `idx` is given, only the contents of `map[idx]` are returned. Generating the full map is likely faster than
    looping over the index.

    The first argument can be any object that has an `alphabets` field. """
    alphabets = align.alphabets
    if len(alphabets) == 0:
        if idx is not None:
            raise IndexError('index_map called with out-of-range index.')
        else:
            return []

    widths = np.asarray([_[1] for _ in alphabets])
    start_idxs = np.hstack(([0], np.cumsum(widths)))

    bin_widths = np.asarray([width*alphabet.size(no_gap=True) for alphabet, width in alphabets])
    start_bin_idxs = np.hstack(([0], np.cumsum(bin_widths)))
    if idx is not None:
        if idx < 0 or idx >= start_idxs[-1]:
            raise IndexError('index_map called with out-of-range index.')
        alpha_idx = (idx >= start_idxs).nonzero()[0][-1]
        alpha_loc = idx - start_idxs[alpha_idx]
        alpha_len = alphabets[alpha_idx][0].size(no_gap=True)
        start_bin_idx = start_bin_idxs[alpha_idx] + alpha_loc*alpha_len
        end_bin_idx = start_bin_idx + alpha_len
        return (start_bin_idx, end_bin_idx)
    else:
        width = np.sum(widths)
        full_map = np.zeros((width, 2), dtype=int)
        crt_map_row = 0
        for start_bin_idx, alpha_info in zip(start_bin_idxs, alphabets):
            alpha_len = alpha_info[0].size(no_gap=True)
            alpha_width = alpha_info[1]
            crt_rows = slice(crt_map_row, crt_map_row + alpha_width)
            full_map[crt_rows, 0] = start_bin_idx + np.arange(alpha_width, dtype=int)*alpha_len
            full_map[crt_rows, 1] = full_map[crt_rows, 0] + alpha_len

            crt_map_row += alpha_width

        return full_map
