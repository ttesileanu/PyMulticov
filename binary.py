""" This file defines the binary alignment class. """

from multicov.alignment import ReferenceMapping

from scipy import sparse

import numpy as np
import pandas as pd


class BinaryAlignment(object):
    def __init__(self, data=None, alphabet=None, include_gaps=None):
        self.data = sparse.coo_matrix([])
        self.reference = ReferenceMapping()
        self.alphabets = []
        self.annotations = pd.DataFrame({'seqw': []})

        if include_gaps is None:
            if hasattr(data, 'include_gaps'):
                include_gaps = data.include_gaps
            else:
                include_gaps = False

        self.include_gaps = include_gaps

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

            # make sure the other alignment has the same state of include_gaps
            # XXX this is a pretty dumb way of doing it
            if data.include_gaps and not self.include_gaps:
                data = BinaryAlignment(data)
                data.remove_gap_positions()
            elif not data.include_gaps and self.include_gaps:
                data = BinaryAlignment(data)
                data.add_gap_positions()

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

            n_letters = alphabet.size(no_gap=(not self.include_gaps))
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
    def from_alignment(cls, align, include_gaps=None):
        """ By default this function does not generate columns for gaps, unless the `align` argument has a member
        `include_gaps` that is set to `True`.  The `include_gaps` argument can be used to override this behavior. It
        also overrides any value contained by the `include_gaps` member of `align`, if it exists. """
        result = cls()

        if len(align) == 0 or len(align.alphabets) == 0:
            return result

        if include_gaps is None:
            if hasattr(align, 'include_gaps'):
                include_gaps = align.include_gaps
            else:
                include_gaps = False

        start_idx = 0
        bin_blocks = []
        for alpha, alpha_width in align.alphabets:
            subdata = align.data[:, start_idx:start_idx+alpha_width]
            bin_subdata = _make_binary_from_mat(subdata, alpha, include_gaps)
            bin_blocks.append(bin_subdata)
            start_idx += alpha_width

        result.data = sparse.hstack(bin_blocks)
        result.reference = align.reference
        result.alphabets = align.alphabets
        result.annotations = align.annotations
        result.include_gaps = include_gaps

        return result

    def __eq__(self, other):
        # compare _data, reference, annotations, alphabets
        if self is other:
            return True
        if not isinstance(other, BinaryAlignment):
            return False

        if np.shape(self.data) != np.shape(other.data):
            return False
        if self.include_gaps != other.include_gaps:
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

    def index_map(self, idx=None, include_gaps=None):
        """ Map indices from character alignments to the corresponding ranges in the binary alignment.

        If `idx` is not given, a full map is returned as an Nx2 array, such that `map[i]` is the binary index range for
        position `i`.
        If `idx` is given, only the contents of `map[idx]` are returned. Generating the full map is likely faster than
        looping over the index. """
        return binary_index_map(self, idx, include_gaps=include_gaps)

    def add_gap_positions(self):
        """ Add columns for gaps, and set `include_gaps` to `True`. If `include_gaps` is already `True`, this function
        does nothing.

        Returns `self`. """
        if self.include_gaps:
            return self

        # XXX this is a pretty dumb way to do it
        align = self.to_alignment()
        self.data = BinaryAlignment.from_alignment(align, include_gaps=True).data

        self.include_gaps = True

    def remove_gap_positions(self):
        """ Remove columns for gaps, and set `include_gaps` to `False`. If `include_gaps` is already `False`, this
        functions does nothing.

        Returns `self`. """
        if not self.include_gaps:
            return self

        # XXX this is a pretty dumb way to do it
        align = self.to_alignment()
        self.data = BinaryAlignment.from_alignment(align).data

        self.include_gaps = False

    def to_alignment(self):
        # _binary_mat_to_char_mat(m, alphabet, include_gaps):
        from multicov.alignment import Alignment
        res = Alignment()

        # transform the data
        crt_idx = 0
        m_all = np.asarray(self.data.todense())
        for alphabet, alphawidth in self.alphabets:
            nletts = len(alphabet.letters(no_gap=(not self.include_gaps)))
            width = alphawidth*nletts

            m = m_all[:, crt_idx:crt_idx+width]
            res.add(_binary_mat_to_char_mat(m, alphabet, self.include_gaps), alphabet)

            crt_idx += width

        # copy reference mapping, annotations
        res.reference = self.reference
        res.annotations = self.annotations

        return res


def _make_binary_from_mat(m, alphabet, include_gaps):
    """ Create a sparse binary alignment matrix from the given dense matrix, using the given alphabet. Gaps are
    represented as all zeros. """

    # this is about 30% slower than the Matlab version; maybe it's better to do it like in Matlab, where the binary
    # alignment is first created as a dense matrix, and then sparsified?

    letters = alphabet.letters(no_gap=(not include_gaps))
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


def _binary_mat_to_char_mat(m, alphabet, include_gaps):
    """ Convert a binary alignment matrix (in dense format) to the corresponding character alignment in the given
    alphabet, assuming gaps were or were not included (according to `include_gaps`). """
    letters = alphabet.letters(no_gap=(not include_gaps))
    nletts = len(letters)
    m_3d = np.reshape(m, (m.shape[0], -1, nletts))
    if include_gaps or not alphabet.has_gap:
        m_3d_num = np.argmax(m_3d, axis=2)
    else:
        m_3d_num_no_gap = np.argmax(m_3d, axis=2) + 1
        m_3d_nongap_mask = np.any((m_3d > 0), axis=2)
        m_3d_num = m_3d_num_no_gap * m_3d_nongap_mask

    return alphabet.from_int(m_3d_num.astype(int))


def binary_index_map(align, idx=None, include_gaps=None):
    """ Map indices from character alignments to the corresponding slices in the binary alignment.

    If `idx` is not given, a full map is returned as an Nx2 array, such that `map[i]` is the binary index range for
    position `i`.
    If `idx` is given, only the contents of `map[idx]` are returned. Generating the full map is likely faster than
    looping over the index.

    The first argument can be any object that has an `alphabets` field.

    By default this function does not generate indices for gaps, unless the first argument has a member `include_gaps`
    that is set to `True`.  The `include_gaps` argument can be used to override this behavior. It also overrides any
    value contained by the `include_gaps` member of `align`, if it exists. """
    alphabets = align.alphabets
    if len(alphabets) == 0:
        if idx is not None:
            raise IndexError('index_map called with out-of-range index.')
        else:
            return []

    if include_gaps is None:
        if hasattr(align, 'include_gaps'):
            include_gaps = align.include_gaps
        else:
            include_gaps = False

    widths = np.asarray([_[1] for _ in alphabets])
    start_idxs = np.hstack(([0], np.cumsum(widths)))

    bin_widths = np.asarray([width*alphabet.size(no_gap=(not include_gaps)) for alphabet, width in alphabets])
    start_bin_idxs = np.hstack(([0], np.cumsum(bin_widths)))
    if idx is not None:
        if idx < 0 or idx >= start_idxs[-1]:
            raise IndexError('index_map called with out-of-range index.')
        alpha_idx = (idx >= start_idxs).nonzero()[0][-1]
        alpha_loc = idx - start_idxs[alpha_idx]
        alpha_len = alphabets[alpha_idx][0].size(no_gap=(not include_gaps))
        start_bin_idx = start_bin_idxs[alpha_idx] + alpha_loc*alpha_len
        end_bin_idx = start_bin_idx + alpha_len
        return (start_bin_idx, end_bin_idx)
    else:
        width = np.sum(widths)
        full_map = np.zeros((width, 2), dtype=int)
        crt_map_row = 0
        for start_bin_idx, alpha_info in zip(start_bin_idxs, alphabets):
            alpha_len = alpha_info[0].size(no_gap=(not include_gaps))
            alpha_width = alpha_info[1]
            crt_rows = slice(crt_map_row, crt_map_row + alpha_width)
            full_map[crt_rows, 0] = start_bin_idx + np.arange(alpha_width, dtype=int)*alpha_len
            full_map[crt_rows, 1] = full_map[crt_rows, 0] + alpha_len

            crt_map_row += alpha_width

        return full_map
