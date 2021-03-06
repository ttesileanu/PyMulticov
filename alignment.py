""" This file defines the alignment class, as well as a reference mapping object. """

from .alphabet import NumericAlphabet

import numpy as np
import pandas as pd

import copy

from scipy.spatial import distance

try:
    from numba import jit
    jit_cond = jit
    have_numba = True
except ModuleNotFoundError:
    def jit_cond(fct):
        return fct
    have_numba = False


class Alignment(object):
    """ An alignment is a list of sequences that are aligned. The sequences can be drawn from a single alphabet, or can be
    a collection of subsequences from different alphabets (e.g., a protein sequence and an RNA sequence).

    Public data members:
        - alphabets: a list of tuples of the form (alphabet_object, width) identifying the alphabets used in the
          alignment; the widths should be treated as read-only (use the 'add' method for adding data to the alignment)
        - data: a matrix containing the alignment data
        - reference: a ReferenceMapping object showing the mapping between column indices in the alignment and positions
          in some reference sequence
        - annotations: a dataframe with the same size as the alignment, containing at the least a column called 'seqw'
          for sequence weights; the number of rows should be treated as read-only
    """

    def __init__(self, data=None, alphabet=None):
        """ Initialize the alignment, potentially starting from a pre-existing alignment or data matrix.

        Can do:
          __init__(data, alphabet)
        to make an alignment based on a list of lists/strings, or a matrix, using the given alphabet.

        Or:
          __init__(alignment)
        to make a copy of the alignment.
        """

        self.alphabets = []
        self.data = np.asmatrix([])
        self.reference = ReferenceMapping()
        self.annotations = pd.DataFrame({'seqw': []})

        if data is not None:
            self.add(data, alphabet)

    def __len__(self):
        """ Get number of sequences in the alignment. """
        # len returns 1 for empty matrices, so we need to fix it a bit...
        if np.size(self.data) == 0:
            return 0
        else:
            return len(self.data)

    def __getitem__(self, idx):
        """ If used with an index for a single dimension, return a new alignment containing a subset of rows. The index
        can be an integer location, a slice, or a list.

        If used with a tuple of indices, return a squeezed array (not a matrix!) corresponding to
        `np.asarray(self.data[idx]).squeeze()`.

        If used with a string, return self.annotations[idx].
        """
        if len(self) == 0:
            raise IndexError('__getitem__ on empty alignment.')
        if isinstance(idx, tuple):
            if len(idx) > 2:
                raise TypeError('Alignment indices must be at most two-dimensional.')

            return np.asarray(self.data[idx]).squeeze()
        elif isinstance(idx, str):
            return self.annotations[idx]
        else:
            sub_align = Alignment()
            # XXX should these be copied by reference instead?
            sub_align.data = np.matrix(self.data[idx, :], copy=True)
            if np.size(sub_align.data) > 0:
                sub_align.alphabets = copy.copy(self.alphabets)
                sub_align.reference = ReferenceMapping(self.reference)
                if not hasattr(idx, '__getitem__') and type(idx) is not slice:
                    idx = [idx]
                sub_align.annotations = self.annotations.iloc[idx].copy()
                sub_align.annotations.reset_index(drop=True, inplace=True)
            else:
                sub_align.data = np.asmatrix([])
            return sub_align

    def __repr__(self):
        return "(" + repr(self.alphabets) + " x " + str(len(self)) + " seqs,\n" + repr(self.as_matrix()) + ")"

    def __eq__(self, other):
        # compare _data, reference, annotations, alphabets
        if self is other:
            return True
        if not isinstance(other, Alignment):
            return False

        if np.size(self.data) != np.size(other.data):
            return False
        if np.size(self.data) > 0 and not np.array_equal(self.data, other.data):
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

    def add(self, subdata, alphabet=None):
        """ Add some columns of data to the alignment. If `alphabet` is `None` and `subdata` behaves like an alignment,
        then it is added with the alphabet information that it contains. Otherwise `subdata` should be a data matrix or
        a list of lists/strings, which will be added with the given `alphabet`.

        If `self` is empty, sequence weight information (column 'seqw' in `annotations`) will be set to all 1s.
        """

        if len(subdata) == 0:  # nothing to do
            return self

        # length-check, unless this alignment is empty
        if len(self) != 0 and len(self) != len(subdata):
            raise ValueError('Combining alignments with different sizes.')

        if alphabet is None:
            # we're adding an alignment
            # simple check to help ensure that alphabets are valid
            align_ok = all(len(_) == 2 and _[1] > 0 for _ in subdata.alphabets)
            if not align_ok:
                raise TypeError("Argument to add is not a valid alignment.")

            # if self is empty, replace by a copy of subdata
            if len(self) == 0:
                self.data = np.matrix(subdata.data, copy=True)
                self.alphabets = list(subdata.alphabets)
                self.reference = ReferenceMapping(subdata.reference)
                self.annotations = subdata.annotations.copy()
            else:
                if self.data.dtype == subdata.data.dtype:
                    self.data = np.asmatrix(np.hstack((self.data, subdata.data)))
                else:
                    self.data = np.asmatrix(np.hstack((self.data.astype('object'), subdata.data)))
                self.alphabets.extend(subdata.alphabets)
                self.reference.extend(subdata.reference)
        else:
            # take care of one special case: turn list of strings into matrix of chars
            if isinstance(subdata[0], (str, bytes)):
                data_to_add = np.asmatrix([list(_) for _ in subdata])
            else:
                # make sure to copy the data if self was empty (not keep reference)
                data_to_add = np.matrix(subdata, copy=(len(self) == 0))

            # update the alphabets
            self.alphabets.append((alphabet, data_to_add.shape[1]))

            # update the data
            if len(self) != 0:
                if self.data.dtype == data_to_add.dtype:
                    self.data = np.asmatrix(np.hstack((self.data, data_to_add)))
                else:
                    self.data = np.asmatrix(np.hstack((self.data.astype('object'), data_to_add)))
            else:
                self.data = data_to_add
                self.annotations['seqw'] = np.ones(len(self))

            # update the reference sequence
            self.reference.append(list(range(data_to_add.shape[1])))

        return self

    def truncate_columns(self, cols, in_place=False):
        """ Generate a new alignment containing only the columns in the `cols` argument. This should be a sequence even
        if a single column is to be kept.

        The change can also be done in-place, by setting `in_place` to `True`. """

        cols = np.asarray(cols)
        if cols.dtype == bool:
            cols = cols.nonzero()[0]
        if np.any(cols < 0) or np.any(cols >= self.data.shape[1]):
            raise IndexError('Out-of-range indices in truncate_columns.')

        if not in_place:
            result = Alignment()
        else:
            result = None

        alpha_end_list = np.cumsum([_[1] for _ in self.alphabets])
        alpha_start_list = np.hstack(([0], alpha_end_list[:-1]))
        new_ref_seqs = []
        res_alphabets = []
        for alpha_info, alpha_start, alpha_end, ref_seq in zip(
                self.alphabets, alpha_start_list, alpha_end_list, self.reference.seqs):
            mask = (cols >= alpha_start) & (cols < alpha_end)
            n_crt = np.sum(mask)
            if n_crt > 0:
                crt_col_idxs = mask.nonzero()[0]
                if np.any(np.diff(crt_col_idxs) != 1):
                    raise IndexError('Attempt to split alphabet columns in two disjoint sets in truncate_columns.')
                crt_cols = cols[crt_col_idxs]
                res_alphabets.append((alpha_info[0], n_crt))
                new_ref_seqs.append(np.asarray(ref_seq)[crt_cols - alpha_start])

        if not in_place:
            result.alphabets = res_alphabets
            result.reference = ReferenceMapping(new_ref_seqs)
            result.data = self.data[:, cols]
            result.annotations = self.annotations.copy()
        else:
            self.alphabets = res_alphabets
            self.data = self.data[:, cols]
            self.reference = ReferenceMapping(new_ref_seqs)

        return result

    def swap(self, idx1, idx2):
        """ Swap the sequences at positions idx1 and idx2. """

        self.data[[idx1, idx2]] = self.data[[idx2, idx1]]
        self.annotations.iloc[[idx1, idx2]] = self.annotations.iloc[[idx2, idx1]].values

    def as_matrix(self):
        """ Get all the data as a matrix. """

        return self.data

    def get_width(self):
        return self.data.shape[1]

    def to_int(self, single_chunk=False, as_matrix=False):
        """ Get a numeric alignment from `self`. If `single_chunk == False` (the default), the structure of the
        alignment is preserved, in the sense that different numeric alphabets are used for each portion of `self` that
        had a different alphabet. If `uniform == True`, a single numeric alphabet large enough for all the data in
        `self` is employed, and the resulting alignment has only one chunk of data.

        Usually an `Alignment` object is returned, but if `as_matrix` is set to `True`, the result will be a matrix.
        This matches the `data` member of the alignment that would be generated with `single_chunk == True """

        if as_matrix:
            result = None
        else:
            result = Alignment()

        start_idx = 0
        for sub_alpha, sub_width in self.alphabets:
            subdata = self.data[:, start_idx:(start_idx + sub_width)]

            numeric_subdata = sub_alpha.to_int(subdata)

            if as_matrix:
                result = numeric_subdata if result is None else np.hstack((result, numeric_subdata))
            else:
                result.add(numeric_subdata, alphabet=NumericAlphabet(sub_alpha.size(), has_gap=sub_alpha.has_gap))

            start_idx += sub_width

        if single_chunk and not as_matrix:
            max_size = max(e[0].size() for e in self.alphabets)
            result.alphabets = [(NumericAlphabet(max_size, has_gap=all(e[0].has_gap for e in self.alphabets)),
                                 result.data.shape[1])]
            result.reference = ReferenceMapping(list(range(sum(_[1] for _ in result.alphabets))))

        return result

    @classmethod
    def from_int(cls, ndata, alphabets):
        result = cls()

        # check trivial case
        if np.size(ndata) == 0:
            return result

        # some heuristic to try to guess whether the data is a single data matrix, or a sequence of matrices
        # noinspection PyUnresolvedReferences
        if np.ndim(ndata[0]) < 2:
            # we seem to be adding only one chunk of data
            ndata = (ndata, )
        elif np.shape(ndata[0])[0] == 1:
            # we seem to be adding only one chunk of data
            ndata = (ndata, )

        # some heuristic to try to guess whether there is a single alphabet, or a sequence of alphabets
        if hasattr(alphabets, 'letters'):
            alphabets = (alphabets, )

        # if there's only one data block and several alphabets, the alphabets list should be made of tuples
        # (alphabet, alpha_width), and this should allow us to split the data appropriately
        if len(ndata) == 1 and len(alphabets) > 1:
            whole_chunk = np.asmatrix(ndata[0])
            ndata = []
            start_idx = 0
            for crt_alpha, crt_width in alphabets:
                ndata.append(whole_chunk[:, start_idx:start_idx+crt_width])
                start_idx += crt_width

        # finally convert and add all chunks with their alphabets
        for crt_ndata, crt_alpha in zip(ndata, alphabets):
            # if the alphabets contain widths in this case, ignore them
            if not hasattr(crt_alpha, 'letters') and len(crt_alpha) == 2:
                crt_alpha = crt_alpha[0]
            crt_data = crt_alpha.from_int(crt_ndata)
            result.add(crt_data, crt_alpha)

        return result

    def update_sequence_weights(self, threshold, memory_saver=None, no_numba=False):
        """ Estimate sequence weights for the alignment, using the given `threshold`.

        Setting `memory_saver` to `True` calculates the number of sequences within `threshold` of any other sequence in
        a memory efficient way (for N sequences, memory O(n) is required, as opposed to O(n^2) for the current
        approach). This is the default if Numba is present because it ends up also being faster (by about 2.5x) compared
         to the default version (which uses `scipy.spatial.distance.pdist`. Otherwise, using the memory saver option
         can incur a 2.5x penalty in speed, so when Numba is not installed or when `no_numba` is set to `True`, the
         default `memory_saver` is `False`.
        """
        if len(self) == 0:
            return self

        if not have_numba:
            no_numba = True
        if memory_saver is None:
            # the Numba routine is actually faster than pdist2, and uses far less memory
            memory_saver = not no_numba

        nalign = self.to_int(as_matrix=True)
        if not memory_saver:
            dists = distance.pdist(nalign, 'hamming')
            counts = np.sum(distance.squareform(dists) < (1 - threshold), 1)
        else:
            if not no_numba:
                counts = _get_seq_counts_memsave_numba(np.asarray(nalign), threshold)
            else:
                counts = _get_seq_counts_memsave_nonumba(np.asarray(nalign), threshold)

        self.annotations['seqw'] = 1.0 / counts

        return self

    def to_binary(self, include_gaps=False):
        from .binary import BinaryAlignment
        return BinaryAlignment.from_alignment(self, include_gaps=include_gaps)

    def eliminate_similar_sequences(self, threshold, memory_saver=None, no_numba=False):
        """ Trim the alignment by making sure no two sequences are more similar than the given threshold (in terms of
        1 - Hamming distance normalized by sequence length).

        All pairs of sequences are compared, and a graph is built with the sequences as vertices and an edge between two
        sequences if they are at least as similar as the threshold. The alignment is then trimmed by keeping a single
        representative from every connected component. The representative is chosen to have the smallest possible number
        of gaps; if several such choices exist, an arbitrary one is selected.

        Using the `memory_saver` method reduces memory usage to O(E), where E is the number of pairs of sequences that
        are more similar than the threshold, which is typically much smaller than the O(N^2) used when `memory_saver ==
        False`. This can be much slower, however, unless the Numba package is installed.
        """
        # build graph adjacency matrix
        n_seq = len(self)
        if n_seq == 0:
            return self

        if not have_numba:
            no_numba = True
        if memory_saver is None:
            memory_saver = not no_numba

        from scipy import sparse
        from scipy.sparse import csgraph
        if not memory_saver:
            seq_dists = distance.pdist(self.to_int(as_matrix=True), 'hamming')
            adjacency_matrix = sparse.csr_matrix((distance.squareform(seq_dists) < (1 - threshold)))
        else:
            if not no_numba:
                inds = _get_seq_sim_graph_memsave_numba(self.to_int(as_matrix=True), threshold)
            else:
                inds = _get_seq_sim_graph_memsave_nonumba(self.to_int(as_matrix=True), threshold)
            adjacency_matrix = sparse.csr_matrix((np.ones(len(inds[0]), dtype=bool), inds), shape=(n_seq, n_seq))

        # find connect components
        n_seq_comps, seq_labels = csgraph.connected_components(adjacency_matrix, directed=False)

        # choose representatives
        trimmed_seqs = []
        mask = np.zeros(len(self), dtype=bool)
        gap_fractions = np.mean(self.get_gap_structure(), axis=1)
        for i in range(n_seq_comps):
            k = (seq_labels == i).nonzero()[0]
            k_best = k[gap_fractions[k].argmin()]
            trimmed_seqs.append(self[k_best, :])
            mask[k_best] = True

        self.data = np.asmatrix(trimmed_seqs)
        self.annotations = self.annotations.iloc[mask].copy()
        self.annotations.reset_index(drop=True, inplace=True)

        return self

    def get_gap_structure(self):
        """ Return a boolean matrix the same size as `self.data` containing `True` at all positions where there are
        gap characters. """
        gap_structure = np.zeros(np.shape(self.data), dtype=bool)
        start_idx = 0
        for alpha, width in self.alphabets:
            if alpha.has_gap:
                gap_ch = alpha[0]
                gap_structure[:, start_idx:start_idx + width] = (self.data[:, start_idx:start_idx + width] == gap_ch)

            start_idx += width

        return gap_structure

    def extend(self, data, alphabet=None, ignore_reference=False):
        """ Add sequences at the bottom of the alignment. This can be used to add sequences from another alignment
        (when the `alphabet` argument is missing), or to add sequences from a list of strings or matrix, assuming that
        the data uses the given `alphabet`. In the latter case, sequence weights are automatically extended with 1s for
        the added sequences, while other annotations are extended with NAs.

        Both the alphabets and their widths, and the reference mapping sequences must match; otherwise `ValueError` is
        raised. Set `ignore_reference` to `True` to ignore a mismatch in the reference mapping, keeping the `reference`
        field from the original alignment.

        When adding an alignment, it is possible some annotations are present only in one alignment. The resulting
        alignment has the union of the columns from the two alignments, with missing values filled with NAs. """
        if len(self) == 0:
            # we're adding to nothing; might as well use add
            self.add(data, alphabet)
            return self

        if alphabet is None:
            # we're adding an alignment
            if len(data) == 0:
                # it's an empty alignment, so nothing changes
                return self
            if data.data.shape[1] != self.data.shape[1]:
                raise ValueError('Alignment sequences to add have different shape.')
            if data.alphabets != self.alphabets:
                raise ValueError('Alignment sequences to add have different alphabet structure.')
            if not ignore_reference and data.reference != self.reference:
                raise ValueError('Alignment sequences to add have a different mapping to reference sequence.')
            self.data = np.vstack((self.data, data.data))
            self.annotations = self.annotations.append(data.annotations, ignore_index=True)
        else:
            # we're adding a matrix/list of sequences
            if np.size(data) == 0:
                # it's empty, so nothing changes
                return self
            # turn list of strings into matrix
            if isinstance(data[0], (str, bytes)):
                data = [list(_) for _ in data]
            data = np.asarray(data)
            if data.shape[1] != self.data.shape[1]:
                raise ValueError('Alignment sequences to add have different shape.')
            if len(self.alphabets) != 1 or self.alphabets[0][0] != alphabet:
                raise ValueError('Alignment sequences to add have different alphabet structure.')
            self.data = np.vstack((self.data, data))
            self.annotations = self.annotations.append(pd.DataFrame({'seqw': np.ones(len(data))}), ignore_index=True)

        return self


@jit_cond
def _get_seq_counts_memsave_numba(data, threshold):
    n = len(data)
    w = data.shape[1]
    threshold_as_count = threshold * data.shape[1]
    counts = np.ones(n)
    for i in range(n - 1):
        for j in range(i+1, n):
            n_eq = 0
            for k in range(w):
                n_eq += (data[i, k] == data[j, k])
            if n_eq >= threshold_as_count:
                counts[i] += 1
                counts[j] += 1

    return counts


def _get_seq_counts_memsave_nonumba(data, threshold):
    n = len(data)
    threshold_as_count = threshold * data.shape[1]
    counts = np.ones(n)
    for i in range(n - 1):
        comparisons = (data[i] == data[i + 1:])
        # noinspection PyTypeChecker
        similars = (np.sum(comparisons, axis=1) >= threshold_as_count)
        counts[i + 1:] += similars
        counts[i] += np.sum(similars)

    return counts


@jit_cond
def _get_seq_sim_graph_memsave_numba(data, threshold):
    """ Returns tuple (row_ind, col_ind) such that for every k row_ind[k] and col_ind[k] are more similar than
    threshold. """
    n = len(data)
    w = data.shape[1]
    threshold_as_count = threshold * data.shape[1]
    row_ind = []
    col_ind = []
    for i in range(n - 1):
        for j in range(i+1, n):
            n_eq = 0
            for k in range(w):
                n_eq += (data[i, k] == data[j, k])
            if n_eq >= threshold_as_count:
                row_ind.append(i)
                col_ind.append(j)

    return row_ind, col_ind


def _get_seq_sim_graph_memsave_nonumba(data, threshold):
    """ Returns tuple (row_ind, col_ind) such that for every k row_ind[k] and col_ind[k] are more similar than
    threshold. """
    n = len(data)
    threshold_as_count = threshold * data.shape[1]
    row_ind = []
    col_ind = []
    for i in range(n - 1):
        comparisons = (data[i] == data[i + 1:])
        # noinspection PyTypeChecker
        similars = (np.sum(comparisons, axis=1) >= threshold_as_count).nonzero()[0]

        row_ind.extend([i]*len(similars))
        # noinspection PyTypeChecker
        col_ind.extend((i+1) + similars)

    return row_ind, col_ind


class ReferenceMapping(object):
    """ An object holding the mapping between a multi-alphabet sequence and several reference sequences.

    This is done by holding a list of lists, spanning all the columns of a multi-alphabet alignment. """

    def __init__(self, maps=None):
        if maps is None:
            self.seqs = []
        elif hasattr(maps, 'seqs'):
            self.seqs = list(maps.seqs)
        elif len(maps) > 0:
            if hasattr(maps[0], '__getitem__') and not isinstance(maps[0], (str, bytes)):
                self.seqs = list(maps)
            else:
                self.seqs = [maps]
        else:
            self.seqs = []

    def extend(self, more_seqs):
        if hasattr(more_seqs, 'seqs'):
            more_seqs = more_seqs.seqs
        self.seqs.extend(more_seqs)

    def append(self, seq):
        self.seqs.append(seq)

    def __getitem__(self, item):
        if item < 0:
            raise IndexError('Negative index in ReferenceMapping.')
        seq_end = np.cumsum([len(_) for _ in self.seqs])
        # noinspection PyUnresolvedReferences
        alpha_idx = (item < seq_end).nonzero()[0]
        if len(alpha_idx) == 0:
            raise IndexError('Out-of-range index in ReferenceMapping.')
        alpha_idx = alpha_idx[0]
        if alpha_idx > 0:
            offset = seq_end[alpha_idx - 1]
        else:
            offset = 0
        return self.seqs[alpha_idx][item - offset]

    def __len__(self):
        return sum(len(_) for _ in self.seqs)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ReferenceMapping):
            return False

        return len(self.seqs) == len(other.seqs) and all(np.array_equal(a, b) for a, b in zip(self.seqs, other.seqs))

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "ReferenceMapping(" + repr(self.seqs) + ")"
