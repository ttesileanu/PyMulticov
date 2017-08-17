""" Define a class that calculates statistics for multiple sequence alignments, as well as a class that helps fit a
maximum entropy model to data. """

import numpy as np

from scipy import sparse

from binary import binary_index_map


# noinspection SpellCheckingInspection
class Statistics(object):
    def __init__(self, alignment, precompute=(), regularizer='pseudocount', regularization_amount=0):
        """ Initialize the statistics structure, attaching it to the given alignment. This can be either a character
        `Alignment` or a `BinaryAlignment`.

        The evaluation of the actual statistics is delayed until they are requested. This means that their values
        might be different if the alignment changed in the meantime. Once one of the statistics is calculated, it is
        never automatically recalculated, so changes to the alignment that are made after the first access to the
        statistics will not be reflected in the statistics. See `update`. Note that unexpected results can occur if, for
        example, `freq1` is updated first, and then, before `cmat` is called, the alignment changes.

        Set `precompute` to a tuple containing one or several of 'freq1', 'freq2', 'cmat' to compute these values at
        the time of construction.

        Set `regularization_amount` to a value between 0 and 1 to apply some amount of pseudocount regularization. """
        if hasattr(alignment, 'to_binary'):
            self.alignment = alignment
            self.bin_align = None
        else:
            self.alignment = None
            self.bin_align = alignment

        self.alphabets = alignment.alphabets
        self.reference = alignment.reference
        self.annotations = alignment.annotations

        self._freq1 = None
        self._freq2 = None
        self._cmat = None

        self.regularizer = regularizer
        self.regularization_amount = regularization_amount

        self.update(precompute)

    @property
    def freq1(self):
        if self._freq1 is None:
            self._freq1 = self._calculate_freq1()
        return self._freq1

    @property
    def freq2(self):
        if self._freq2 is None:
            self._freq2 = self._calculate_freq2()
        return self._freq2

    @property
    def cmat(self):
        """ Note that while this needs to calculate both `freq1` and `freq2` as intermediate values, it does not store
        them. It does, however, use them if they are already stored, avoiding a potentially expensive calculation. """
        if self._cmat is None:
            self._cmat = self._calculate_cmat()
        return self._cmat

    def _is_empty(self):
        if self.bin_align is not None and len(self.bin_align) == 0:
            return True
        if self.alignment is not None and len(self.alignment) == 0:
            return True
        if self.bin_align is None and self.alignment is None:
            return True
        return False

    # noinspection SpellCheckingInspection
    def _calculate_freq1(self):
        if self._is_empty():
            return np.asarray([])

        if self.bin_align is None:
            self.bin_align = self.alignment.to_binary()

        bin_align = self.bin_align

        seqw = self.annotations['seqw'].as_matrix()
        n_eff = np.sum(seqw)

        freq1 = seqw * bin_align.data / n_eff

        if self.regularization_amount > 0:
            freq1 = self._regularize_freq1(freq1)

        return freq1

    def _calculate_freq2(self):
        if self._is_empty():
            return np.asarray([])

        if self.bin_align is None:
            self.bin_align = self.alignment.to_binary()

        bin_align = self.bin_align

        seqw = self.annotations['seqw'].as_matrix()
        n_eff = np.sum(seqw)

        bindata = bin_align.data
        seqw_mat = sparse.diags(seqw, format='csr')

        freq2 = (bindata.T * seqw_mat * bindata).todense() / n_eff

        if self.regularization_amount > 0:
            freq2 = self._regularize_freq2(freq2)

        return freq2

    def _calculate_cmat(self):
        if self._is_empty():
            return np.asarray([])

        freq1 = self._freq1 if self._freq1 is not None else self._calculate_freq1()
        freq2 = self._freq2 if self._freq2 is not None else self._calculate_freq2()
        return freq2 - np.outer(freq1, freq1)

    def _regularize_freq1(self, freq1):
        if self.regularizer != 'pseudocount':
            raise ValueError("Only 'pseudocount' regularizer is supported. {} requested.".format(str(self.regularizer)))

        bkg_freq1 = np.hstack(np.ones(width * alphabet.size(no_gap=True)) / alphabet.size()
                              for alphabet, width in self.alphabets)
        return (1 - self.regularization_amount) * freq1 + self.regularization_amount * bkg_freq1

    def _regularize_freq2(self, freq2):
        if self.regularizer != 'pseudocount':
            raise ValueError("Only 'pseudocount' regularizer is supported. {} requested.".format(str(self.regularizer)))

        bkg_freq1 = np.hstack(np.ones(width * alphabet.size(no_gap=True)) / alphabet.size()
                              for alphabet, width in self.alphabets)
        bkg_freq2 = np.outer(bkg_freq1, bkg_freq1)

        # bkg_freq2 needs to be corrected on the diagonal blocks, because those
        # elements are not independent
        crt_col = 0
        for alphabet, width in self.alphabets:
            for i in range(width):
                crt_nlett_ng = alphabet.size(no_gap=True)
                idxs = slice(crt_col, crt_col + crt_nlett_ng)

                bkg_freq2[idxs, idxs] = np.diag(bkg_freq1[idxs])

                crt_col += crt_nlett_ng

        return (1 - self.regularization_amount) * freq2 + self.regularization_amount * bkg_freq2

    def update(self, what):
        for crt_what in what:
            if crt_what == 'freq1':
                self._freq1 = self._calculate_freq1()
            elif crt_what == 'freq2':
                self._freq2 = self._calculate_freq2()
            elif crt_what == 'cmat':
                self._cmat = self._calculate_cmat()
            else:
                raise ValueError("Unknown statistics update target: " + str(crt_what) + ".")

    def __getitem__(self, item):
        """ Indexing the statistics object with a string key is equivalent to indexing the annotations,
        `self[key] is self.annotations[key]`. """
        if isinstance(item, str):
            return self.annotations[item]
        else:
            raise IndexError('Trying to use Statistics.__getitem__ with non-string key.')


class MaxentModel(object):
    """ This class calculates and stores the parameters of a maximum entropy model for a multiple sequence alignment.
    """
    def __init__(self, stats):
        """ Initialize the maximum entropy model based on the given statistics structure. """
        self.alphabets = stats.alphabets
        self.annotations = stats.annotations
        self.reference = stats.reference
        self.stats = stats

        self.calculate_couplings()

    def calculate_couplings(self):
        """ Calculate the couplings for the statistics in `self.stats`.

        The couplings are stored in `self.couplings`, and they contain an entry for each pair of alignment position and
        character at that location, J_{ij}(a, b). The pairs of position and character, (i, a) and (j, b), are flattened
        into a linear index such that the values corresponding to all the possible characters at position 0 occur before
        those for position 1, which occur before those at position 2, etc. There is no location for gaps; the coupling
        between a gap and any other character is considered to be 0.

        Couplings between different characters at the same location, J_{ii}(a, b), are zero. The diagonal entries,
        J_{ii}(a, a) are non-zero; they correspond to double the "fields", J_{ii}(a, a) = 2*h_i(a) (see below).

        The couplings calculated by this model are used to provide a probabilistic model of sequence alignments. The
        probability of a sequence a_i is given by
          P = e^{-E(a)} / Z ,
        where E is the 'energy' score of the sequence, and Z is a normalization constant ensuring that the sum of P over
        all possible sequences equals 1. The energy is given by a quadratic form
          E(a) = -1/2*\sum_{i, j, a, b} J_{ij}(a, b) = -\sum_{i<j} J_{ij}(a, b) - \sum_i h_i(a).
        """
        freq1 = self.stats.freq1
        cmat = self.stats.cmat

        if np.size(cmat) == 0:
            self.couplings = np.asarray([])
            return

        self.couplings = -np.linalg.inv(cmat)

        bin_map = binary_index_map(self.stats)
        # diagnoal needs to be fixed
        for start, stop in bin_map:
            crt_slice = slice(start, stop)
            crt_Pi = freq1[crt_slice]
            # normalize such that fields are zero for gaps; make sure we don't divide by zero, though
            crt_Pgap = max(1e-10, 1 - np.sum(crt_Pi))
            self.couplings[crt_slice, crt_slice] = np.diag(2*np.log(crt_Pi / crt_Pgap))

    def score(self, seqs, internal_method='numpy'):
        """ Calculate the energy scores for the given sequences.

        See calculate_couplings for a definition of the energy.
        """
        input_type = None
        if hasattr(seqs, 'to_binary'):
            # is this a character alignment?
            if len(seqs) == 0:
                return np.asarray([])
            input_type = 'alignment'
            align = seqs
            if internal_method == 'numpy':
                bin_align = seqs.to_binary()
        elif hasattr(seqs, 'from_alignment'):
            # is this a binary alignment?
            if len(seqs) == 0:
                return np.asarray([])
            input_type = 'binary'
            bin_align = seqs
        elif hasattr(seqs, '__len__'):
            if len(seqs) == 0:
                return np.asarray([])
            if isinstance(seqs[0], str):
                input_type = 'list'
                seqs = np.asmatrix([list(_) for _ in seqs])
            else:
                input_type = 'matrix'

            from multicov.alignment import Alignment

            crt_idx = 0
            align = Alignment()
            for alpha, width in self.alphabets:
                crt_data = seqs[:, crt_idx:crt_idx+width]
                align.add(crt_data, alpha)

            if internal_method == 'numpy':
                bin_align = align.to_binary()
        else:
            raise ValueError('The input to MaxentModel.score should be a character or binary alignment, or a list of'
                             'sequences or matrix of characters.')

        bin_data = sparse.csr_matrix(bin_align.data)

        if internal_method == 'numpy' or input_type == 'binary':
            energies = np.asarray((-1/2) * (bin_data.multiply(bin_data.dot(self.couplings))).sum(axis=1)).squeeze()
        elif internal_method == 'numba1':
            # XXX should implement this at some point
            pass
#            self._calculate_energies_numba1(align.data)

        return energies
