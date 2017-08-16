""" Define a class that calculates statistics for multiple sequence alignments. """

import numpy as np

from scipy import sparse


# noinspection SpellCheckingInspection
class Statistics(object):
    def __init__(self, alignment, precompute=()):
        """ Initialize the statistics structure, attaching it to the given alignment. This can be either a character
        `Alignment` or a `BinaryAlignment`.

        The evaluation of the actual statistics is delayed until they are requested. This means that their values
        might be different if the alignment changed in the meantime. Once one of the statistics is calculated, it is
        never automatically recalculated, so changes to the alignment that are made after the first access to the
        statistics will not be reflected in the statistics. See `update`. Note that unexpected results can occur if, for
        example, `freq1` is updated first, and then, before `cmat` is called, the alignment changes.

        Set `precompute` to a tuple containing one or several of 'freq1', 'freq2', 'cmat' to compute these values at
        the time of construction. """
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

        return seqw * bin_align.data / n_eff

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

        return (bindata.T * seqw_mat * bindata).todense() / n_eff

    def _calculate_cmat(self):
        if self._is_empty():
            return np.asarray([])

        freq1 = self._freq1 if self._freq1 is not None else self._calculate_freq1()
        freq2 = self._freq2 if self._freq2 is not None else self._calculate_freq2()
        return freq2 - np.outer(freq1, freq1)

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
