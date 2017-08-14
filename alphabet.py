""" This module defines alignment alphabets. """

import numpy as np
import copy


class Alphabet(object):
    """ Base class for alignment alphabets. This is similar to the Biopython class, but seemed to have sufficiently
    different goals as to justify creating it as a separate class.

    The alphabet is viewed as a mapping between 'letters' (which can be arbitrary hashable objects, but typically will
    be either characters or numbers) and integers from 0 to the number of characters in the alphabet. It may or may not
    contain one special letter, called the 'gap'.

    Public data members:
        - name: string description of the alphabet
        - has_gap: whether the alphabet contains a gap or not
    """

    def __init__(self, *args, name=None, letters=None, has_gap=None):
        if len(args) > 1:
            raise TypeError("Alphabet takes at most one positional argument -- another Alphabet(-like) object.")
        if len(args) == 1:
            if name is not None or letters is not None or has_gap is not None:
                raise TypeError("Alphabet copy constructor does not accept keyword arguments.")
            self.name = args[0].name
            self._letters = copy.copy(args[0].letters())
            self.has_gap = args[0].has_gap
            self._generate_map()
        else:
            self.name = name if name is not None else 'none'
            self._letters = list(letters) if letters is not None else []
            self.has_gap = has_gap if has_gap is not None else False
            self._generate_map()

    def _generate_map(self):
        self._map = {}
        for i, ch in enumerate(self._letters):
            self._map[ch] = i

    def __repr__(self):
        return str(self.name)

    def __getitem__(self, key):
        return self._letters[key]

    def __eq__(self, other):
        """ Name, letters, and whether alphabet has a gap or not must all match for equality. """
        if self is other:
            return True
        if not isinstance(other, Alphabet):
            return False

        if not np.array_equal(list(self._letters), list(other.letters())):
            return False
        if self.has_gap != other.has_gap:
            return False

        return self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return self.size()

    def size(self, no_gap=False):
        """ Returns the number of letters in the alphabet. If no_gap == True, ignores the gap, if any. """
        if no_gap and self.has_gap:
            return len(self._letters) - 1
        else:
            return len(self._letters)

    def letters(self, no_gap=False):
        """ Get a sequence of letters in the alphabet, omitting the gap (if any) if no_gap == True. """
        if no_gap and self.has_gap:
            return self._letters[1:]
        else:
            return self._letters

    def to_int(self, seqs):
        """ Convert a letter, a sequence of letters, or a matrix of letters to a corresponding integer, sequence of
        integers, or matrix of integers. This also works for lists of sequences of unequal lengths. """
        # can't really do any conversion if we don't have any letters
        if len(self._letters) == 0:
            raise KeyError("to_int called on empty alphabet.")

        if np.ndim(seqs) == 0:
            return self._map[seqs]

        was_1d = False
        if np.ndim(seqs) == 1:
            if len(seqs) == 0:
                return []
            # this could be either a single sequence, or a list of sequences with different lengths
            if np.ndim(seqs[0]) == 0:
                # if it is a single sequence, just turn it into a matrix and treat it like a matrix
                seqs = [seqs]
                was_1d = True
            else:
                # otherwise transform each sequence separately
                return [self.to_int(_) for _ in seqs]

        if np.ndim(seqs) > 2:
            raise TypeError("Input to to_int should be a scalar, a vector, or a matrix.")

        seqs = np.asarray(seqs)
        result = np.empty(seqs.shape, dtype=np.int16)
        mask_valid = np.zeros(seqs.shape, dtype=bool)
        for ch in self._map:
            result[seqs == ch] = self._map[ch]
            mask_valid[seqs == ch] = True

        if not np.all(mask_valid):
            raise KeyError("Invalid letters for alphabet {} in to_int.".format(self.name))

        if was_1d:
            result = result[0]

        return result

    def from_int(self, nseqs):
        """ Convert an integer, a sequence of integers, or a matrix of integers to a corresponding letter, sequence of
        letters, or matrix of letters. This also works for lists of sequences of unequal lengths. """
        # can't really do any conversion if we don't have any letters
        if len(self._letters) == 0:
            raise KeyError("from_int called on empty alphabet.")

        if np.ndim(nseqs) == 0:
            return self._letters[nseqs]

        was_1d = False
        if np.ndim(nseqs) == 1:
            if len(nseqs) == 0:
                return []
            # this could be either a single sequence, or a list of sequences with different lengths
            if np.ndim(nseqs[0]) == 0:
                # if it is a single sequence, just turn it into a matrix and treat it like a matrix
                nseqs = [nseqs]
                was_1d = True
            else:
                # otherwise transform each sequence separately
                return [self.from_int(_) for _ in nseqs]

        if np.ndim(nseqs) > 2:
            raise TypeError("Input to from_int should be a scalar, a vector, or a matrix.")

        nseqs = np.asarray(nseqs)

        if not np.all(np.isfinite(nseqs)) or np.any(nseqs >= self.size()) or np.any(nseqs < 0):
            raise IndexError("Out of range numbers for alphabet {} in from_int.".format(self.name))

        # noinspection PyTypeChecker
        result = self._letters[nseqs]

        if was_1d:
            result = result[0]

        return result


class DNAAlphabet(Alphabet):
    """ DNA alphabet. """

    def __init__(self):
        super(DNAAlphabet, self).__init__()
        self.name = 'dna'
        self._letters = np.asarray(list('-ACGT'))
        self.has_gap = True
        self._generate_map()


class RNAAlphabet(Alphabet):
    """ RNA alphabet. """

    def __init__(self):
        super(RNAAlphabet, self).__init__()
        self.name = 'rna'
        self._letters = np.asarray(list('-ACGU'))
        self.has_gap = True
        self._generate_map()


class ProteinAlphabet(Alphabet):
    """ Protein alphabet. """

    def __init__(self):
        super(ProteinAlphabet, self).__init__()
        self.name = 'protein'
        self._letters = np.asarray(list("-ACDEFGHIKLMNPQRSTVWY"))
        self.has_gap = True
        self._generate_map()


class NumericAlphabet(Alphabet):
    """ Numeric alphabet in which all the characters are integers. """

    def __init__(self, *args, has_gap=None):
        """ Create an alphabet with numbers in the range [base:base+size]. """
        super(NumericAlphabet, self).__init__()
        if len(args) > 2 or len(args) == 0:
            raise TypeError("NumericAlphabet takes one or two positional arguments.")
        if len(args) == 1:
            base = 0
            high = args[0]
        else:
            base, high = args
        self.name = 'numeric[%d:%d]' % (base, high)
        self._letters = np.arange(base, high)
        self.has_gap = has_gap if has_gap is not None else base == 0
        self._generate_map()


# some commonly-used alphabets
dna_alphabet = DNAAlphabet()
rna_alphabet = RNAAlphabet()
protein_alphabet = ProteinAlphabet()
