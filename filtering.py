from alphabet import protein_alphabet, dna_alphabet, rna_alphabet
from alignment import Alignment

import numpy as np

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo


def search(align, seq, move_to_top=False, substitution_matrix=None, gap_open=-10, gap_extend=-0.5):
    """ Searches for the best match to the given sequence in the given alignment, and returns its index.

    If `move_to_top == True`, the sequence is swapped with the first alignment sequence. The return value remains
    the position of the sequence before it got moved.

    The default substitution matrix is BLOSUM50 for proteins and NUC.4.4 for DNA. A version of NUC.4.4 with T replaced
    by U is used by default for RNA. For any other alphabets, a substitution matrix needs to be specified (that is a
    dict from pairs of letters to scores).

    The function currently does not work on multi-alphabet alignments.
    """
    if len(align.alphabets) == 0 or len(align) == 0:
        raise ValueError('search on empty alignment.')
    if len(align.alphabets) > 1:
        raise ValueError('search not implemented on multi-alphabet alignments.')
    if len(seq) == 0:
        raise ValueError('search with empty sequence.')

    if substitution_matrix is None:
        if align.alphabets[0][0] == protein_alphabet:
            substitution_matrix = MatrixInfo.blosum50
        elif align.alphabets[0][0] == dna_alphabet:
            substitution_matrix = {
                ('A', 'A'): 5,
                ('C', 'A'): -4, ('C', 'C'): 5,
                ('G', 'A'): -4, ('G', 'C'): -4, ('G', 'G'): 5,
                ('T', 'A'): -4, ('T', 'C'): -4, ('T', 'G'): -4, ('T', 'T'): 5
            }
        elif align.alphabets[0][0] == rna_alphabet:
            substitution_matrix = {
                ('A', 'A'): 5,
                ('C', 'A'): -4, ('C', 'C'): 5,
                ('G', 'A'): -4, ('G', 'C'): -4, ('G', 'G'): 5,
                ('U', 'A'): -4, ('U', 'C'): -4, ('U', 'G'): -4, ('U', 'U'): 5
            }
        else:
            raise ValueError('explicit substitution_matrix missing on alignment with alphabet that is not protein, dna,'
                             ' or rna')

    alphabet = align.alphabets[0][0]

    # make sure the sequence is a string
    seq = ''.join(seq)
    # turn alignment into sequence of strings, stripping gaps
    if not alphabet.has_gap:
        raise ValueError('search requires the alignment alphabet to have a gap.')
    gap_char = alphabet[0]
    align_seqs = [''.join(x for x in _ if x != gap_char) for _ in np.asarray(align[:, :])]

    scores = []
    for i, align_seq in enumerate(align_seqs):
        scores.append(pairwise2.align.globalds(seq, align_seq, substitution_matrix,
                                               gap_open, gap_extend, one_alignment_only=1, score_only=1))

    # find the highest scoring sequence
    best_id = np.argmax(scores)

    # swap to first position?
    if move_to_top:
        align.swap(0, best_id)

    return best_id


def filter_rows(align, max_gaps=0.5):
    """ Return a new alignment where rows that have too many gaps are removed (a fraction larger than max_gaps). """
    if len(align) == 0:
        return Alignment()
    gap_structure = np.zeros(np.shape(align.data), dtype=bool)
    start_idx = 0
    for alpha, width in align.alphabets:
        if alpha.has_gap:
            gap_ch = alpha[0]
            gap_structure[:, start_idx:start_idx+width] = (align.data[:, start_idx:start_idx+width] == gap_ch)

        start_idx += width

    gap_fractions = np.mean(gap_structure, axis=1)
    mask = (gap_fractions <= max_gaps)

    return align[mask]
