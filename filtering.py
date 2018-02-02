from .alphabet import protein_alphabet, dna_alphabet, rna_alphabet
from .alignment import Alignment, ReferenceMapping

import numpy as np

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo


def _get_substitution_matrix(alphabet):
    """ Return a tuple with default parameters `(substitution_matrix, gap_open, gap_extend) for the given alphabet. """
    if alphabet == protein_alphabet:
        return MatrixInfo.blosum50, -8, -8
    elif alphabet == dna_alphabet:
        return ({
            ('A', 'A'): 5,
            ('C', 'A'): -4, ('C', 'C'): 5,
            ('G', 'A'): -4, ('G', 'C'): -4, ('G', 'G'): 5,
            ('T', 'A'): -4, ('T', 'C'): -4, ('T', 'G'): -4, ('T', 'T'): 5
        }, -2, -0.5)
    elif alphabet == rna_alphabet:
        return ({
            ('A', 'A'): 5,
            ('C', 'A'): -4, ('C', 'C'): 5,
            ('G', 'A'): -4, ('G', 'C'): -4, ('G', 'G'): 5,
            ('U', 'A'): -4, ('U', 'C'): -4, ('U', 'G'): -4, ('U', 'U'): 5
        }, -2, -0.5)
    else:
        raise ValueError('explicit substitution_matrix missing on alignment with alphabet that is not protein, dna,'
                         ' or rna')


def search(align, seq, move_to_top=False, substitution_matrix=None, gap_open=None, gap_extend=None):
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

    alphabet = align.alphabets[0][0]
    if substitution_matrix is None:
        substitution_matrix, default_gap_open, default_gap_extend = _get_substitution_matrix(alphabet)
        if gap_open is None:
            gap_open = default_gap_open
        if gap_extend is None:
            gap_extend = default_gap_extend

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
                                               gap_open, gap_extend, one_alignment_only=True, score_only=True,
                                               penalize_end_gaps=False))

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
    gap_structure = align.get_gap_structure()

    gap_fractions = np.mean(gap_structure, axis=1)
    mask = (gap_fractions <= max_gaps)

    return align[mask]


def align_to_sequence(align, seq, ref_idx_names=None, truncate=False, force_idx=None):
    """ Set the reference mapping for the alignment according to the given sequence.

    By default, the function searches for the best match to `seq` within the alignment, and uses this match to infer a
    mapping between alignment columns and positions in `seq`. Columns that do not match any position in `seq` are marked
    with `None`. If `truncate` is `True`, the columns that do not have a match in `seq` are removed.

    If `force_idx` is set, the search is not done, and only the sequence at that position is used for the matching.

    By default, the positions in `seq` are numbered consecutively, starting from 0. If `ref_idx_names` is given,
    position `i` in `seq` will have name `ref_idx_names[i]`, and these names will be used in the reference mapping that
    will be attached to the alignment.

    Currently this only works for single alphabet alignments, and the alphabet needs to be protein, DNA, or RNA.

    The position of the matched sequence in the alignment, and the accuracy of the match, are returned in a dictionary.
    """
    if len(align) == 0:
        # nothing to do
        return align

    if len(align.alphabets) > 1:
        raise ValueError('align_to_sequence not implemented on multi-alphabet alignments.')

    alphabet = align.alphabets[0][0]
    substitution_matrix, gap_open, gap_extend = _get_substitution_matrix(alphabet)
    if force_idx is None:
        # find the best matching sequence
        force_idx = search(align, seq, substitution_matrix=substitution_matrix,
                           gap_open=gap_open, gap_extend=gap_extend)

    # find the best match
    gap_ch = alphabet[0]
    # need the alignment sequence without gaps
    align_seq = np.asarray(align.data[force_idx])[0]
    align_gap_mask = (align_seq == gap_ch)
    align_seq_no_gaps = align_seq[~align_gap_mask]
    align_seq_no_gaps_as_str = ''.join(align_seq_no_gaps)

    seq = ''.join(seq)
    p_al = pairwise2.align.globalds(seq, align_seq_no_gaps_as_str, substitution_matrix, gap_open, gap_extend,
                                    penalize_end_gaps=False)

    # this will be the mapping from indices in alignment to indices in `seq`
    ref_idxs = np.asarray([None for _ in range(len(align_seq))])
    # the ungapped positions in p_al[0][0] correspond to positions in the reference sequence
    # let's label them
    p_al_ref_idxs = np.asarray([None for _ in range(len(p_al[0][0]))])
    p_al_ref_idxs[np.asarray(list(p_al[0][0])) != gap_ch] = list(range(len(seq)))
    # now the ungapped positions in p_al[0][1] correspond to ungapped positions in the alignment sequence
    ref_idxs[~align_gap_mask] = p_al_ref_idxs[np.asarray(list(p_al[0][1])) != gap_ch]

    # calculate some details
    details = {'align_accuracy': np.mean(
                    [a == b for a, b in zip(p_al[0][0], p_al[0][1]) if a != gap_ch and b != gap_ch]),
               'idx': force_idx}

    # do we want to truncate the alignment?
    if truncate:
        # noinspection PyComparisonWithNone
        truncate_mask = (ref_idxs != None)
        align.truncate_columns(truncate_mask, in_place=True)
        ref_idxs = ref_idxs[truncate_mask]

    if ref_idx_names is not None:
        ref_seq = [ref_idx_names[_] if _ is not None else None for _ in ref_idxs]
    else:
        ref_seq = ref_idxs

    align.reference = ReferenceMapping(list(ref_seq))

    return details
