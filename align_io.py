""" Define input-output routines for alignments. """

from multicov.alignment import Alignment

import numpy as np


def load_fasta(fname, alphabet, strip_ws_in_annot=True, invalid_letter_policy='uppergap'):
    """ Load a FASTA file into an alignment structure.

    If `strip_ws_in_annot == True`, whitespace surrounding sequence annotations is removed.
    The sequences are processed based on the `invalid_letter_policy` option. `invalid_letter_policy` can be:
        * 'unchanged' or `None`, in which case invalid letters are kept in the alignment
        * 'gap', in which all invalid letters are replaced by gaps; this fails if the alphabet is gap-less
        * 'upper', in which lower-case letters are converted to uppercase, without further processing
        * 'uppergap', in which lower-case letters are converted to uppercase, after which the 'gap' policy is applied
    """
    seqs = []
    annots = []
    if invalid_letter_policy is None or invalid_letter_policy == 'unchanged':

        def process(s):
            return s

    elif invalid_letter_policy == 'gap':
        if not alphabet.has_gap:
            raise ValueError("Can't use 'gap' or 'uppergap' policies on gap-less alphabet.")

        def process(s):
            return _invalid_to_gap(s, alphabet)

    elif invalid_letter_policy == 'upper':
        process = str.upper
    elif invalid_letter_policy == 'uppergap':
        if not alphabet.has_gap:
            raise ValueError("Can't use 'gap' or 'uppergap' policies on gap-less alphabet.")

        def process(s):
            return _invalid_to_gap(s.upper(), alphabet)

    else:
        raise ValueError('Unknown invalid letter policy {}'.format(str(invalid_letter_policy)))
    with open(fname, 'r') as f:
        crt_seq = ''
        width = None
        just_started = True
        for line in f:
            line = line.strip('\r\n').lstrip()
            if len(line) == 0 or line[0] == ';':
                # skip comments or empty lines
                continue
            if line[0] == '>':
                # new sequence
                if len(crt_seq) > 0:
                    if width is None:
                        width = len(crt_seq)
                    elif len(crt_seq) != width:
                        raise ValueError("FASTA sequences don't all have the same length.")
                    seqs.append(process(crt_seq))
                elif not just_started:
                    raise ValueError('There should be only one annotation per sequence in FASTA file.')

                crt_seq = ''
                crt_annot = line[1:]
                if strip_ws_in_annot:
                    crt_annot = crt_annot.strip()
                annots.append(crt_annot)
                just_started = False
            else:
                crt_seq = crt_seq + line.strip()

        if not just_started:
            if len(crt_seq) == 0:
                raise ValueError("Last FASTA annotation doesn't have a matching sequence.")
            if width is not None and width != len(crt_seq):
                raise ValueError("FASTA sequences don't all have the same length.")
            seqs.append(process(crt_seq))
        else:
            if len(crt_seq) > 0:
                raise ValueError("Sequence without annotation in FASTA file.")

    if len(seqs) != len(annots):
        raise Exception("This shouldn't happen: len(seqs) != len(annots).")

    result = Alignment(seqs, alphabet)
    result.annotations['name'] = annots

    return result


def _invalid_to_gap(s, alphabet):
    """ Convert letters that are not in the alphabet to gaps.

    This does not check that the alphabet has a gap, but assumes that the first letter is the gap.
    """
    letters = alphabet.letters()
    s = np.asarray(list(s))
    mask = np.in1d(s, letters, invert=True)
    s[mask] = letters[0]
    return ''.join(s)
