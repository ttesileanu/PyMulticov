""" Define input-output routines for alignments. """

from multicov.alignment import Alignment, ReferenceMapping

import numpy as np


def load_fasta(fname, alphabet, strip_ws_in_annot=True, invalid_letter_policy='uppergap', mask_fct=None):
    """ Load a FASTA file into an alignment structure.

    If `strip_ws_in_annot == True`, whitespace surrounding sequence annotations is removed.
    The sequences are processed based on the `invalid_letter_policy` option. `invalid_letter_policy` can be:
        * 'unchanged' or `None`, in which case invalid letters are kept in the alignment
        * 'gap', in which all invalid letters are replaced by gaps; this fails if the alphabet is gap-less
        * 'upper', in which lower-case letters are converted to uppercase, without further processing
        * 'uppergap', in which lower-case letters are converted to uppercase, after which the 'gap' policy is applied
    If `mask_fct` is provided, it is called with the first sequence, `mask_fct(s0)`, and should return a Numpy mask
    selecting which columns of the alignment to keep. If `mask_fct` is the string 'upper', then the mask is such that
    columns with lowercase letters or the character '.' in the first sequence are rejected. The string 'uppergap' also
    rejects gaps, '-'.
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

    if mask_fct == 'upper':
        mask_fct = lambda s: [not _.islower() and _ != '.' for _ in s]
    elif mask_fct == 'uppernogap':
        mask_fct = lambda s: [not _.islower() and _ != '.' and _ != '-' for _ in s]

    with open(fname, 'r') as f:
        crt_seq = ''
        width = None
        just_started = True
        mask = None
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
                    if mask_fct is not None and len(seqs) == 0:
                        mask = mask_fct(crt_seq)
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
            if mask_fct is not None and len(seqs) == 0:
                mask = mask_fct(crt_seq)
            seqs.append(process(crt_seq))
        else:
            if len(crt_seq) > 0:
                raise ValueError("Sequence without annotation in FASTA file.")

    if len(seqs) != len(annots):
        raise Exception("This shouldn't happen: len(seqs) != len(annots).")

    result = Alignment(seqs, alphabet)
    result.annotations['name'] = annots

    if mask is not None and len(seqs) > 0:
        result.truncate_columns(mask, in_place=True)
        result.reference = ReferenceMapping(list(range(result.data.shape[1])))

    return result


def _name_to_alpha(name):
    from multicov.alphabet import protein_alphabet, dna_alphabet, rna_alphabet, NumericAlphabet
    if name == 'protein':
        return protein_alphabet
    elif name == 'dna':
        return dna_alphabet
    elif name == 'rna':
        return rna_alphabet
    elif hasattr(name, 'startswith') and name.startswith('numeric'):
        try:
            params = [int(_) for _ in name[8:-1].split(':')]
            # XXX the naming convention for numeric alphabets doesn't indicate whether there are gaps
            return NumericAlphabet(*params)
        except:
            raise Exception('Ill-formed numeric alphabet ' + str(name) + '.')
    else:
        raise Exception('Unrecognized alphabet ' + str(name) + '.')


def to_hdf(alignment, hdf_store, key):
    """ Save the alignment in a group inside a Pandas `HDFStore`. """
    from pandas import Series, DataFrame
    alphabet_names = Series([_[0].name for _ in alignment.alphabets])
    alphabet_widths = Series([_[1] for _ in alignment.alphabets])
    data_as_list = Series([''.join(_.getA1()) for _ in alignment.data])

    max_ref_len = max(len(_) for _ in alignment.reference.seqs)
    ref_extended = [np.hstack((np.asarray(_), np.tile(np.nan, max_ref_len - len(_)))) for _ in alignment.reference.seqs]
    ref_as_df = DataFrame(ref_extended)

    hdf_store.put(key + '/alphabet_names', alphabet_names)
    hdf_store.put(key + '/alphabet_widths', alphabet_widths)
    hdf_store.put(key + '/data', data_as_list)
    hdf_store.put(key + '/ref', ref_as_df)
    hdf_store.put(key + '/annotations', alignment.annotations)


def from_hdf(hdf_store, key):
    """ Load an alignment from a Pandas `HDFStore`. """
    alignment = Alignment()

    data_as_list = hdf_store.get(key + '/data').as_matrix()
    alignment.data = np.asmatrix([list(_) for _ in data_as_list])

    ref_as_df = hdf_store.get(key + '/ref').as_matrix()
    ref_no_nan = [_[np.isfinite(_)] for _ in ref_as_df]
    alignment.reference = ReferenceMapping(ref_no_nan)

    alphabet_names = hdf_store.get(key + '/alphabet_names').as_matrix()
    alphabet_widths = hdf_store.get(key + '/alphabet_widths').as_matrix()
    alignment.alphabets = [(_name_to_alpha(name), width) for name, width in zip(alphabet_names, alphabet_widths)]

    alignment.annotations = hdf_store.get(key + '/annotations')

    # XXX should validate alignment

    return alignment


def _invalid_to_gap(s, alphabet):
    """ Convert letters that are not in the alphabet to gaps.

    This does not check that the alphabet has a gap, but assumes that the first letter is the gap.
    """
    letters = alphabet.letters()
    s = np.asarray(list(s))
    mask = np.in1d(s, letters, invert=True)
    s[mask] = letters[0]
    return ''.join(s)
