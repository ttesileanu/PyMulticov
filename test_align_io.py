import unittest

import os


class TestLoadFasta(unittest.TestCase):
    def test_protein_unchanged_invalid(self):
        from multicov.alignment import Alignment
        from multicov.align_io import load_fasta
        from multicov.alphabet import protein_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln1.fasta'), protein_alphabet,
                           invalid_letter_policy='unchanged')
        expected = Alignment(['IVGGYTCQ', 'XVGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)
        expected.annotations['name'] = ['seq1', 'seq2', 'seq3']
        self.assertEqual(align, expected)

    def test_protein_keep_annot_ws(self):
        from multicov.alignment import Alignment
        from multicov.align_io import load_fasta
        from multicov.alphabet import protein_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln1.fasta'), protein_alphabet, strip_ws_in_annot=False)
        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)
        expected.annotations['name'] = ['seq1  ', ' seq2', 'seq3']
        self.assertEqual(align, expected)

    def test_dna_unchanged_invalid(self):
        from multicov.alignment import Alignment
        from multicov.align_io import load_fasta
        from multicov.alphabet import dna_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln2.fasta'), dna_alphabet,
                           invalid_letter_policy='unchanged')
        expected = Alignment(['GATTACA', 'ACCA--T', 'G.c-a-c'], dna_alphabet)
        expected.annotations['name'] = ['one', 'sequence', 'one line']
        self.assertEqual(align, expected)

    def test_replace_invalid_by_gap_protein(self):
        from multicov.alignment import Alignment
        from multicov.align_io import load_fasta
        from multicov.alphabet import protein_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln1.fasta'), protein_alphabet, invalid_letter_policy='gap')
        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)
        expected.annotations['name'] = ['seq1', 'seq2', 'seq3']
        self.assertEqual(align, expected)

    def test_replace_invalid_by_gap_dna(self):
        from multicov.alignment import Alignment
        from multicov.align_io import load_fasta
        from multicov.alphabet import dna_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='gap')
        expected = Alignment(['GATTACA', 'ACCA--T', 'G------'], dna_alphabet)
        expected.annotations['name'] = ['one', 'sequence', 'one line']
        self.assertEqual(align, expected)

    def test_replace_invalid_by_uppercase_then_gap(self):
        from multicov.alignment import Alignment
        from multicov.align_io import load_fasta
        from multicov.alphabet import dna_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='uppergap')
        expected = Alignment(['GATTACA', 'ACCA--T', 'G-C-A-C'], dna_alphabet)
        expected.annotations['name'] = ['one', 'sequence', 'one line']
        self.assertEqual(align, expected)

    def test_replace_invalid_by_uppercase_then_leave(self):
        from multicov.alignment import Alignment
        from multicov.align_io import load_fasta
        from multicov.alphabet import dna_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='upper')
        expected = Alignment(['GATTACA', 'ACCA--T', 'G.C-A-C'], dna_alphabet)
        expected.annotations['name'] = ['one', 'sequence', 'one line']
        self.assertEqual(align, expected)

    def test_mask_from_first_seq(self):
        from multicov.alignment import Alignment
        from multicov.align_io import load_fasta
        from multicov.alphabet import protein_alphabet
        from numpy import in1d
        align = load_fasta(os.path.join('test_data', 'test_aln1.fasta'), protein_alphabet,
                           invalid_letter_policy='unchanged',
                           mask_fct=lambda s: ~in1d(list(s), ['V', 'G']))
        expected = Alignment(['IYTCQ', 'XTEAQ', 'IKDT-'], alphabet=protein_alphabet)
        expected.annotations['name'] = ['seq1', 'seq2', 'seq3']
        self.assertEqual(align, expected)

    def test_mask_before_process(self):
        from multicov.alignment import ReferenceMapping
        from multicov.align_io import load_fasta
        from multicov.alphabet import protein_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln3.fasta'), protein_alphabet,
                           invalid_letter_policy='upper',
                           mask_fct=lambda s: [not _.islower() for _ in s])
        align0 = load_fasta(os.path.join('test_data', 'test_aln3.fasta'), protein_alphabet,
                            invalid_letter_policy='unchanged')
        mask = [not _.islower() for _ in align0[0, :]]
        expected = align0.truncate_columns(mask)
        expected.reference = ReferenceMapping(list(range(expected.data.shape[1])))
        self.assertEqual(align, expected)

    def test_mask_upper(self):
        from multicov.alignment import ReferenceMapping
        from multicov.align_io import load_fasta
        from multicov.alphabet import protein_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln3.fasta'), protein_alphabet,
                           invalid_letter_policy='upper',
                           mask_fct='upper')
        align0 = load_fasta(os.path.join('test_data', 'test_aln3.fasta'), protein_alphabet,
                            invalid_letter_policy='unchanged')
        mask = [not _.islower() and _ != '.' for _ in align0[0, :]]
        expected = align0.truncate_columns(mask)
        expected.reference = ReferenceMapping(list(range(expected.data.shape[1])))
        self.assertEqual(align, expected)

    def test_mask_upper_gap(self):
        from multicov.alignment import ReferenceMapping
        from multicov.align_io import load_fasta
        from multicov.alphabet import protein_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln3.fasta'), protein_alphabet,
                           invalid_letter_policy='upper',
                           mask_fct='uppernogap')
        align0 = load_fasta(os.path.join('test_data', 'test_aln3.fasta'), protein_alphabet,
                            invalid_letter_policy='unchanged')
        mask = [not _.islower() and _ != '.' and _ != '-' for _ in align0[0, :]]
        expected = align0.truncate_columns(mask)
        expected.reference = ReferenceMapping(list(range(expected.data.shape[1])))
        self.assertEqual(align, expected)


class TestHDFStoreIO(unittest.TestCase):
    def test_load(self):
        from multicov.align_io import from_hdf
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from pandas import HDFStore
        store = HDFStore(os.path.join('test_data', 'test_aln.h5'), 'r')
        align = from_hdf(store, 'align1')
        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        expected.annotations['seqw'] = [0.5, 1, 0.5]
        store.close()
        self.assertEqual(align, expected)

    def test_load_multi_alpha(self):
        from multicov.align_io import from_hdf
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        from pandas import HDFStore
        store = HDFStore(os.path.join('test_data', 'test_aln.h5'), 'r')
        align = from_hdf(store, 'align2')
        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        expected2 = Alignment(['AGCT', '-G-G', 'TA-T'], dna_alphabet)
        expected.add(expected2)
        store.close()
        self.assertEqual(align, expected)

    def test_roundtrip(self):
        from multicov.align_io import from_hdf, to_hdf
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        from pandas import HDFStore
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align2 = Alignment(['AGCT', '-G-G', 'TA-T'], dna_alphabet)
        align.add(align2)

        store = HDFStore(os.path.join('test_data', 'tmp.h5'), 'w')
        to_hdf(align, store, 'test_align')
        store.close()

        store = HDFStore(os.path.join('test_data', 'tmp.h5'), 'r')
        reloaded = from_hdf(store, 'test_align')
        store.close()

        os.remove(os.path.join('test_data', 'tmp.h5'))

        self.assertEqual(align, reloaded)
