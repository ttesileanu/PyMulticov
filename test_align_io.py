import unittest

import os


class TestLoadFasta(unittest.TestCase):
    def test_protein_unchanged_invalid(self):
        from alignment import Alignment
        from align_io import load_fasta
        from alphabet import protein_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln1.fasta'), protein_alphabet,
                           invalid_letter_policy='unchanged')
        expected = Alignment(['IVGGYTCQ', 'XVGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)
        expected.annotations['name'] = ['seq1', 'seq2', 'seq3']
        self.assertEqual(align, expected)

    def test_protein_keep_annot_ws(self):
        from alignment import Alignment
        from align_io import load_fasta
        from alphabet import protein_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln1.fasta'), protein_alphabet, strip_ws_in_annot=False)
        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)
        expected.annotations['name'] = ['seq1  ', ' seq2', 'seq3']
        self.assertEqual(align, expected)

    def test_dna_unchanged_invalid(self):
        from alignment import Alignment
        from align_io import load_fasta
        from alphabet import dna_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln2.fasta'), dna_alphabet,
                           invalid_letter_policy='unchanged')
        expected = Alignment(['GATTACA', 'ACCA--T', 'G.c-a-c'], dna_alphabet)
        expected.annotations['name'] = ['one', 'sequence', 'one line']
        self.assertEqual(align, expected)

    def test_replace_invalid_by_gap_protein(self):
        from alignment import Alignment
        from align_io import load_fasta
        from alphabet import protein_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln1.fasta'), protein_alphabet, invalid_letter_policy='gap')
        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)
        expected.annotations['name'] = ['seq1', 'seq2', 'seq3']
        self.assertEqual(align, expected)

    def test_replace_invalid_by_gap_dna(self):
        from alignment import Alignment
        from align_io import load_fasta
        from alphabet import dna_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='gap')
        expected = Alignment(['GATTACA', 'ACCA--T', 'G------'], dna_alphabet)
        expected.annotations['name'] = ['one', 'sequence', 'one line']
        self.assertEqual(align, expected)

    def test_replace_invalid_by_uppercase_then_gap(self):
        from alignment import Alignment
        from align_io import load_fasta
        from alphabet import dna_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='uppergap')
        expected = Alignment(['GATTACA', 'ACCA--T', 'G-C-A-C'], dna_alphabet)
        expected.annotations['name'] = ['one', 'sequence', 'one line']
        self.assertEqual(align, expected)

    def test_replace_invalid_by_uppercase_then_leave(self):
        from alignment import Alignment
        from align_io import load_fasta
        from alphabet import dna_alphabet
        align = load_fasta(os.path.join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='upper')
        expected = Alignment(['GATTACA', 'ACCA--T', 'G.C-A-C'], dna_alphabet)
        expected.annotations['name'] = ['one', 'sequence', 'one line']
        self.assertEqual(align, expected)
