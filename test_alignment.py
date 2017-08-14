import unittest

import numpy as np


class TestConstructor(unittest.TestCase):
    def test_make_empty(self):
        from multicov.alignment import Alignment
        align = Alignment()
        self.assertTrue(hasattr(align, 'alphabets'))
        self.assertTrue(hasattr(align, 'data'))
        self.assertTrue(hasattr(align, 'reference'))
        self.assertTrue(hasattr(align, 'annotations'))
        self.assertEqual(np.size(align.data), 0)
        self.assertEqual(len(align.alphabets), 0)

    def test_make_from_list_of_strings(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        self.assertEqual(len(align.alphabets), 1)
        self.assertEqual(align.alphabets[0][0], protein_alphabet)
        self.assertEqual(align.alphabets[0][1], 4)
        self.assertSequenceEqual(np.shape(align.data), (3, 4))
        self.assertSequenceEqual(list(np.asarray(align.data)[0]), ['A', 'C', 'C', 'W'])
        self.assertSequenceEqual(list(np.asarray(align.data)[1]), ['F', 'F', 'H', '-'])
        self.assertSequenceEqual(list(np.asarray(align.data)[2]), ['-', '-', '-', 'D'])

    def test_make_from_matrix(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet
        align1 = Alignment(['AGU-', 'C-C-', 'AGAU', '--U-'], rna_alphabet)
        align2 = Alignment([['A', 'G', 'U', '-'],
                            ['C', '-', 'C', '-'],
                            ['A', 'G', 'A', 'U'],
                            ['-', '-', 'U', '-']], alphabet=rna_alphabet)
        self.assertEqual(len(align1.alphabets), 1)
        self.assertEqual(align1.alphabets[0][0], rna_alphabet)
        self.assertEqual(align1.alphabets[0][1], 4)
        self.assertTrue(np.array_equal(align1.data, align2.data))
        self.assertEqual(align1.alphabets, align2.alphabets)

    def test_copy(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align1 = Alignment(['AA', 'T-', 'CG', 'TT'], dna_alphabet)
        align2 = Alignment(align1)
        self.assertIsNot(align1, align2)
        self.assertEqual(align1, align2)

    def test_changing_copy_leaves_original_unchanged(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align1 = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        align2 = Alignment(align1)
        align2.annotations['seqw'] = [1, 2, 3]
        self.assertNotEqual(align1, align2)
        self.assertTrue(np.array_equal(align2.annotations['seqw'], [1, 2, 3]))

    def test_deep_copy_data(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import NumericAlphabet
        align1 = Alignment([[3, 1, 0, 4], [2, 2, 1, 4]], NumericAlphabet(5))
        align2 = Alignment(align1)
        align2.data[0, 2] = 3
        self.assertNotEqual(align1, align2)
        self.assertEqual(align2.data[0, 2], 3)

    def test_trivial_sequence_weights(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        self.assertIn('seqw', align.annotations.columns)
        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(align.annotations['seqw'], 1))

    def test_set_default_reference(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        self.assertEqual(align.reference, ReferenceMapping(list(range(4))))


class TestLength(unittest.TestCase):
    def test_empty_len(self):
        from multicov.alignment import Alignment
        self.assertEqual(len(Alignment()), 0)

    def test_nonempty(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        self.assertEqual(len(align), 3)


class TestGetItem(unittest.TestCase):
    def test_get_letter(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        self.assertEqual(align[1, 3], 'A')
        self.assertEqual(align[2, 2], '-')
        self.assertEqual(align[0, 0], 'A')
        self.assertEqual(align[0, 3], 'C')
        self.assertEqual(align[1, 1], 'A')
        self.assertEqual(align[0, 5], 'T')

    def test_get_row(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet
        align = Alignment(['UAG', 'GGU', 'TAU', '--G'], rna_alphabet)
        subalign = align[1]
        subalign_expected = Alignment(['GGU'], rna_alphabet)
        self.assertEqual(subalign, subalign_expected)

    def test_select_range_of_rows(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        subalign = Alignment(['FFH-', '---D'], protein_alphabet)
        self.assertEqual(align[1:], subalign)

    def test_select_rows_keeps_data_as_matrix(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        subalign = align[1:]
        self.assertEqual(np.ndim(subalign.data), 2)
        self.assertEqual(np.ndim(subalign.data[0]), 2)

    def test_get_row_with_annotations(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet
        align = Alignment(['UAG', 'GGU', 'TAU', '--G'], rna_alphabet)
        align.annotations['seqw'] = [0.5, 1.2, 0.7, 0.8]
        subalign = align[1]
        subalign_expected = Alignment(['GGU'], rna_alphabet)
        subalign_expected.annotations['seqw'] = [1.2]
        self.assertEqual(subalign, subalign_expected)

    def test_select_range_with_annotations(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        align.annotations['seqw'] = [0.7, 1.5, 2.5]
        align.annotations['other'] = [-1, 0, 1]
        subalign = Alignment(['FFH-', '---D'], protein_alphabet)
        subalign.annotations['seqw'] = [1.5, 2.5]
        subalign.annotations['other'] = [0, 1]
        self.assertEqual(align[1:], subalign)

    def test_select_discontiguous_set_of_rows(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        subalign = align[[0, 2]]
        expected = Alignment(['IVGGYTCQ', 'IGG-KDT-'], protein_alphabet)
        self.assertEqual(subalign, expected)

    def test_get_letter_matrix(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        subdata = align[1:3, 2:5]
        expected = np.asarray([['T', 'A', 'C'], ['-', '-', 'G']])
        self.assertTrue(np.array_equal(subdata, expected))

    def test_get_letter_matrix_for_single_row(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        subdata = align[2, :]
        expected = np.asarray(['A', 'A', '-', '-', 'G', 'G'])
        self.assertTrue(np.array_equal(subdata, expected))

    def test_selecting_empty_removes_annots_refmap_alpha(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        self.assertEqual(align[[]], Alignment())

    def test_raise_on_empty(self):
        from multicov.alignment import Alignment
        with self.assertRaises(IndexError):
            _ = Alignment()[0]

    def test_raise_on_out_of_range(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        with self.assertRaises(IndexError):
            _ = align[3]
        with self.assertRaises(IndexError):
            _ = align[2, 20]

    def test_get_str_gets_annotation(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.annotations['name'] = ['seq1', 'seq2', 'seq3']
        self.assertAlmostEqual(align['seqw'][2], 1)
        self.assertEqual(align['name'][1], 'seq2')

    def test_select_bool(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-', '-VGGTEAQ'], protein_alphabet)
        subalign = align[[True, False, True, False]]
        expected = Alignment(['IVGGYTCQ', 'IGG-KDT-'], protein_alphabet)
        self.assertEqual(subalign, expected)


class TestTruncateColumns(unittest.TestCase):
    def test_raise_on_empty(self):
        from multicov.alignment import Alignment
        with self.assertRaises(IndexError):
            _ = Alignment().truncate_columns([0])

    def test_protein_example(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        subalign = align.truncate_columns([0, 3, 5, 7])
        expected = Alignment(['IGTQ', '-GEQ', 'I-D-'], protein_alphabet)
        expected.reference = ReferenceMapping([0, 3, 5, 7])
        self.assertEqual(subalign, expected)

    def test_with_mask(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        subalign = align.truncate_columns([True, False, True, True, False, False, False, True])
        expected = Alignment(['IGGQ', '-GGQ', 'IG--'], protein_alphabet)
        expected.reference = ReferenceMapping([0, 2, 3, 7])
        self.assertEqual(subalign, expected)

    def test_in_place(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.truncate_columns([0, 3, 5, 7], in_place=True)
        expected = Alignment(['IGTQ', '-GEQ', 'I-D-'], protein_alphabet)
        expected.reference = ReferenceMapping([0, 3, 5, 7])
        self.assertEqual(align, expected)

    def test_multi_alpha_select_single_alpha_example(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import rna_alphabet, dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align.add(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        subalign = align.truncate_columns([1, 3, 4, 7, 9])
        expected = Alignment(['TCA', 'AAC', 'A-G'], dna_alphabet).add(['G-', '--', 'GU'], rna_alphabet)
        expected.reference = ReferenceMapping([[1, 3, 4], [1, 3]])
        self.assertEqual(subalign, expected)

    def test_multi_alpha_select_multi_alpha_example(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align2 = Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        align.add(align2)
        self.assertEqual(align.truncate_columns(list(range(6, 10))), align2)

    def test_raise_on_out_of_range(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        with self.assertRaises(IndexError):
            _ = align.truncate_columns([1, 5, 9])

    def test_raise_on_split_alphabets(self):
        # raise if selected columns from multicov.one alphabet are not contiguous
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align2 = Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        align.add(align2)
        with self.assertRaises(IndexError):
            _ = align.truncate_columns([1, 5, 9, 4])

    def test_copy_on_truncate(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        subalign = align.truncate_columns(list(range(1, 5)))
        subalign.data[0, 0] = '-'
        self.assertEqual(subalign.data[0, 0], '-')
        self.assertEqual(align.data[0, 1], 'V')

    def test_skip_alpha(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align.add(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        align.add(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        subalign = align.truncate_columns([1, 5, 10, 14, 15])
        self.assertEqual(len(subalign.alphabets), 2)
        self.assertSequenceEqual(subalign.alphabets, ((dna_alphabet, 2), (dna_alphabet, 3)))

    def test_update_reference(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.reference = ReferenceMapping(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        subalign = align.truncate_columns([1, 2, 4, 5, 7])
        expected = Alignment(['VGYTQ', 'VGTEQ', 'GGKD-'], protein_alphabet)
        expected.reference = ReferenceMapping(['b', 'c', 'e', 'f', 'h'])
        self.assertEqual(subalign, expected)


class TestComparison(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment
        self.assertTrue(Alignment() == Alignment())
        self.assertFalse(Alignment() != Alignment())

    def test_empty_vs_not(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        self.assertFalse(Alignment() == align)
        self.assertTrue(Alignment() != align)

    def test_equality_of_empties(self):
        """ Empty matrices can have different shapes. Make sure that they all count as equal. """
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        sub_align = align[[]]
        sub_align.reference = ReferenceMapping()
        self.assertEqual(Alignment(), sub_align)

    def test_equal_self(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        self.assertTrue(align == align)
        self.assertFalse(align != align)

    def test_equal_self_multi_alpha(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align2 = Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        align.add(align2)
        self.assertTrue(align == align)
        self.assertFalse(align != align)

    def test_unequal_different_alphabets(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        data = ['AGACATA', 'A--G-C-']
        align1 = Alignment(data, protein_alphabet)
        align2 = Alignment(data, dna_alphabet)
        self.assertFalse(align1 == align2)
        self.assertTrue(align1 != align2)

    def test_unequal_different_data(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align1 = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align2 = Alignment(['ATACAT', 'GGTACA', 'AA--GG'], dna_alphabet)
        self.assertTrue(align1 != align2)
        self.assertFalse(align1 == align2)

    def test_unequal_different_annotations(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet, rna_alphabet
        align1 = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align1.add(Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet))

        align2 = Alignment(align1)
        align2.annotations['seqw'] = [0.5, 1.2, 0.7]
        align2.annotations['fitness'] = [0, -1, 1]
        self.assertTrue(align1 != align2)
        self.assertFalse(align1 == align2)

    def test_unequal_different_alphabet_widths(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        align1 = Alignment(['AGA', 'A--'], protein_alphabet).add(['CATA', 'G-C-'], dna_alphabet)
        align2 = Alignment(['AGACA', 'A--G-'], protein_alphabet).add(['TA', 'C-'], dna_alphabet)
        self.assertTrue(np.array_equal(align1.data, align2.data))
        self.assertTrue(align1 != align2)
        self.assertFalse(align1 == align2)

    def test_unequal_different_reference(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet
        align1 = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align2 = Alignment(align1)
        align1.reference = ReferenceMapping(list(range(8)))
        align2.reference = ReferenceMapping(list(range(1, 9)))
        self.assertTrue(align1 != align2)
        self.assertFalse(align1 == align2)


class TestAdd(unittest.TestCase):
    def test_add_empty(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align1 = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align2 = Alignment(align1)
        align2.add(Alignment())
        self.assertEqual(align1, align2)

    def test_add_to_empty(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align1 = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align2 = Alignment()
        align2.add(align1)
        self.assertEqual(align1, align2)

    def test_add_alignment(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        align1 = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align2 = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align = Alignment(align1)
        align.add(align2)
        self.assertNotEqual(align, align1)
        self.assertNotEqual(align, align2)
        self.assertSequenceEqual(align.alphabets, [(protein_alphabet, 8), (dna_alphabet, 6)])
        self.assertTrue(np.array_equal(align.data, np.asmatrix([
            ['I', 'V', 'G', 'G', 'Y', 'T', 'C', 'Q', 'A', 'T', 'A', 'C', 'A', 'T'],
            ['-', 'V', 'G', 'G', 'T', 'E', 'A', 'Q', 'G', 'A', 'T', 'A', 'C', 'A'],
            ['I', 'G', 'G', '-', 'K', 'D', 'T', '-', 'A', 'A', '-', '-', 'G', 'G']
        ])))

    def test_add_list_of_strings(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, rna_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        align.add(['AG', 'AA', '--'], rna_alphabet)
        self.assertSequenceEqual(align.alphabets, [(protein_alphabet, 4), (rna_alphabet, 2)])
        self.assertTrue(np.array_equal(align.data, np.asmatrix([
            ['A', 'C', 'C', 'W', 'A', 'G'],
            ['F', 'F', 'H', '-', 'A', 'A'],
            ['-', '-', '-', 'D', '-', '-']
        ])))

    def test_add_array(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, rna_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        align.add([['A', 'G'], ['A', 'A'], ['-', '-']], rna_alphabet)
        self.assertSequenceEqual(align.alphabets, [(protein_alphabet, 4), (rna_alphabet, 2)])
        self.assertTrue(np.array_equal(align.data, np.asmatrix([
            ['A', 'C', 'C', 'W', 'A', 'G'],
            ['F', 'F', 'H', '-', 'A', 'A'],
            ['-', '-', '-', 'D', '-', '-']
        ])))

    def test_return_self(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, rna_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        ret_align = align.add([['A', 'G'], ['A', 'A'], ['-', '-']], rna_alphabet)
        self.assertIs(align, ret_align)

    def test_raise_on_add_wrong_length(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, rna_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        with self.assertRaises(ValueError):
            align.add([['A', 'G'], ['A', 'A'], ['-', '-'], ['C', 'C']], rna_alphabet)

    def test_set_default_reference_on_empty(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet
        align = Alignment()
        align.add(['ACCW', 'FFH-', '---D'], protein_alphabet)
        self.assertEqual(align.reference, ReferenceMapping(list(range(4))))

    def test_set_default_reference_on_nonempty(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        align.add([['A', 'G'], ['A', 'A'], ['-', '-']], dna_alphabet)
        self.assertEqual(len(align.reference.seqs), 2)
        self.assertSequenceEqual(align.reference.seqs[1], list(range(2)))


class TestAsMatrix(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment
        self.assertTrue(np.array_equal(Alignment().as_matrix(), np.asmatrix([])))

    def test_protein_example(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        expected = np.asmatrix([
            ['I', 'V', 'G', 'G', 'Y', 'T', 'C', 'Q'],
            ['-', 'V', 'G', 'G', 'T', 'E', 'A', 'Q'],
            ['I', 'G', 'G', '-', 'K', 'D', 'T', '-']
        ])
        self.assertTrue(np.array_equal(align.as_matrix(), expected))


class TestConvertToInt(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment
        self.assertEqual(Alignment().to_int(), Alignment())

    def test_rna(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, NumericAlphabet
        align = Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        nalign = align.to_int()
        expected = Alignment([[1, 3, 4, 0], [2, 0, 2, 0], [1, 3, 1, 4]], NumericAlphabet(5))
        self.assertEqual(nalign, expected)

    def test_rna_as_matrix(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet
        align = Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        nalign = align.to_int(as_matrix=True)
        expected = np.asmatrix([[1, 3, 4, 0], [2, 0, 2, 0], [1, 3, 1, 4]])
        self.assertTrue(np.array_equal(nalign, expected))

    def test_multi_alpha(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, protein_alphabet, NumericAlphabet
        align = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        align.add(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        nalign = align.to_int()
        expected = Alignment([[1, 2, 2, 19], [5, 5, 7, 0], [0, 0, 0, 3]], NumericAlphabet(21))
        expected.add(Alignment([[1, 3, 4, 0], [2, 0, 2, 0], [1, 3, 1, 4]], NumericAlphabet(5)))
        self.assertEqual(nalign, expected)

    def test_multi_alpha_single_chunk(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, protein_alphabet, NumericAlphabet
        align = Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        align.add(['ACCW', 'FFH-', '---D'], protein_alphabet)
        nalign = align.to_int(single_chunk=True)
        expected = Alignment([
            [1, 3, 4, 0, 1, 2, 2, 19],
            [2, 0, 2, 0, 5, 5, 7, 0],
            [1, 3, 1, 4, 0, 0, 0, 3]], NumericAlphabet(21))
        self.assertEqual(nalign, expected)

    def test_multi_alpha_as_matrix(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, protein_alphabet
        align = Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        align.add(['ACCW', 'FFH-', '---D'], protein_alphabet)
        nalign = align.to_int(as_matrix=True)
        expected = np.asmatrix([
            [1, 3, 4, 0, 1, 2, 2, 19],
            [2, 0, 2, 0, 5, 5, 7, 0],
            [1, 3, 1, 4, 0, 0, 0, 3]])
        self.assertTrue(np.array_equal(nalign, expected))


class TestConvertFromInt(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        self.assertEqual(Alignment.from_int([[]], protein_alphabet), Alignment())

    def test_rna(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet
        align = Alignment.from_int([[1, 3, 4, 0], [2, 0, 2, 0], [1, 3, 1, 4]], rna_alphabet)
        expected = Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        self.assertEqual(align, expected)

    def test_multi_alpha(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, protein_alphabet
        align = Alignment.from_int((
            [[1, 2, 2, 19], [5, 5, 7, 0], [0, 0, 0, 3]],
            [[1, 3, 4, 0], [2, 0, 2, 0], [1, 3, 1, 4]]
        ), (protein_alphabet, rna_alphabet))

        expected = Alignment(['ACCW', 'FFH-', '---D'], protein_alphabet)
        expected.add(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        self.assertEqual(align, expected)

    def test_multi_alpha_single_chunk(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, protein_alphabet
        align = Alignment.from_int([
            [1, 3, 4, 0, 1, 2, 2, 19],
            [2, 0, 2, 0, 5, 5, 7, 0],
            [1, 3, 1, 4, 0, 0, 0, 3]],
            ((rna_alphabet, 4), (protein_alphabet, 4))
        )

        expected = Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        expected.add(['ACCW', 'FFH-', '---D'], protein_alphabet)
        self.assertEqual(align, expected)

    def test_to_int_roundtrip(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.add(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        nalign1 = align.to_int()
        nalign2 = align.to_int(single_chunk=True)

        round1 = Alignment.from_int(nalign1.data, ((protein_alphabet, 8), (dna_alphabet, 6)))
        round2 = Alignment.from_int(nalign2.data, ((protein_alphabet, 8), (dna_alphabet, 6)))

        self.assertEqual(round1, align)
        self.assertEqual(round2, align)


class TestAlphabets(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment
        self.assertEqual(len(Alignment().alphabets), 0)

    def test_single(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        self.assertEqual(len(align.alphabets), 1)
        self.assertEqual(align.alphabets[0], (protein_alphabet, 8))

    def test_multi(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.add(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        self.assertEqual(len(align.alphabets), 2)
        self.assertEqual(align.alphabets[0], (protein_alphabet, 8))
        self.assertEqual(align.alphabets[1], (dna_alphabet, 6))


class TestAnnotations(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment
        self.assertEqual(Alignment().annotations.size, 0)
        self.assertIn('seqw', Alignment().annotations.columns)

    def test_matching_length(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        self.assertIn('seqw', align.annotations.columns)
        self.assertEqual(len(align), 3)
        self.assertEqual(len(align.annotations), 3)

    def test_initially_one(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG', 'AC-T-G'], dna_alphabet)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(align.annotations['seqw'] == 1))


class TestUpdateSequenceWeights(unittest.TestCase):
    def test_against_lucy(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from scipy.io import loadmat
        import os.path
        lucy = loadmat(os.path.join('test_data', 'lucy_dca_pdz_small.mat'),
                       squeeze_me=True)
        alignment = Alignment(lucy['alignment']['data'][()], protein_alphabet)
        alignment.update_sequence_weights(0.7, memory_saver=False)
        self.assertTrue(np.allclose(alignment.annotations['seqw'], lucy['seqw']))

    def test_protein_example_threshold_75(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC',
            'CC--CCCCCC'], protein_alphabet)
        align.update_sequence_weights(0.75, memory_saver=False)
        expected_seqw = [0.5, 0.5, 1, 0.25, 0.25, 0.25, 0.25]
        self.assertTrue(np.allclose(align.annotations['seqw'], expected_seqw))

    def test_protein_example_threshold_85(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC',
            'CC--CCCCCC'], protein_alphabet)
        align.update_sequence_weights(0.85)
        expected_seqw = [1, 1, 1, 1.0/3, 1.0/3, 1.0/3, 1.0/3]
        self.assertTrue(np.allclose(align.annotations['seqw'], expected_seqw))

    def test_protein_example_threshold_75_memory_saver_option_with_numba(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC',
            'CC--CCCCCC'], protein_alphabet)
        align.update_sequence_weights(0.75, memory_saver=True, no_numba=False)
        expected_seqw = [0.5, 0.5, 1, 0.25, 0.25, 0.25, 0.25]
        self.assertTrue(np.allclose(align.annotations['seqw'], expected_seqw))

    def test_protein_example_threshold_75_memory_saver_option_no_numba(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC',
            'CC--CCCCCC'], protein_alphabet)
        align.update_sequence_weights(0.75, memory_saver=True, no_numba=True)
        expected_seqw = [0.5, 0.5, 1, 0.25, 0.25, 0.25, 0.25]
        self.assertTrue(np.allclose(align.annotations['seqw'], expected_seqw))


class TestReferenceMapping(unittest.TestCase):
    def test_empty_align(self):
        from multicov.alignment import Alignment, ReferenceMapping
        self.assertTrue(hasattr(Alignment(), 'reference'))
        self.assertEqual(Alignment().reference, ReferenceMapping())

    def test_empty(self):
        from multicov.alignment import ReferenceMapping
        ref = ReferenceMapping()
        self.assertTrue(hasattr(ref, 'seqs'))
        self.assertEqual(len(ref), 0)
        self.assertEqual(len(ref.seqs), 0)

    def test_constructor_single_ref(self):
        from multicov.alignment import ReferenceMapping
        ref = ReferenceMapping(list(range(6)))
        self.assertEqual(len(ref), 6)
        self.assertEqual(len(ref.seqs), 1)
        self.assertTrue(np.array_equal(ref.seqs[0], list(range(6))))

    def test_constructor_multi_ref(self):
        from multicov.alignment import ReferenceMapping
        ref = ReferenceMapping((list(range(1, 5)), [3, 7, 10, 12], ['1', '3']))
        self.assertEqual(len(ref), 10)
        self.assertEqual(len(ref.seqs), 3)
        self.assertTrue(np.array_equal(ref.seqs[0], [1, 2, 3, 4]))
        self.assertTrue(np.array_equal(ref.seqs[1], [3, 7, 10, 12]))
        self.assertTrue(np.array_equal(ref.seqs[2], ['1', '3']))

    def test_constructor_mixed_type_ref(self):
        from multicov.alignment import ReferenceMapping
        ref = ReferenceMapping([1, '7a', 9])
        self.assertEqual(len(ref), 3)
        self.assertEqual(len(ref.seqs), 1)
        self.assertTrue(np.array_equal(ref.seqs[0], [1, '7a', 9]))

    def test_getitem(self):
        from multicov.alignment import ReferenceMapping
        ref = ReferenceMapping([1, '7a', 9])
        self.assertEqual(ref[0], 1)
        self.assertEqual(ref[1], '7a')
        self.assertEqual(ref[2], 9)

    def test_getitem_multi_alpha(self):
        from multicov.alignment import ReferenceMapping
        ref = ReferenceMapping((list(range(1, 5)), [3, 7, 10, 12], ['1', '3']))
        self.assertEqual(ref[0], 1)
        self.assertEqual(ref[3], 4)
        self.assertEqual(ref[5], 7)
        self.assertEqual(ref[8], '1')
        self.assertEqual(ref[9], '3')

    def test_indexing_raises_on_empty(self):
        from multicov.alignment import ReferenceMapping
        with self.assertRaises(IndexError):
            _ = ReferenceMapping()[1]

    def test_indexing_raises_on_out_of_range(self):
        from multicov.alignment import ReferenceMapping
        ref = ReferenceMapping([1, 8, 9])
        with self.assertRaises(IndexError):
            _ = ref[-1]
        with self.assertRaises(IndexError):
            _ = ref[4]

    def test_compare_empty(self):
        from multicov.alignment import ReferenceMapping
        self.assertTrue(ReferenceMapping() == ReferenceMapping())
        self.assertFalse(ReferenceMapping() != ReferenceMapping())

    def test_compare_self(self):
        from multicov.alignment import ReferenceMapping
        ref = ReferenceMapping([1, 8, 9])
        self.assertTrue(ref == ref)
        self.assertFalse(ref != ref)

    def test_compare_self_multi(self):
        from multicov.alignment import ReferenceMapping
        ref = ReferenceMapping((list(range(1, 5)), [3, 7, 10, 12], ['1', '3']))
        self.assertTrue(ref == ref)
        self.assertFalse(ref != ref)

    def test_compare_different_nseq(self):
        from multicov.alignment import ReferenceMapping
        ref1 = ReferenceMapping([1, 8, 9])
        ref2 = ReferenceMapping((list(range(1, 5)), [3, 7, 10, 12], ['1', '3']))
        self.assertFalse(ref1 == ref2)
        self.assertTrue(ref1 != ref2)

    def test_compare_different_seqs(self):
        from multicov.alignment import ReferenceMapping
        ref1 = ReferenceMapping([1, 8, 9])
        ref2 = ReferenceMapping([1, 8, '9'])
        self.assertFalse(ref1 == ref2)
        self.assertTrue(ref1 != ref2)

    def test_compare_same_contents_different_object(self):
        from multicov.alignment import ReferenceMapping
        ref1 = ReferenceMapping([1, 8, 9])
        ref2 = ReferenceMapping([1, 8, 9])
        self.assertTrue(ref1 == ref2)
        self.assertFalse(ref1 != ref2)


class TestSwap(unittest.TestCase):
    def test_protein_example(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.swap(1, 2)
        expected = Alignment(['IVGGYTCQ', 'IGG-KDT-', '-VGGTEAQ'], protein_alphabet)
        self.assertEqual(align, expected)


class TestToBinary(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment, ReferenceMapping
        bin_align = Alignment().to_binary()
        self.assertTrue(hasattr(bin_align, 'data'))
        self.assertTrue(hasattr(bin_align, 'alphabets'))
        self.assertTrue(hasattr(bin_align, 'reference'))
        self.assertTrue(hasattr(bin_align, 'annotations'))
        self.assertEqual(np.size(bin_align.data), 0)
        self.assertEqual(np.size(bin_align.alphabets), 0)
        self.assertEqual(bin_align.annotations.size, 0)
        self.assertEqual(bin_align.reference, ReferenceMapping())

    def test_rna_example(self):
        from multicov.alignment import Alignment
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import rna_alphabet
        align = Alignment(['ACA', 'GUA', '-A-'], alphabet=rna_alphabet)
        bin_align = align.to_binary()
        expected = BinaryAlignment([
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], rna_alphabet)
        self.assertEqual(bin_align, expected)

    def test_multi_alpha(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet, rna_alphabet
        align1 = Alignment(['ACA', 'GUA', '-A-'], alphabet=rna_alphabet)
        align2 = Alignment(['DF', 'YA', '-C'], alphabet=protein_alphabet)

        align = Alignment(align1).add(align2)
        align.reference = ReferenceMapping((list(range(1, 4)), list(range(2))))
        align.annotations['seqw'] = [0.5, 1.5, 0.2]

        bin_align = align.to_binary()

        expected1 = np.asmatrix([[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        #                         A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
        expected2 = np.asmatrix([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.assertSequenceEqual(bin_align.alphabets, [(rna_alphabet, 3), (protein_alphabet, 2)])
        self.assertTrue(np.array_equal(bin_align.data.todense(), np.hstack((expected1, expected2))))
        self.assertIs(bin_align.reference, align.reference)
        self.assertIs(bin_align.annotations, align.annotations)


class TestEliminateSimilarSequences(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment
        self.assertEqual(Alignment().eliminate_similar_sequences(0.8), Alignment())

    def test_protein_threshold_1(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC',
            'CC--CCCCCC'], protein_alphabet)
        align.eliminate_similar_sequences(0.75, memory_saver=False)
        expected = Alignment([
            'AAAAAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC'], protein_alphabet)
        self.assertEqual(align, expected)

    def test_protein_threshold_2(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC',
            'CC--CCCCCC'], protein_alphabet)
        align.eliminate_similar_sequences(0.85)
        expected = Alignment([
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC'], protein_alphabet)
        self.assertEqual(align, expected)

    def test_least_gapped(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'A-FGAAAAAA',
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'CC--CCCCCC',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC'], protein_alphabet)
        align.eliminate_similar_sequences(0.85, memory_saver=False)
        expected = Alignment([
            'A-FGAAAAAA',
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'CCCCCCCCCC'], protein_alphabet)
        self.assertEqual(align, expected)

    def test_mem_saver(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC',
            'CC--CCCCCC'], protein_alphabet)
        align.eliminate_similar_sequences(0.75, memory_saver=True)
        expected = Alignment([
            'AAAAAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC'], protein_alphabet)
        self.assertEqual(align, expected)

    def test_mem_saver_nonumba(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC',
            'CC--CCCCCC'], protein_alphabet)
        align.eliminate_similar_sequences(0.75, memory_saver=True, no_numba=True)
        expected = Alignment([
            'AAAAAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC'], protein_alphabet)
        self.assertEqual(align, expected)

    def test_least_gapped_mem_saver(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'A-FGAAAAAA',
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'CC--CCCCCC',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC'], protein_alphabet)
        align.eliminate_similar_sequences(0.85, memory_saver=True)
        expected = Alignment([
            'A-FGAAAAAA',
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'CCCCCCCCCC'], protein_alphabet)
        self.assertEqual(align, expected)


class TestGapStructure(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment
        self.assertEqual(np.size(Alignment().get_gap_structure()), 0)

    def test_protein(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment([
            'AAAAAAAAAA',
            'AACDAAAAAA',
            'A-FGAAAAAA',
            'CCCCCCCCCC',
            'CC-CCCCCCC',
            'CCC-CCCCCC',
            'CC--CCCCCC'], protein_alphabet)
        expected = np.zeros((7, 10), dtype=bool)
        expected[[2, 4, 5, 6, 6], [1, 2, 3, 2, 3]] = True
        self.assertTrue(np.array_equal(align.get_gap_structure(), expected))

    def test_multi_alpha(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, rna_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.add(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        expected = np.zeros((3, 12), dtype=bool)
        expected[[0, 1, 1, 1, 2, 2], [11, 0, 9, 11, 3, 7]] = True
        self.assertTrue(np.array_equal(align.get_gap_structure(), expected))

    def test_gapless_alpha(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import NumericAlphabet
        align = Alignment([[0, 1, 2, 3], [2, 3, 2, 0]], NumericAlphabet(4, has_gap=False))
        self.assertFalse(np.any(align.get_gap_structure()))


class TestExtend(unittest.TestCase):
    def test_add_list_to_empty(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align1 = Alignment()
        data = ['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-']
        align1.extend(data, protein_alphabet)
        expected = Alignment(data, protein_alphabet)
        self.assertEqual(align1, expected)

    def test_add_alignment_to_empty(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align1 = Alignment()
        align2 = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align1.extend(align2)
        self.assertEqual(align1, align2)

    def test_add_single_alphabet(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align1 = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align2 = Alignment(['-VGGTEAQ', 'IGG-KDT-', 'IVGGYTCQ'], protein_alphabet)
        align1.extend(align2)
        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-',
                              '-VGGTEAQ', 'IGG-KDT-', 'IVGGYTCQ'], protein_alphabet)
        self.assertEqual(align1, expected)

    def test_add_multi_alphabet(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet, protein_alphabet
        align1 = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet).add(
            ['AGT-', 'C-C-', 'AGAT'], dna_alphabet)
        align2 = Alignment(['GTEGYTCQ', 'TCQ-VGAQ', 'IGVGADT-'], protein_alphabet).add(
            ['GAT-', 'CACT', 'A--T'], dna_alphabet)
        align1.extend(align2)
        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-',
                              'GTEGYTCQ', 'TCQ-VGAQ', 'IGVGADT-'], protein_alphabet).add(
            ['AGT-', 'C-C-', 'AGAT', 'GAT-', 'CACT', 'A--T'], dna_alphabet
        )
        self.assertEqual(align1, expected)

    def test_raise_on_different_structure(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet, protein_alphabet
        align1 = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet).add(
            ['AGT-', 'C-C-', 'AGAT'], dna_alphabet)
        align2 = Alignment(['GTEGYTC', 'TCQ-VGA', 'IGVGADT'], protein_alphabet).add(
            ['GAT-A', 'CACTA', 'A--TA'], dna_alphabet)
        with self.assertRaises(ValueError):
            align1.extend(align2)

    def test_raise_on_different_refseq(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet
        align1 = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align2 = Alignment(['-VGGTEAQ', 'IGG-KDT-', 'IVGGYTCQ'], protein_alphabet)
        align2.reference = ReferenceMapping([2, 3, 4, 6, 7, 8, 10, 11])
        with self.assertRaises(ValueError):
            align1.extend(align2)

    def test_ignore_reference(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet
        align1 = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align2 = Alignment(['-VGGTEAQ', 'IGG-KDT-', 'IVGGYTCQ'], protein_alphabet)
        align2.reference = ReferenceMapping([2, 3, 4, 6, 7, 8, 10, 11])
        align1.extend(align2, ignore_reference=True)
        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-',
                              '-VGGTEAQ', 'IGG-KDT-', 'IVGGYTCQ'], protein_alphabet)
        self.assertEqual(align1, expected)

    def test_join_annotations(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        import pandas as pd
        align1 = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align2 = Alignment(['-VGGTEAQ', 'IGG-KDT-', 'IVGGYTCQ'], protein_alphabet)
        align1.annotations['seqw'] = [0.5, 0.7, 0.9]
        align2.annotations['seqw'] = [0.3, 1.2, 1.7]
        align1.annotations['field1'] = ['foo', 'bar', 'z']
        align2.annotations['field2'] = ['oof', 'rab', 'z']
        align1.extend(align2)
        self.assertIn('seqw', align1.annotations.columns)
        self.assertIn('field1', align1.annotations.columns)
        self.assertIn('field2', align1.annotations.columns)
        self.assertTrue(np.allclose(align1.annotations['seqw'], [0.5, 0.7, 0.9, 0.3, 1.2, 1.7]))
        self.assertSequenceEqual(list(align1.annotations['field1'][:3]), ['foo', 'bar', 'z'])
        self.assertSequenceEqual(list(align1.annotations['field2'][3:]), ['oof', 'rab', 'z'])
        self.assertSequenceEqual(list(pd.isnull(align1['field1'])), [False, False, False, True, True, True])
        self.assertSequenceEqual(list(pd.isnull(align1['field2'])), [True, True, True, False, False, False])

    def test_return_self(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet, protein_alphabet
        align1 = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet).add(
            ['AGT-', 'C-C-', 'AGAT'], dna_alphabet)
        align2 = Alignment(['GTEGYTCQ', 'TCQ-VGAQ', 'IGVGADT-'], protein_alphabet).add(
            ['GAT-', 'CACT', 'A--T'], dna_alphabet)
        ret_val = align1.extend(align2)
        self.assertIs(ret_val, align1)


class TestGetWidth(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment
        self.assertEqual(Alignment().get_width(), 0)

    def test_protein(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        self.assertEqual(align.get_width(), 8)

    def test_multi_alpha(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet, dna_alphabet
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align2 = Alignment(['AGU-', 'C-C-', 'AGAU'], rna_alphabet)
        align.add(align2)
        self.assertEqual(align.get_width(), 10)
