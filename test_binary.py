import unittest

import numpy as np


class TestConstructor(unittest.TestCase):
    def test_make_empty(self):
        from multicov.binary import BinaryAlignment
        bin_align = BinaryAlignment()
        self.assertTrue(hasattr(bin_align, 'alphabets'))
        self.assertTrue(hasattr(bin_align, 'data'))
        self.assertTrue(hasattr(bin_align, 'reference'))
        self.assertTrue(hasattr(bin_align, 'annotations'))
        self.assertEqual(np.size(bin_align.data), 0)
        self.assertEqual(len(bin_align.alphabets), 0)

    def test_make_from_matrix(self):
        from multicov.binary import BinaryAlignment
        from multicov.alignment import ReferenceMapping
        from multicov.alphabet import rna_alphabet
        bin_data = np.asmatrix([
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        bin_align = BinaryAlignment(bin_data, rna_alphabet)
        self.assertEqual(len(bin_align.alphabets), 1)
        self.assertEqual(bin_align.alphabets[0][0], rna_alphabet)
        self.assertEqual(bin_align.alphabets[0][1], 3)
        self.assertEqual(bin_align.reference, ReferenceMapping(list(range(3))))
        self.assertTrue(np.array_equal(bin_align.data.todense(), bin_data))

    def test_copy(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import dna_alphabet
        bin_align1 = BinaryAlignment([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1]], dna_alphabet)
        bin_align2 = BinaryAlignment(bin_align1)
        self.assertIsNot(bin_align1, bin_align2)
        self.assertEqual(bin_align1, bin_align2)

    def test_changing_copy_leaves_original_unchanged(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet
        bin_align1 = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        bin_align2 = BinaryAlignment(bin_align1)
        bin_align2.annotations['seqw'] = [1, 2, 3]
        self.assertNotEqual(bin_align1, bin_align2)
        self.assertTrue(np.array_equal(bin_align2.annotations['seqw'], [1, 2, 3]))

    def test_deep_copy_data(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import NumericAlphabet
        bin_align1 = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], NumericAlphabet(5))
        bin_align2 = BinaryAlignment(bin_align1)
        self.assertIsNot(bin_align1.data, bin_align2.data)

    def test_trivial_sequence_weights(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        self.assertIn('seqw', bin_align.annotations.columns)
        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(bin_align.annotations['seqw'], 1))

    def test_set_default_reference(self):
        from multicov.binary import BinaryAlignment
        from multicov.alignment import ReferenceMapping
        from multicov.alphabet import NumericAlphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], NumericAlphabet(5))
        self.assertEqual(bin_align.reference, ReferenceMapping(list(range(4))))


class TestLength(unittest.TestCase):
    def test_empty_len(self):
        from multicov.binary import BinaryAlignment
        self.assertEqual(len(BinaryAlignment()), 0)

    def test_nonempty(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import dna_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dna_alphabet)
        self.assertEqual(len(bin_align), 2)


class TestComparison(unittest.TestCase):
    def test_empty(self):
        from multicov.binary import BinaryAlignment
        self.assertTrue(BinaryAlignment() == BinaryAlignment())
        self.assertFalse(BinaryAlignment() != BinaryAlignment())

    def test_empty_vs_not(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        self.assertFalse(BinaryAlignment() == bin_align)
        self.assertTrue(BinaryAlignment() != bin_align)

    def test_equal_self(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        self.assertTrue(bin_align == bin_align)
        self.assertFalse(bin_align != bin_align)

    def test_equal_self_multi_alpha(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import rna_alphabet, dna_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dna_alphabet)
        bin_align.add([
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]], rna_alphabet)
        self.assertTrue(bin_align == bin_align)
        self.assertFalse(bin_align != bin_align)

    def test_unequal_different_alphabets(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import rna_alphabet, dna_alphabet
        bin_data = [
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]]
        bin_align1 = BinaryAlignment(bin_data, dna_alphabet)
        bin_align2 = BinaryAlignment(bin_data, rna_alphabet)
        self.assertFalse(bin_align1 == bin_align2)
        self.assertTrue(bin_align1 != bin_align2)

    def test_unequal_different_data(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import dna_alphabet
        bin_align1 = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dna_alphabet)
        bin_align2 = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dna_alphabet)
        self.assertTrue(bin_align1 != bin_align2)
        self.assertFalse(bin_align1 == bin_align2)

    def test_unequal_different_annotations(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import dna_alphabet, rna_alphabet

        bin_align1 = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dna_alphabet)
        bin_align1.add([
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]], rna_alphabet)

        bin_align2 = BinaryAlignment(bin_align1)
        bin_align2.annotations['seqw'] = [0.5, 1.2]
        bin_align2.annotations['fitness'] = [0, -1]
        self.assertTrue(bin_align1 != bin_align2)
        self.assertFalse(bin_align1 == bin_align2)

    def test_unequal_different_alphabet_widths(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import rna_alphabet, dna_alphabet
        bin_align1 = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dna_alphabet)
        bin_align1.add([
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]], rna_alphabet)

        bin_align2 = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]], dna_alphabet)
        bin_align2.add([
            [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]], rna_alphabet)

        self.assertTrue(np.array_equal(bin_align1.data.todense(), bin_align2.data.todense()))
        self.assertTrue(bin_align1 != bin_align2)
        self.assertFalse(bin_align1 == bin_align2)

    def test_unequal_different_reference(self):
        from multicov.binary import BinaryAlignment
        from multicov.alignment import ReferenceMapping
        from multicov.alphabet import protein_alphabet
        bin_align1 = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        bin_align2 = BinaryAlignment(bin_align1)
        bin_align1.reference = ReferenceMapping(list(range(8)))
        bin_align2.reference = ReferenceMapping(list(range(1, 9)))
        self.assertTrue(bin_align1 != bin_align2)
        self.assertFalse(bin_align1 == bin_align2)


class TestAdd(unittest.TestCase):
    def test_add_empty(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import dna_alphabet
        bin_align1 = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dna_alphabet)
        bin_align2 = BinaryAlignment(bin_align1)
        bin_align2.add(BinaryAlignment())
        self.assertEqual(bin_align1, bin_align2)

    def test_add_to_empty(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import dna_alphabet
        bin_align1 = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dna_alphabet)
        bin_align2 = BinaryAlignment()
        bin_align2.add(bin_align1)
        self.assertEqual(bin_align1, bin_align2)

    def test_add_alignment(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        bin_align1 = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        bin_align2 = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dna_alphabet)
        bin_align = BinaryAlignment(bin_align1)
        bin_align.add(bin_align2)
        self.assertNotEqual(bin_align, bin_align1)
        self.assertNotEqual(bin_align, bin_align2)
        self.assertSequenceEqual(bin_align.alphabets, [(protein_alphabet, 2), (dna_alphabet, 4)])
        self.assertTrue(np.array_equal(bin_align.data.todense(), np.asmatrix([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])))

    def test_add_array(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import dna_alphabet, rna_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dna_alphabet)
        bin_align.add([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], rna_alphabet)

        self.assertSequenceEqual(bin_align.alphabets, [(dna_alphabet, 4), (rna_alphabet, 4)])
        self.assertTrue(np.array_equal(bin_align.data.todense(), np.asmatrix([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
             0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])))

    def test_return_self(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import dna_alphabet, rna_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dna_alphabet)
        ret_align = bin_align.add([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], rna_alphabet)
        self.assertIs(bin_align, ret_align)

    def test_raise_on_add_wrong_length(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet, rna_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        with self.assertRaises(ValueError):
            bin_align.add([
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], rna_alphabet)

    def test_raise_on_add_wrong_width(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet, rna_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        data = np.asmatrix([
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]])
        # don't raise on properly-sized data
        try:
            bin_align.add(data, rna_alphabet)
        except ValueError:
            self.fail("BinaryAlignment.add() raised ValueError on properly-sized data.")
        # raised on improperly-sized
        with self.assertRaises(ValueError):
            bin_align.add(data[:, :-1], rna_alphabet)

    def test_set_default_reference_on_empty(self):
        from multicov.binary import BinaryAlignment
        from multicov.alignment import ReferenceMapping
        from multicov.alphabet import rna_alphabet
        bin_align = BinaryAlignment()
        bin_align.add([
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], rna_alphabet)
        self.assertEqual(bin_align.reference, ReferenceMapping(list(range(4))))

    def test_set_default_reference_on_nonempty(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        bin_align.add([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dna_alphabet)
        self.assertEqual(len(bin_align.reference.seqs), 2)
        self.assertSequenceEqual(bin_align.reference.seqs[1], list(range(4)))


class TestAlphabets(unittest.TestCase):
    def test_empty(self):
        from multicov.binary import BinaryAlignment
        self.assertEqual(len(BinaryAlignment().alphabets), 0)

    def test_single(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        self.assertEqual(len(bin_align.alphabets), 1)
        self.assertEqual(bin_align.alphabets[0], (protein_alphabet, 2))

    def test_multi(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet, dna_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        bin_align.add([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dna_alphabet)
        self.assertEqual(len(bin_align.alphabets), 2)
        self.assertEqual(bin_align.alphabets[0], (protein_alphabet, 2))
        self.assertEqual(bin_align.alphabets[1], (dna_alphabet, 4))


class TestAnnotations(unittest.TestCase):
    def test_empty(self):
        from multicov.binary import BinaryAlignment
        self.assertEqual(BinaryAlignment().annotations.size, 0)
        self.assertIn('seqw', BinaryAlignment().annotations.columns)

    def test_matching_length(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import protein_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], protein_alphabet)
        self.assertIn('seqw', bin_align.annotations.columns)
        self.assertEqual(len(bin_align), 3)
        self.assertEqual(len(bin_align.annotations), 3)

    def test_initially_one(self):
        from multicov.binary import BinaryAlignment
        from multicov.alphabet import dna_alphabet
        bin_align = BinaryAlignment([
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dna_alphabet)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(bin_align.annotations['seqw'] == 1))


class TestFromAlignment(unittest.TestCase):
    def test_empty(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.binary import BinaryAlignment
        bin_align = BinaryAlignment.from_alignment(Alignment())
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
        bin_align = BinaryAlignment.from_alignment(align)
        expected = BinaryAlignment([
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], rna_alphabet)
        self.assertEqual(bin_align, expected)

    def test_multi_alpha(self):
        from multicov.alignment import Alignment, ReferenceMapping
        from multicov.alphabet import protein_alphabet, rna_alphabet
        from multicov.binary import BinaryAlignment
        align1 = Alignment(['ACA', 'GUA', '-A-'], alphabet=rna_alphabet)
        align2 = Alignment(['DF', 'YA', '-C'], alphabet=protein_alphabet)

        align = Alignment(align1).add(align2)
        align.reference = ReferenceMapping((list(range(1, 4)), list(range(2))))
        align.annotations['seqw'] = [0.5, 1.5, 0.2]

        bin_align = BinaryAlignment.from_alignment(align)

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
