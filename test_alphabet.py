import unittest

import numpy as np


class TestName(unittest.TestCase):
    def test_default_name(self):
        from alphabet import Alphabet
        alpha = Alphabet()
        self.assertTrue(hasattr(alpha, 'name'))
        self.assertEqual(alpha.name, 'none')

    def test_protein_name(self):
        from alphabet import protein_alphabet
        self.assertEqual(protein_alphabet.name, 'protein')

    def test_dna_name(self):
        from alphabet import dna_alphabet
        self.assertEqual(dna_alphabet.name, 'dna')

    def test_rna_name(self):
        from alphabet import rna_alphabet
        self.assertEqual(rna_alphabet.name, 'rna')

    def test_numeric_name_base_0(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(7)
        self.assertEqual(alpha.name, 'numeric[0:7]')

    def test_numeric_name_nonzero_base(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(3, 7)
        self.assertEqual(alpha.name, 'numeric[3:7]')


class TestHasGap(unittest.TestCase):
    def test_default_nogap(self):
        from alphabet import Alphabet
        alpha = Alphabet()
        self.assertTrue(hasattr(alpha, 'has_gap'))
        self.assertFalse(alpha.has_gap)

    def test_protein_hasgap(self):
        from alphabet import protein_alphabet
        self.assertTrue(protein_alphabet.has_gap)

    def test_dna_hasgap(self):
        from alphabet import dna_alphabet
        self.assertTrue(dna_alphabet.has_gap)

    def test_rna_hasgap(self):
        from alphabet import rna_alphabet
        self.assertTrue(rna_alphabet.has_gap)

    def test_numeric_base0_default_gap(self):
        from alphabet import NumericAlphabet
        self.assertTrue(NumericAlphabet(6).has_gap)

    def test_numeric_nonzero_base_default_nogap(self):
        from alphabet import NumericAlphabet
        self.assertFalse(NumericAlphabet(3, 6).has_gap)

    def test_numeric_base0_force_nogap(self):
        from alphabet import NumericAlphabet
        self.assertFalse(NumericAlphabet(8, has_gap=False).has_gap)

    def test_numeric_nonzero_base_force_gap(self):
        from alphabet import NumericAlphabet
        self.assertTrue(NumericAlphabet(3, 9, has_gap=True).has_gap)


class TestLetters(unittest.TestCase):
    def test_default_letters(self):
        from alphabet import Alphabet
        alpha = Alphabet()
        self.assertTrue(hasattr(alpha, 'letters'))
        self.assertEqual(len(alpha.letters()), 0)

    def test_default_letters_no_gap(self):
        from alphabet import Alphabet
        alpha = Alphabet()
        self.assertEqual(len(alpha.letters(no_gap=True)), 0)

    def test_protein_letters(self):
        from alphabet import protein_alphabet
        self.assertTrue(np.array_equal(protein_alphabet.letters(), list('-ACDEFGHIKLMNPQRSTVWY')))

    def test_protein_letters_nogap(self):
        from alphabet import protein_alphabet
        self.assertTrue(np.array_equal(protein_alphabet.letters()[1:], protein_alphabet.letters(no_gap=True)))

    def test_dna_letters(self):
        from alphabet import dna_alphabet
        self.assertTrue(np.array_equal(dna_alphabet.letters(), list('-ACGT')))

    def test_dna_letters_nogap(self):
        from alphabet import dna_alphabet
        self.assertTrue(np.array_equal(dna_alphabet.letters()[1:], dna_alphabet.letters(no_gap=True)))

    def test_rna_letters(self):
        from alphabet import rna_alphabet
        self.assertTrue(np.array_equal(rna_alphabet.letters(), list('-ACGU')))

    def test_rna_letters_nogap(self):
        from alphabet import rna_alphabet
        self.assertTrue(np.array_equal(rna_alphabet.letters()[1:], rna_alphabet.letters(no_gap=True)))

    def test_numeric_letters_base0(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(5)
        self.assertTrue(np.array_equal(alpha.letters(), list(range(5))))

    def test_numeric_letters_nogap_base0_gapful(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(4)
        self.assertTrue(np.array_equal(alpha.letters(no_gap=True), list(range(1, 4))))

    def test_numeric_letters_nogap_base0_gapless(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(4, has_gap=False)
        self.assertTrue(np.array_equal(alpha.letters(no_gap=True), list(range(4))))

    def test_numeric_letters_nonzero_base(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(2, 7)
        self.assertTrue(np.array_equal(alpha.letters(), list(range(2, 7))))

    def test_numeric_letters_nogap_nonzero_base_gapless(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(5, 11)
        self.assertTrue(np.array_equal(alpha.letters(no_gap=True), list(range(5, 11))))

    def test_numeric_letters_nogap_nonzero_base_gapful(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(5, 11, has_gap=True)
        self.assertTrue(np.array_equal(alpha.letters(no_gap=True), list(range(6, 11))))

    def test_numeric_base0_nonpositive_end(self):
        from alphabet import NumericAlphabet
        self.assertEqual(len(NumericAlphabet(0).letters()), 0)
        self.assertEqual(len(NumericAlphabet(-5).letters()), 0)

    def test_numeric_nonzero_base_smaller_end(self):
        from alphabet import NumericAlphabet
        self.assertEqual(len(NumericAlphabet(7, 5).letters()), 0)


class TestConstructor(unittest.TestCase):
    def test_construct_with_name(self):
        from alphabet import Alphabet
        alpha = Alphabet(name='test')
        self.assertEqual(alpha.name, 'test')
        self.assertEqual(len(alpha.letters()), 0)
        self.assertFalse(alpha.has_gap)

    def test_construct_with_letters(self):
        from alphabet import Alphabet
        alpha = Alphabet(letters=list('abc'))
        self.assertEqual(alpha.name, 'none')
        self.assertSequenceEqual(alpha.letters(), 'abc')
        self.assertFalse(alpha.has_gap)

    def test_construct_with_letters_and_gap(self):
        from alphabet import Alphabet
        alpha = Alphabet(letters=list('.defg'), has_gap=True)
        self.assertEqual(alpha.name, 'none')
        self.assertSequenceEqual(alpha.letters(), '.defg')
        self.assertTrue(alpha.has_gap)

    def test_copy(self):
        from alphabet import Alphabet, protein_alphabet
        alpha = Alphabet(protein_alphabet)
        self.assertEqual(alpha, protein_alphabet)

    def test_changing_copy_leaves_original_unchanged(self):
        from alphabet import Alphabet
        alphabet1 = Alphabet(name='the_one', letters=list('fog'), has_gap=False)
        alphabet2 = Alphabet(alphabet1)
        alphabet2.has_gap = True
        alphabet2.name = 'the_two'
        self.assertNotEqual(alphabet1.has_gap, alphabet2.has_gap)
        self.assertNotEqual(alphabet1.name, alphabet2.name)

    def test_deep_copy_letters(self):
        from alphabet import Alphabet
        alphabet1 = Alphabet(name='the_one', letters=list('fog'), has_gap=False)
        alphabet2 = Alphabet(alphabet1)
        alphabet2._letters[1] = '-'
        self.assertSequenceEqual(alphabet1.letters(), list('fog'))
        self.assertSequenceEqual(alphabet2.letters(), list('f-g'))


class TestComparison(unittest.TestCase):
    def test_empty_equal(self):
        from alphabet import Alphabet
        # test both == and != operators
        self.assertTrue(Alphabet() == Alphabet())
        self.assertFalse(Alphabet() != Alphabet())

    def test_equal_self(self):
        from alphabet import protein_alphabet
        self.assertTrue(protein_alphabet == protein_alphabet)
        self.assertFalse(protein_alphabet != protein_alphabet)

    def test_name_difference(self):
        from alphabet import Alphabet
        alphabet1 = Alphabet(name='aleph')
        alphabet2 = Alphabet(name='bet')
        # test both == and != operators
        self.assertTrue(alphabet1 != alphabet2)
        self.assertFalse(alphabet1 == alphabet2)

    def test_gap_difference(self):
        from alphabet import Alphabet, protein_alphabet
        alphabet2 = Alphabet(protein_alphabet)
        alphabet2.has_gap = False
        # test both == and != operators
        self.assertTrue(protein_alphabet != alphabet2)
        self.assertFalse(protein_alphabet == alphabet2)

    def test_letter_difference(self):
        from alphabet import Alphabet
        alphabet1 = Alphabet(letters=list('aleph'))
        alphabet2 = Alphabet(letters=list('bet'))
        # test both == and != operators
        self.assertTrue(alphabet1 != alphabet2)
        self.assertFalse(alphabet1 == alphabet2)


class TestIndex(unittest.TestCase):
    def test_out_of_range(self):
        from alphabet import Alphabet
        with self.assertRaises(IndexError):
            _ = Alphabet()[0]

    def test_index_matches_letter_order(self):
        from alphabet import protein_alphabet
        for i, ch in enumerate(protein_alphabet.letters()):
            self.assertEqual(protein_alphabet[i], ch)

    def test_index_numeric(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(4, 8)
        for i, n in enumerate(alpha.letters()):
            self.assertEqual(alpha[i], n)


class TestSize(unittest.TestCase):
    def test_size_empty(self):
        from alphabet import Alphabet
        self.assertEqual(Alphabet().size(), 0)

    def test_len_empty(self):
        from alphabet import Alphabet
        self.assertEqual(len(Alphabet()), 0)

    def test_protein_size(self):
        from alphabet import protein_alphabet
        self.assertEqual(protein_alphabet.size(), 21)

    def test_protein_size_nogap(self):
        from alphabet import protein_alphabet
        self.assertEqual(protein_alphabet.size(no_gap=True), 20)

    def test_protein_len(self):
        from alphabet import protein_alphabet
        self.assertEqual(len(protein_alphabet), 21)

    def test_dna_size(self):
        from alphabet import dna_alphabet
        self.assertEqual(dna_alphabet.size(), 5)

    def test_dna_size_nogap(self):
        from alphabet import dna_alphabet
        self.assertEqual(dna_alphabet.size(no_gap=True), 4)

    def test_dna_len(self):
        from alphabet import dna_alphabet
        self.assertEqual(len(dna_alphabet), 5)

    def test_rna_size(self):
        from alphabet import rna_alphabet
        self.assertEqual(rna_alphabet.size(), 5)

    def test_rna_size_nogap(self):
        from alphabet import rna_alphabet
        self.assertEqual(rna_alphabet.size(no_gap=True), 4)

    def test_rna_len(self):
        from alphabet import rna_alphabet
        self.assertEqual(len(rna_alphabet), 5)

    def test_numeric_base0_size(self):
        from alphabet import NumericAlphabet
        self.assertEqual(NumericAlphabet(8).size(), 8)

    def test_numeric_base0_size_nogap(self):
        from alphabet import NumericAlphabet
        self.assertEqual(NumericAlphabet(8).size(no_gap=True), 7)

    def test_numeric_base0_len(self):
        from alphabet import NumericAlphabet
        self.assertEqual(len(NumericAlphabet(8)), 8)

    def test_numeric_nonzero_base_size(self):
        from alphabet import NumericAlphabet
        self.assertEqual(NumericAlphabet(11, 17).size(), 6)

    def test_numeric_nonzero_base_size_nogap(self):
        from alphabet import NumericAlphabet
        self.assertEqual(NumericAlphabet(10, 16).size(no_gap=True), 6)

    def test_numeric_gapless_nonzero_base_size_nogap(self):
        from alphabet import NumericAlphabet
        self.assertEqual(NumericAlphabet(10, 16, has_gap=True).size(no_gap=True), 5)

    def test_numeric_nonzero_base_len(self):
        from alphabet import NumericAlphabet
        self.assertEqual(len(NumericAlphabet(11, 17)), 6)


class TestConvertToInt(unittest.TestCase):
    def test_empty_sequence(self):
        from alphabet import protein_alphabet
        self.assertEqual(len(protein_alphabet.to_int([])), 0)

    def test_raises_on_empty_alpha(self):
        from alphabet import Alphabet
        with self.assertRaises(KeyError):
            _ = Alphabet().to_int([])

    def test_rna_scalar_example(self):
        from alphabet import rna_alphabet
        self.assertEqual(rna_alphabet.to_int('C'), 2)

    def test_protein_sequence_example(self):
        from alphabet import protein_alphabet
        self.assertTrue(np.array_equal(protein_alphabet.to_int(list('A-TC')), [1, 0, 17, 2]))

    def test_dna_sequence_matrix_example(self):
        from alphabet import dna_alphabet
        result = dna_alphabet.to_int([list('ACG'), list('T-G')])
        self.assertEqual(len(result), 2)
        self.assertTrue(np.array_equal(result[0], [1, 2, 3]))
        self.assertTrue(np.array_equal(result[1], [4, 0, 3]))

    def test_dna_multi_sequence_uneq_len_example(self):
        from alphabet import dna_alphabet
        result = dna_alphabet.to_int([list('GTC-'), list('AGGGT'), list('-A')])
        self.assertEqual(len(result), 3)
        self.assertTrue(np.array_equal(result[0], [3, 4, 2, 0]))
        self.assertTrue(np.array_equal(result[1], [1, 3, 3, 3, 4]))
        self.assertTrue(np.array_equal(result[2], [0, 1]))

    def test_numeric_sequence_example_base0(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(7)
        result = alpha.to_int([0, 2, 3])
        self.assertTrue(np.array_equal(result, [0, 2, 3]))

    def test_numeric_sequence_example_nonzero_base(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(3, 8)
        self.assertTrue(np.array_equal(alpha.to_int([3, 7, 5]), [0, 4, 2]))

    def test_raises_on_invalid_letter(self):
        from alphabet import rna_alphabet
        with self.assertRaises(KeyError):
            _ = rna_alphabet.to_int(list('-GHT.'))


class TestConvertFromInt(unittest.TestCase):
    def test_empty_sequence(self):
        from alphabet import protein_alphabet
        self.assertEqual(len(protein_alphabet.from_int([])), 0)

    def test_raises_on_empty_alpha(self):
        from alphabet import Alphabet
        with self.assertRaises(KeyError):
            _ = Alphabet().from_int([])

    def test_rna_sequence_example(self):
        from alphabet import rna_alphabet
        self.assertTrue(np.array_equal(rna_alphabet.from_int([0, 4, 2, 3, 1]), list('-UCGA')))

    def test_protein_sequence_matrix_example(self):
        from alphabet import protein_alphabet
        result = protein_alphabet.from_int([[10, 5, 0], [20, 7, 12]])
        self.assertTrue(np.array_equal(result[0], list('LF-')))
        self.assertTrue(np.array_equal(result[1], list('YHN')))

    def test_numeric_sequence_example_base0(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(10)
        self.assertTrue(np.array_equal(alpha.from_int([8, 0, 4, 2]), [8, 0, 4, 2]))

    def test_numeric_sequence_example_nonzero_base(self):
        from alphabet import NumericAlphabet
        alpha = NumericAlphabet(5, 10)
        self.assertTrue(np.array_equal(alpha.from_int([3, 0, 2, 4]), [8, 5, 7, 9]))

    def test_raises_on_invalid_number(self):
        from alphabet import protein_alphabet
        with self.assertRaises(IndexError):
            _ = protein_alphabet.from_int([[5, 0, 20], [3, 22, 3]])

    def test_raises_on_negative_number(self):
        from alphabet import protein_alphabet
        with self.assertRaises(IndexError):
            _ = protein_alphabet.from_int([3, -1, 3])

    def test_protein_scalar_example(self):
        from alphabet import protein_alphabet
        self.assertEqual(protein_alphabet.to_int('H'), 7)
