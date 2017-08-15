import unittest

import numpy as np


def _slow_get_freq1(alignment):
    width = alignment.get_width()
    bin_width = sum(_[0].size(no_gap=True)*_[1] for _ in alignment.alphabets)
    # here alphabets[i] is alphabet at position i in alignment
    alphabets = sum((_[1]*[_[0]] for _ in alignment.alphabets), [])
    result = np.zeros(bin_width)
    res_i = 0
    for i in range(width):
        letters1 = alphabets[i].letters(no_gap=True)
        for letter1 in letters1:
            # noinspection PyUnresolvedReferences
            mask1 = (alignment.data[:, i] == letter1).squeeze()
            result[res_i] = np.dot(mask1, alignment['seqw'])
            res_i += 1

    result /= np.sum(alignment['seqw'])
    return result


def _slow_get_freq2(alignment):
    width = alignment.get_width()
    bin_width = sum(_[0].size(no_gap=True)*_[1] for _ in alignment.alphabets)
    # here alphabets[i] is alphabet at position i in alignment
    alphabets = sum((_[1]*[_[0]] for _ in alignment.alphabets), [])
    result = np.zeros((bin_width, bin_width))
    res_i = 0
    for i in range(width):
        letters1 = alphabets[i].letters(no_gap=True)
        for letter1 in letters1:
            # noinspection PyUnresolvedReferences
            mask1 = (alignment.data[:, i] == letter1).squeeze()
            res_j = 0
            for j in range(width):
                letters2 = alphabets[j].letters(no_gap=True)
                for letter2 in letters2:
                    # noinspection PyUnresolvedReferences
                    mask2 = (alignment.data[:, j] == letter2).squeeze()
#                    result[res_i][res_j] = np.sum(np.dot(mask1 & mask2, alignment['seqw']))
                    result[res_i][res_j] = np.dot(mask1 & mask2, alignment['seqw'])
                    res_j += 1
            res_i += 1

    result /= np.sum(alignment['seqw'])
    return result


class TestStatisticsConstructor(unittest.TestCase):
    def test_on_empty(self):
        from multicov.alignment import Alignment
        from multicov.statistics import Statistics
        stats = Statistics(Alignment())
        self.assertTrue(hasattr(stats, 'freq1'))
        self.assertTrue(hasattr(stats, 'freq2'))
        self.assertTrue(hasattr(stats, 'cmat'))
        self.assertEqual(len(stats.freq1), 0)
        self.assertEqual(len(stats.freq2), 0)
        self.assertEqual(len(stats.cmat), 0)

    def test_freq1_on_protein(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics
        align = Alignment([
            'WKHNAYDML',
            'KHRCDANC-',
            'LGVVGYYFK',
            'LIGDDHRN-',
            'CMPRYWYTY',
            'QWFWRARPF',
            'VTMPEGHHC',
            'LNYINMHVD',
            'WHV-EWKPV',
            'PIWGGFNFP',
            'PPCWVEAPY',
            'E-MWRGLIW',
            'RFGKFTCMG',
            'CGRCGSH-E',
            'T-PMVWRLV',
            'LNCPYADLD'
        ], protein_alphabet)
        stats = Statistics(align)
        expected_f1 = _slow_get_freq1(align)
        self.assertTrue(np.allclose(stats.freq1, expected_f1))

    def test_freq2_on_protein(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics
        align = Alignment([
            'WKHNAY',
            'KHRCDA',
            'LGVVGY',
            'LIGDDH',
            'CMPRYW',
            'QWFWRA',
            'VTMPEG',
            'LNYINM',
            'WHV-EW',
            'PIWGGF',
            'PPCWVE',
            'E-MWRG',
            'RFGKFT',
            'CGRCGS',
            'T-PMVW',
            'LNCPYA'
        ], protein_alphabet)
        stats = Statistics(align)
        expected_f2 = _slow_get_freq2(align)
        self.assertTrue(np.allclose(stats.freq2, expected_f2))

    def test_cmat_on_protein(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics
        align = Alignment([
            'WKHNAYD',
            'KHRCDAN',
            'LGVVGYY',
            'LIGDDHR',
            'CMPRYWY',
            'QWFWRAR',
            'VTMPEGH',
            'LNYINMH',
            'WHV-EWK',
            'PIWGGFN',
            'PPCWVEA',
            'E-MWRGL',
            'RFGKFTC',
            'CGRCGSH',
            'T-PMVWR',
            'LNCPYAD'
        ], protein_alphabet)
        stats = Statistics(align)
        expected_f1 = _slow_get_freq1(align)
        expected_f2 = _slow_get_freq2(align)
        expected_cmat = expected_f2 - np.outer(expected_f1, expected_f1)
        self.assertTrue(np.allclose(stats.cmat, expected_cmat))

    def test_freq1_on_multi_alpha(self):
        from multicov.alphabet import protein_alphabet, dna_alphabet
        from multicov.align_io import load_fasta
        from multicov.statistics import Statistics
        from os.path import join
        align = load_fasta(join('test_data', 'test_aln1.fasta'), protein_alphabet, invalid_letter_policy='uppergap')
        align2 = load_fasta(join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='uppergap')
        align.add(align2)
        stats = Statistics(align)
        expected_f1 = _slow_get_freq1(align)
        self.assertTrue(np.allclose(stats.freq1, expected_f1))

    def test_freq2_on_multi_alpha(self):
        from multicov.alphabet import protein_alphabet, dna_alphabet
        from multicov.align_io import load_fasta
        from multicov.statistics import Statistics
        from os.path import join
        align = load_fasta(join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='gap')
        align2 = load_fasta(join('test_data', 'test_aln1.fasta'), protein_alphabet, invalid_letter_policy='gap')
        align.add(align2)
        stats = Statistics(align)
        expected_f2 = _slow_get_freq2(align)
        self.assertTrue(np.allclose(stats.freq2, expected_f2))

    def test_cmat_on_multi_alpha(self):
        from multicov.alphabet import protein_alphabet, dna_alphabet, rna_alphabet
        from multicov.align_io import load_fasta
        from multicov.statistics import Statistics
        from os.path import join
        align = load_fasta(join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='gap')
        align2 = load_fasta(join('test_data', 'test_aln1.fasta'), protein_alphabet, invalid_letter_policy='gap')
        align3 = load_fasta(join('test_data', 'test_aln2.fasta'), rna_alphabet, invalid_letter_policy='uppergap')
        align.add(align2).add(align3)
        stats = Statistics(align)
        expected_f1 = _slow_get_freq1(align)
        expected_f2 = _slow_get_freq2(align)
        expected_cmat = expected_f2 - np.outer(expected_f1, expected_f1)
        self.assertTrue(np.allclose(stats.cmat, expected_cmat))

    def test_delayed_evaluation(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics
        align = Alignment([
            'WKHNA',
            'KHRCD',
            'LGVVG',
            'LIGDD',
            'CMPRY',
            'QWFWR',
            'VTMPE',
            'LNYIN',
            'WHV-E',
            'PIWGG',
            'PPCWV',
            'E-MWR',
            'RFGKF',
            'CGRCG',
            'T-PMV',
            'LNCPY'
        ], protein_alphabet)

        stats = Statistics(align)

        old_expected_f1 = _slow_get_freq1(align)
        old_expected_f2 = _slow_get_freq2(align)
        old_expected_cmat = old_expected_f2 - np.outer(old_expected_f1, old_expected_f1)

        # modify align, and test that the statistics are calculated for the modified one
        align.data[1, :] = list('CMPRY')
        align.data[10, :] = list('KHRCD')

        expected_f1 = _slow_get_freq1(align)
        expected_f2 = _slow_get_freq2(align)
        expected_cmat = expected_f2 - np.outer(expected_f1, expected_f1)

        self.assertFalse(np.allclose(expected_f1, old_expected_f1))
        self.assertFalse(np.allclose(expected_f2, old_expected_f2))
        self.assertFalse(np.allclose(expected_cmat, old_expected_cmat))

        self.assertTrue(np.allclose(stats.freq1, expected_f1))
        self.assertTrue(np.allclose(stats.freq2, expected_f2))
        self.assertTrue(np.allclose(stats.cmat, expected_cmat))

    def test_on_empty_binalign(self):
        from multicov.binary import BinaryAlignment
        from multicov.statistics import Statistics
        stats = Statistics(BinaryAlignment())
        self.assertTrue(hasattr(stats, 'freq1'))
        self.assertTrue(hasattr(stats, 'freq2'))
        self.assertTrue(hasattr(stats, 'cmat'))
        self.assertEqual(len(stats.freq1), 0)
        self.assertEqual(len(stats.freq2), 0)
        self.assertEqual(len(stats.cmat), 0)

    def test_on_protein_binalign(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics
        align = Alignment([
            'WKHNA',
            'KHRCD',
            'LGVVG',
            'LIGDD',
            'CMPRY',
            'QWFWR',
            'VTMPE',
            'LNYIN',
            'WHV-E',
            'PIWGG',
            'PPCWV',
            'E-MWR',
            'RFGKF',
            'CGRCG',
            'T-PMV',
            'LNCPY'
        ], protein_alphabet)

        stats = Statistics(align.to_binary())
        expected_f1 = _slow_get_freq1(align)
        expected_f2 = _slow_get_freq2(align)
        expected_cmat = expected_f2 - np.outer(expected_f1, expected_f1)

        self.assertTrue(np.allclose(stats.freq1, expected_f1))
        self.assertTrue(np.allclose(stats.freq2, expected_f2))
        self.assertTrue(np.allclose(stats.cmat, expected_cmat))

    def test_precompute(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics
        align = Alignment([
            'WKHNAY',
            'KHRCDA',
            'LGVVGY',
            'LIGDDH',
            'CMPRYW',
            'QWFWRA',
            'VTMPEG',
            'LNYINM',
            'WHV-EW',
            'PIWGGF',
            'PPCWVE',
            'E-MWRG',
            'RFGKFT',
            'CGRCGS',
            'T-PMVW',
            'LNCPYA'
        ], protein_alphabet)
        stats = Statistics(align, precompute=('freq1', 'freq2'))
        self.assertIsNotNone(stats._freq1)
        self.assertIsNotNone(stats._freq2)
