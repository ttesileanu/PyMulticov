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


class TestConstructor(unittest.TestCase):
    def test_on_empty(self):
        from multicov.alignment import Alignment
        from multicov.statistics import Statistics
        stats = Statistics(Alignment())
        self.assertTrue(hasattr(stats, 'freq1'))
        self.assertTrue(hasattr(stats, 'freq2'))
        self.assertTrue(hasattr(stats, 'cmat'))
        self.assertTrue(hasattr(stats, 'alphabets'))
        self.assertTrue(hasattr(stats, 'reference'))
        self.assertTrue(hasattr(stats, 'annotations'))
        self.assertEqual(len(stats.freq1), 0)
        self.assertEqual(len(stats.freq2), 0)
        self.assertEqual(len(stats.cmat), 0)
        self.assertEqual(len(stats.alphabets), 0)
        self.assertEqual(len(stats.reference), 0)
        self.assertEqual(len(stats.annotations), 0)

    def test_on_empty_binalign(self):
        from multicov.binary import BinaryAlignment
        from multicov.statistics import Statistics
        stats = Statistics(BinaryAlignment())
        self.assertTrue(hasattr(stats, 'freq1'))
        self.assertTrue(hasattr(stats, 'freq2'))
        self.assertTrue(hasattr(stats, 'cmat'))
        self.assertTrue(hasattr(stats, 'alphabets'))
        self.assertTrue(hasattr(stats, 'reference'))
        self.assertTrue(hasattr(stats, 'annotations'))
        self.assertEqual(len(stats.freq1), 0)
        self.assertEqual(len(stats.freq2), 0)
        self.assertEqual(len(stats.cmat), 0)
        self.assertEqual(len(stats.alphabets), 0)
        self.assertEqual(len(stats.reference), 0)
        self.assertEqual(len(stats.annotations), 0)

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
        self.assertIsNone(stats._freq1)
        self.assertIsNone(stats._freq2)
        self.assertIsNone(stats._cmat)

    def test_copy_alpha_annots_refmap_by_ref(self):
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
        self.assertIs(stats.alphabets, align.alphabets)
        self.assertIs(stats.reference, align.reference)
        self.assertIs(stats.annotations, align.annotations)

    def test_copy_alpha_annots_refmap_by_ref_for_binalign(self):
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
        bin_align = align.to_binary()

        stats = Statistics(bin_align)
        self.assertIs(stats.alphabets, bin_align.alphabets)
        self.assertIs(stats.reference, bin_align.reference)
        self.assertIs(stats.annotations, bin_align.annotations)


class TestEvaluation(unittest.TestCase):
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

    def test_against_lucy(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics
        from scipy.io import loadmat
        import os.path
        lucy = loadmat(os.path.join('test_data', 'lucy_dca_pdz_small.mat'),
                       squeeze_me=True)
        align = Alignment(lucy['alignment']['data'][()], protein_alphabet)
        align.update_sequence_weights(0.7)
        stats = Statistics(align, regularization_amount=0.5)
        self.assertTrue(np.allclose(stats.cmat, lucy['DCAmat']))

    def test_against_lucy_noseqw(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics
        from scipy.io import loadmat
        import os.path
        lucy = loadmat(os.path.join('test_data', 'lucy_dca_pdz_small.mat'),
                       squeeze_me=True)
        align = Alignment(lucy['alignment']['data'][()], protein_alphabet)
        stats = Statistics(align, regularization_amount=0.5)
        self.assertTrue(np.allclose(stats.cmat, lucy['DCAmat_noseqw']))


class TestGetItem(unittest.TestCase):
    def test_get_str_goes_to_annotations(self):
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
        self.assertIs(stats['seqw'], align['seqw'])


class TestPseudocount(unittest.TestCase):
    def test_protein(self):
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

        alpha = 0.3

        stats1 = Statistics(align)
        stats2 = Statistics(align, regularization_amount=alpha, regularizer='pseudocount')

        expected_f1 = (1 - alpha)*stats1.freq1 + alpha/21.0
        expected_f2 = (1 - alpha)*stats1.freq2 + alpha/21.0**2

        # block-diagonal needs correction because those variables are not independent
        for i in range(align.get_width()):
            idxs = slice(20 * i, 20 * (i + 1))
            # noinspection PyUnresolvedReferences
            expected_f2[idxs, idxs] = np.diag(expected_f1[idxs])

        expected_cmat = expected_f2 - np.outer(expected_f1, expected_f1)

        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(expected_f1, stats2.freq1))
        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(expected_f2, stats2.freq2))
        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(expected_cmat, stats2.cmat))

    def test_multi_alpha(self):
        from multicov.alphabet import protein_alphabet, dna_alphabet
        from multicov.align_io import load_fasta
        from multicov.statistics import Statistics
        from os.path import join
        align = load_fasta(join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='gap')
        align2 = load_fasta(join('test_data', 'test_aln1.fasta'), protein_alphabet, invalid_letter_policy='gap')
        align.add(align2)

        alpha = 0.6

        stats1 = Statistics(align)
        stats2 = Statistics(align, regularization_amount=alpha, regularizer='pseudocount')

        bkg_freq1 = np.hstack(np.ones(width * alphabet.size(no_gap=True)) / alphabet.size()
                              for alphabet, width in align.alphabets)
        bkg_freq2 = np.outer(bkg_freq1, bkg_freq1)

        freq1 = (1 - alpha) * stats1.freq1 + alpha * bkg_freq1
        freq2 = (1 - alpha) * stats1.freq2 + alpha * bkg_freq2

        n_letts = np.hstack(width * [alphabet.size(no_gap=True)] for alphabet, width in align.alphabets)
        idxs0 = np.hstack(([0], np.cumsum(n_letts)[:-1]))
        for idx0, n_lett in zip(idxs0, n_letts):
            idxs = slice(idx0, idx0 + n_lett)
            # noinspection PyUnresolvedReferences
            freq2[idxs, idxs] = np.diag(freq1[idxs])

        cmat = freq2 - np.outer(freq1, freq1)

        self.assertTrue(np.allclose(freq1, stats2.freq1))
        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(freq2, stats2.freq2))
        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(cmat, stats2.cmat))


class TestMaxentConstruction(unittest.TestCase):
    def test_on_empty(self):
        from multicov.alignment import Alignment
        from multicov.statistics import Statistics, MaxentModel
        stats = Statistics(Alignment())
        maxent = MaxentModel(stats)
        self.assertTrue(hasattr(maxent, 'stats'))
        self.assertTrue(hasattr(maxent, 'alphabets'))
        self.assertTrue(hasattr(maxent, 'annotations'))
        self.assertTrue(hasattr(maxent, 'reference'))

        self.assertIs(maxent.stats, stats)
        self.assertIs(maxent.alphabets, stats.alphabets)
        self.assertIs(maxent.annotations, stats.annotations)
        self.assertIs(maxent.reference, stats.reference)
        self.assertEqual(np.size(maxent.couplings), 0)

    def test_on_protein(self):
        from multicov.alignment import Alignment
        from multicov.binary import binary_index_map
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics, MaxentModel
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
        stats = Statistics(align, regularization_amount=0.1)
        maxent = MaxentModel(stats)
        invC = -np.linalg.inv(stats.cmat)
        bin_map = binary_index_map(stats)
        for crt_range in bin_map:
            invC[crt_range[0]:crt_range[1], crt_range[0]:crt_range[1]] = 0
        self.assertTrue(np.allclose(invC, maxent.couplings - np.diag(np.diag(maxent.couplings))))

    def test_multi_alpha_shape_and_symmetry_of_couplings(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet, rna_alphabet
        from multicov.align_io import load_fasta
        from multicov.statistics import Statistics, MaxentModel
        from os.path import join
        align1 = load_fasta(join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='gap')
        align2 = load_fasta(join('test_data', 'test_aln1.fasta'), protein_alphabet, invalid_letter_policy='gap')
        align3 = load_fasta(join('test_data', 'test_aln2.fasta'), rna_alphabet, invalid_letter_policy='uppergap')
        align = Alignment(align1)
        align.add(align2).add(align3)
        stats = Statistics(align, regularization_amount=0.5)
        maxent = MaxentModel(stats)
        self.assertLess(np.max(np.abs(maxent.couplings - maxent.couplings.T)), 1e-10)
        self.assertSequenceEqual(np.shape(maxent.couplings), 2*[4*(align1.get_width() + align3.get_width()) +
                                                              20*align2.get_width()])

    def test_multi_alpha_diagonalness_of_blockdiagonal_blocks(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet, rna_alphabet
        from multicov.align_io import load_fasta
        from multicov.binary import binary_index_map
        from multicov.statistics import Statistics, MaxentModel
        from os.path import join
        align1 = load_fasta(join('test_data', 'test_aln1.fasta'), protein_alphabet, invalid_letter_policy='gap')
        align2 = load_fasta(join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='gap')
        align3 = load_fasta(join('test_data', 'test_aln2.fasta'), rna_alphabet, invalid_letter_policy='gap')
        align = Alignment(align1)
        align.add(align2).add(align3)
        stats = Statistics(align, regularization_amount=0.5)
        maxent = MaxentModel(stats)
        bin_map = binary_index_map(stats)
        for crt_range in bin_map:
            crt_slice = slice(crt_range[0], crt_range[1])
            crt_block = maxent.couplings[crt_slice, crt_slice]
            self.assertLess(np.max(np.abs(crt_block - np.diag(np.diag(crt_block)))), 1e-10)

    def test_against_matlab_example(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics, MaxentModel
        from scipy.io import loadmat
        import os.path
        matlab = loadmat(os.path.join('test_data', 'maxent_sample.mat'),
                         squeeze_me=True)
        align = Alignment(matlab['alignment']['data'][()], protein_alphabet)
        stats = Statistics(align, regularization_amount=0.5)
        maxent = MaxentModel(stats)
        self.assertTrue(np.allclose(stats.cmat, matlab['dca']['cmat'][()]))
        self.assertTrue(np.allclose(maxent.couplings, matlab['params_nogap_nofc_nodiagtrick']['couplings'][()]))


class TestMaxentEnergyEvaluation(unittest.TestCase):
    def test_against_matlab_example(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics, MaxentModel
        from scipy.io import loadmat
        import os.path
        matlab = loadmat(os.path.join('test_data', 'maxent_sample.mat'),
                         squeeze_me=True)
        align = Alignment(matlab['alignment']['data'][()], protein_alphabet)
        stats = Statistics(align, regularization_amount=0.5)
        maxent = MaxentModel(stats)
        energies = maxent.score(align)
        self.assertTrue(np.allclose(energies, matlab['energies'][()]))

    def test_gap_gauge(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet, dna_alphabet, rna_alphabet
        from multicov.align_io import load_fasta
        from multicov.statistics import Statistics, MaxentModel
        from os.path import join
        align1 = load_fasta(join('test_data', 'test_aln2.fasta'), dna_alphabet, invalid_letter_policy='gap')
        align2 = load_fasta(join('test_data', 'test_aln1.fasta'), protein_alphabet, invalid_letter_policy='gap')
        align3 = load_fasta(join('test_data', 'test_aln2.fasta'), rna_alphabet, invalid_letter_policy='uppergap')
        align = Alignment(align1)
        align.add(align2).add(align3)
        stats = Statistics(align, regularization_amount=0.5)
        maxent = MaxentModel(stats)
        energies = maxent.score([align.get_width()*'-'])
        self.assertLess(np.max(np.abs(energies)), 1e-10)

    def test_with_list_of_seqs(self):
        from multicov.alignment import Alignment
        from multicov.binary import binary_index_map
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics, MaxentModel
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
        stats = Statistics(align, regularization_amount=0.1)
        maxent = MaxentModel(stats)
        seqs = ['WHVDYA', 'PP-FR-']
        energies = maxent.score(seqs)
        seq_align = Alignment(seqs, protein_alphabet)
        energies0 = maxent.score(seq_align)
        self.assertTrue(np.allclose(energies, energies0))

    def test_with_matrix(self):
        from multicov.alignment import Alignment
        from multicov.binary import binary_index_map
        from multicov.alphabet import protein_alphabet
        from multicov.statistics import Statistics, MaxentModel
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
        stats = Statistics(align, regularization_amount=0.1)
        maxent = MaxentModel(stats)
        energies = maxent.score(align.data)
        energies0 = maxent.score(align)
        self.assertTrue(np.allclose(energies, energies0))
