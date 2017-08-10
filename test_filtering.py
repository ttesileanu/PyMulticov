import unittest

import numpy as np


class TestSearch(unittest.TestCase):
    def test_raise_on_empty_align(self):
        from multicov.alignment import Alignment
        from multicov.filtering import search

        with self.assertRaises(ValueError):
            search(Alignment(), 'ABC')

    def test_raise_on_empty_seq(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        with self.assertRaises(ValueError):
            search(align, '')

    def test_search_string(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        self.assertEqual(search(align, 'VGGTEAQ'), 1)

    def test_search_list(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        self.assertEqual(search(align, ['I', 'G', 'G', 'K', 'D', 'T']), 2)

    def test_search_approx(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        self.assertEqual(search(align, 'IGGYTCQ'), 0)

    def test_move_to_top(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        search(align, ['I', 'G', 'G', 'K', 'D', 'T'], move_to_top=True)
        self.assertTrue(np.array_equal(align.data, np.asmatrix([
            ['I', 'G', 'G', '-', 'K', 'D', 'T', '-'],
            ['-', 'V', 'G', 'G', 'T', 'E', 'A', 'Q'],
            ['I', 'V', 'G', 'G', 'Y', 'T', 'C', 'Q']
        ])))

    def test_move_to_top_but_return_old_idx(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        self.assertEqual(search(align, ['I', 'G', 'G', 'K', 'D', 'T'], move_to_top=True), 2)

    def test_search_dna(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        from multicov.filtering import search
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)

        self.assertEqual(search(align, 'AAGG'), 2)


class TestFilterRows(unittest.TestCase):
    def test_on_empty(self):
        from multicov.alignment import Alignment
        from multicov.filtering import filter_rows
        align1 = Alignment()
        align2 = filter_rows(Alignment())
        self.assertEqual(align1, align2)

    def test_on_protein(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.filtering import filter_rows
        threshold = 1/21
        align = Alignment([
            'WKHNAYDMLSSDCQFESSHKHTQVCSAGMGYCAKPNNWGYW-LIVKMMW-CDYKQKLIYIAPLN',
            'KHRCDANC-MLAN-SVIKYTHSACALIWTWNS-KIIRYFFVGAWFKEHFDSVPTAQACVCDSTP',
            'LGVVGYYFKPCT-EVPSYSRFNVFHRIFPYLVYRVEE-NHTGHHVQ-KIVRNQYELRSIFDEHG',
            'LIGDDHRN-LALCPS-T-GTTCCNWKWRSEWTMHSDTNCNPVAE--SYSKRCNDIGYITWINYA',
            'CMPRYWYTYQYDCIFGWRFYSVYWPCLDDMFWQPYVDSMELF-NPMVATEWIMENCQGWG-N-K',
            'QWFWRARPFE--FSC-C-PGP-GWVNLIDWMSCNKAMETLMRPYCNPYLKIQLPRSKNLLDDDG',
            'VTMPEGHHCPAM-PLDLNGQR-KMWGSDFKKEDCKGYPEKFDCENLIDMDICLSLNTRPED-QR',
            'LNYINMHVD-IGP-PCPQYDL--KFKCMYW-GQIEDV-NMQ-WKK-RTMDAVEQIVSMYHMSVE',
            'WHV-EWKPVLC-PHWQFYM-VITEYVAMFQWCPPKGMASPKKGNLPRMFQSAKAIGAHRSDM-Y',
            'PIWGGFNFPWID-GSQRQQR-EVTTGCDDFEHKYNPYLVPG-WEFGKYSNCWT-RCWRVNHDTV',
            'PPCWVEAPYKPMGMWN-GRKV-NVAVWHHVIVL-DMYGLHLLRDWTMVKNAAHIFSHNMEMSNI',
            'E-MWRGLIWSKGAY-YQNDNGTFNWPKQKHP-ARCSF-PTVNKDQNPGP-MVQMREFKSQQGQQ',
            'RFGKFTCMGFRWKEYFTKQ-NPYKYRGIVHVKVQMIYSANGNLDWIDIPMIIRLKCPFGTRVTQ',
            'CGRCGSH-EWL-NIMRNCKFIFWWRPTNAAHIWCARHESPKAD-QIAMTYRML-LDAHIIIVR-',
            'T-PMVWRLVWYDHGCDPWMLIV-PIEPCVVKKPQYKDMERFSPDIKCHYLHDKDDGFWGSDKYI',
            'LNCPYADLDGL-NPQR-FVVS-RCMRDGFRAVVRVSPDDLS-MWCKAGA-NTTV-DNRH-IVQW'
        ], protein_alphabet)
        align_clean = filter_rows(align, max_gaps=threshold)

        gap_fraction = np.mean(align.data == '-', axis=1)
        gap_fraction_clean = np.mean(align_clean.data == '-', axis=1)

        self.assertLess(len(align_clean), len(align))
        self.assertLessEqual(np.max(gap_fraction_clean), threshold)
        self.assertEqual(np.sum(gap_fraction <= threshold), len(align_clean))

    def test_on_multi_alpha(self):
        from multicov.alignment import Alignment
        from multicov.filtering import filter_rows
        from multicov.alphabet import protein_alphabet, dna_alphabet

        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.add(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align_clean = filter_rows(align, 0.2)

        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ'], protein_alphabet).add(['ATACAT', 'GATACA'], dna_alphabet)

        self.assertEqual(align_clean, expected)

    def test_returns_copy(self):
        from multicov.alignment import Alignment
        from multicov.filtering import filter_rows
        from multicov.alphabet import protein_alphabet, dna_alphabet

        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.add(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align_clean = filter_rows(align, 0.2)

        self.assertEqual(len(align_clean), 2)
        self.assertNotEqual(align_clean, align)

    def test_returns_copy_even_when_unchanged(self):
        from multicov.alignment import Alignment
        from multicov.filtering import filter_rows
        from multicov.alphabet import protein_alphabet, dna_alphabet

        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.add(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align_clean = filter_rows(align, 0.9)

        self.assertEqual(len(align_clean), 3)
        self.assertIsNot(align_clean, align)


class TestAlignToSequence(unittest.TestCase):
    def test_on_empty(self):
        from multicov.alignment import Alignment
        from multicov.filtering import align_to_sequence
        align = Alignment()
        align_to_sequence(align, 'ACC')
        self.assertEqual(align, Alignment())

    def test_dna(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        from multicov.filtering import align_to_sequence
        align = Alignment(['AA-T', 'ACGT', '--GG', 'A-GG', 'GGC-'], dna_alphabet)
        align_to_sequence(align, 'AAT')
        self.assertEqual(len(align.reference.seqs), 1)
        self.assertSequenceEqual(list(align.reference.seqs[0]), [0, 1, None, 2])

    def test_with_list(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet
        from multicov.filtering import align_to_sequence
        align = Alignment(['AA-U', 'ACGU', '--GG', 'A-GG', 'GGC-'], rna_alphabet)
        align_to_sequence(align, ['A', 'A', 'U'])
        self.assertEqual(len(align.reference.seqs), 1)
        self.assertSequenceEqual(list(align.reference.seqs[0]), [0, 1, None, 2])

    def test_with_truncate(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet
        from multicov.filtering import align_to_sequence
        align = Alignment(['AA-U', 'ACGU', '--GG', 'A-GG', 'GGC-'], rna_alphabet)
        align_to_sequence(align, 'AGG', truncate=True)
        expected = Alignment(['A-U', 'AGU', '-GG', 'AGG', 'GC-'], rna_alphabet)
        self.assertEqual(align, expected)

    def test_with_longer_refseq(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet
        from multicov.filtering import align_to_sequence
        align = Alignment(['AA-U', 'ACGU', '--GG', 'A-GG', 'GGC-'], rna_alphabet)
        align_to_sequence(align, 'GUAACCGUU')
        self.assertEqual(len(align.reference.seqs), 1)
        self.assertSequenceEqual(list(align.reference.seqs[0]), [3, 5, 6, 8])

    def test_with_imperfect_match(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.filtering import align_to_sequence
        align = Alignment(['-AWGGH', 'D-GG-A', 'WWGYPD', 'W--IIK', '--FDGH'], protein_alphabet)
        align_to_sequence(align, 'AWCWGYPPCY')
        self.assertEqual(len(align.reference.seqs), 1)
        self.assertSequenceEqual(list(align.reference.seqs[0]), [1, 3, 4, 5, 6, 7])

    def test_details_index(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet
        from multicov.filtering import align_to_sequence
        align = Alignment(['AG-U', 'ACGU', '--GG', 'A-GG', 'GGC-'], rna_alphabet)
        details = align_to_sequence(align, 'GUAACCGUU')
        self.assertEqual(details['idx'], 1)

    def test_details_align_accuracy(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import protein_alphabet
        from multicov.filtering import align_to_sequence
        align = Alignment(['-AWGGH', 'D-GG-A', 'WWGYPD', 'W--IIK', '--FDGH'], protein_alphabet)
        details = align_to_sequence(align, 'AWCWGYPPCY')
        self.assertAlmostEqual(details['align_accuracy'], 5/6)

    def test_ref_idx_names(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import dna_alphabet
        from multicov.filtering import align_to_sequence
        align = Alignment(['AA-G', 'ACGT', '--GG', 'A-GG', 'GGC-'], dna_alphabet)
        align_to_sequence(align, 'GTAACCGTT', ref_idx_names=['1', '2', '3', '3a', '4', '5', 8, 9, 9.5])
        self.assertEqual(len(align.reference.seqs), 1)
        self.assertSequenceEqual(list(align.reference.seqs[0]), ['3a', '5', 8, 9.5])

    def test_force_idx(self):
        from multicov.alignment import Alignment
        from multicov.alphabet import rna_alphabet
        from multicov.filtering import align_to_sequence
        align = Alignment(['AA-U', 'ACGU', '--GG', 'A-GG', 'GGC-'], rna_alphabet)
        align_to_sequence(align, 'GUAACCGUU', force_idx=0)
        self.assertEqual(len(align.reference.seqs), 1)
        self.assertSequenceEqual(list(align.reference.seqs[0]), [2, 3, None, 8])
