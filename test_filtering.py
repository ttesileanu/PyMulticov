import unittest

import numpy as np


class TestSearch(unittest.TestCase):
    def test_raise_on_empty_align(self):
        from alignment import Alignment
        from filtering import search

        with self.assertRaises(ValueError):
            search(Alignment(), 'ABC')

    def test_raise_on_empty_seq(self):
        from alignment import Alignment
        from alphabet import protein_alphabet
        from filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        with self.assertRaises(ValueError):
            search(align, '')

    def test_search_string(self):
        from alignment import Alignment
        from alphabet import protein_alphabet
        from filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        self.assertEqual(search(align, 'VGGTEAQ'), 1)

    def test_search_list(self):
        from alignment import Alignment
        from alphabet import protein_alphabet
        from filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        self.assertEqual(search(align, ['I', 'G', 'G', 'K', 'D', 'T']), 2)

    def test_search_approx(self):
        from alignment import Alignment
        from alphabet import protein_alphabet
        from filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        self.assertEqual(search(align, 'IGGYTCQ'), 0)

    def test_move_to_top(self):
        from alignment import Alignment
        from alphabet import protein_alphabet
        from filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        search(align, ['I', 'G', 'G', 'K', 'D', 'T'], move_to_top=True)
        self.assertTrue(np.array_equal(align.data, np.asmatrix([
            ['I', 'G', 'G', '-', 'K', 'D', 'T', '-'],
            ['-', 'V', 'G', 'G', 'T', 'E', 'A', 'Q'],
            ['I', 'V', 'G', 'G', 'Y', 'T', 'C', 'Q']
        ])))

    def test_move_to_top_but_return_old_idx(self):
        from alignment import Alignment
        from alphabet import protein_alphabet
        from filtering import search
        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], alphabet=protein_alphabet)

        self.assertEqual(search(align, ['I', 'G', 'G', 'K', 'D', 'T'], move_to_top=True), 2)

    def test_search_dna(self):
        from alignment import Alignment
        from alphabet import dna_alphabet
        from filtering import search
        align = Alignment(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)

        self.assertEqual(search(align, 'AAGG'), 2)


class TestFilterRows(unittest.TestCase):
    def test_on_empty(self):
        from alignment import Alignment
        from filtering import filter_rows
        align1 =  Alignment()
        align2 = filter_rows(Alignment())
        self.assertEqual(align1, align2)

    def test_on_protein(self):
        from alignment import Alignment
        from alphabet import protein_alphabet
        from filtering import filter_rows
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
        from alignment import Alignment
        from filtering import filter_rows
        from alphabet import protein_alphabet, dna_alphabet

        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.add(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align_clean = filter_rows(align, 0.2)

        expected = Alignment(['IVGGYTCQ', '-VGGTEAQ'], protein_alphabet).add(['ATACAT', 'GATACA'], dna_alphabet)

        self.assertEqual(align_clean, expected)

    def test_returns_copy(self):
        from alignment import Alignment
        from filtering import filter_rows
        from alphabet import protein_alphabet, dna_alphabet

        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.add(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align_clean = filter_rows(align, 0.2)

        self.assertEqual(len(align_clean), 2)
        self.assertNotEqual(align_clean, align)

    def test_returns_copy_even_when_unchanged(self):
        from alignment import Alignment
        from filtering import filter_rows
        from alphabet import protein_alphabet, dna_alphabet

        align = Alignment(['IVGGYTCQ', '-VGGTEAQ', 'IGG-KDT-'], protein_alphabet)
        align.add(['ATACAT', 'GATACA', 'AA--GG'], dna_alphabet)
        align_clean = filter_rows(align, 0.9)

        self.assertEqual(len(align_clean), 3)
        self.assertIsNot(align_clean, align)
