import unittest
from dga.datasets.domain_dataset import DomainDataset


class DomainDatasetTest(unittest.TestCase):
    def setUp(self):
        self.data = 'dga/test/data/test_dataset.csv'
        self.domain_ds = DomainDataset(self.data)

    def test_csv_get_data_length(self):
        self.assertEqual(self.domain_ds.__len__(), 199)
        self.assertEqual(self.domain_ds.data_len, 199)

    def test_csv_read_label(self):
        self.assertEqual(self.domain_ds.__getitem__(50)[1], 0)
        self.assertEqual(self.domain_ds.__getitem__(150)[1], 1)

    def test_csv_read_data(self):
        self.assertEqual(
            self.domain_ds.data_df.iloc[5, 0],
            's.w.org')
        self.assertEqual(
            self.domain_ds.data_df.iloc[132, 0],
            'airportconference.com')
