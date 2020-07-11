import unittest
from dga.datasets.domain_dataset import DomainDataset


class DomainDatasetTest(unittest.TestCase):
    def setUp(self):
        self.data = 'football.csv'

    def test_csv_read_data_headers(self):
        self.assertEqual(
            read_data(self.data)[0],
            ['Team', 'Games', 'Wins', 'Losses', 'Draws', 'Goals', 'Goals Allowed', 'Points']
            )

    def test_csv_read_data_team_name(self):
        self.assertEqual(read_data(self.data)[1][0], 'Arsenal')

    def test_csv_read_data_points(self):
        self.assertEqual(read_data(self.data)[1][7], '87')


if __name__ == '__main__':
    unittest.main()