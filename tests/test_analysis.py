import json
import tempfile
import unittest
from pathlib import Path

from analysis import HEADERS, load_data, run


class LaptopModelTests(unittest.TestCase):
    def test_source_schema_and_target(self):
        data = load_data()
        self.assertEqual(data.shape, (238, 12))
        self.assertEqual(list(data.columns), HEADERS)
        self.assertFalse(data["Price"].isna().any())
        self.assertTrue((data["Price"] > 0).all())

    def test_deterministic_split_and_metrics(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, b = run(Path(first)), run(Path(second))
            self.assertEqual(a, b)
            self.assertEqual(a, json.loads((Path(first) / "metrics.json").read_text()))
            self.assertEqual(a["split"], {"train": 142, "calibration": 48, "test": 48, "seed": 42})
            self.assertLess(a["ridge_test"]["rmse_usd"], a["mean_baseline_test"]["rmse_usd"])


if __name__ == "__main__":
    unittest.main()
