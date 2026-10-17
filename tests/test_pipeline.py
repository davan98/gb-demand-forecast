"""Checks for the parts most likely to produce misleading forecasts."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pipeline


class ForecastChecks(unittest.TestCase):
    def test_recent_unavailable_observations_cannot_change_features(self):
        index = pd.date_range('2024-01-01', periods=100 * 48, freq='30min', tz='UTC')
        demand = pd.Series(np.arange(len(index), dtype=float), index=index)
        original = pipeline.features(demand).iloc[-48:]
        changed = demand.copy()
        changed.iloc[-27 * 48:] = 999999
        pd.testing.assert_frame_equal(original, pipeline.features(changed).iloc[-48:])

    def test_clock_change_days_have_continuous_utc_timestamps(self):
        for date, periods in [('31-MAR-2024', 46), ('27-OCT-2024', 50)]:
            with self.subTest(date=date), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'input.csv'
                pd.DataFrame({'SETTLEMENT_DATE': [date] * periods,
                              'SETTLEMENT_PERIOD': np.arange(1, periods + 1),
                              'ND': [20000] * periods}).to_csv(path, index=False)
                with patch.object(pipeline, 'REPORT', Path(directory)):
                    result = pipeline.prepare([path])
                self.assertEqual(len(result), periods)
                self.assertFalse(result.index.duplicated().any())
                self.assertTrue((result.index.to_series().diff().dropna() == pd.Timedelta(minutes=30)).all())

    def test_duplicate_settlement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            pd.DataFrame({'SETTLEMENT_DATE': ['01-JAN-2024'] * 2,
                          'SETTLEMENT_PERIOD': [1, 1], 'ND': [20000, 20000]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                pipeline.prepare([path])

    def test_missing_target_does_not_get_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.csv'
            pd.DataFrame({'SETTLEMENT_DATE': ['01-JAN-2024'] * 2,
                          'SETTLEMENT_PERIOD': [1, 3], 'ND': [20000, 21000]}).to_csv(path, index=False)
            with patch.object(pipeline, 'REPORT', Path(directory)):
                result = pipeline.prepare([path])
            self.assertTrue(pd.isna(result.iloc[1]))

    def test_ridge_handles_constant_features_and_intercept(self):
        x = np.column_stack([np.arange(10), np.ones(10)])
        y = np.arange(10) * 2 + 7
        model = pipeline.fit_ridge(x, y, 0.000001)
        np.testing.assert_allclose(pipeline.predict(model, x), y, atol=1e-5)

    def test_metric_units_and_values(self):
        score = pipeline.metrics(np.array([100., 200.]), np.array([110., 180.]))
        self.assertEqual(score['mae_mw'], 15)
        self.assertEqual(score['wape_percent'], 10)


if __name__ == '__main__':
    unittest.main()
