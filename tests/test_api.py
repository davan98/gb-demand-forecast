"""Artifact round-trip and real HTTP contract checks, with synthetic fitted data."""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from api import create_server, MAX_BODY_BYTES
from model_artifact import export_model, load_model, predict_rows
from pipeline import fit_ridge, predict


class APIChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.columns = ['lag_28d', 'hour_sin_1']
        self.x = np.array([[1., 0.], [2., 1.], [3., -1.], [4., .5]])
        self.model = fit_ridge(self.x, np.array([10., 14., 18., 20.]), 1.)
        export_model(self.model, self.columns, '2023-12-31T23:30:00+00:00', self.directory.name, alpha=1.)
        self.server = create_server(port=0, model_dir=self.directory.name)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        if body is not None and not isinstance(body, (str, bytes)):
            body = json.dumps(body)
        connection.request(method, path, body=body, headers=headers or {'Content-Type': 'application/json'})
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_roundtrip_and_http_predictions_equal_training_implementation(self):
        loaded, metadata = load_model(self.directory.name)
        rows = [dict(zip(self.columns, row)) for row in self.x.tolist()]
        expected = predict(self.model, self.x)
        np.testing.assert_allclose(predict_rows(loaded, metadata, rows), expected, rtol=0, atol=1e-12)
        status, result = self.request('POST', '/predict', {'rows': rows})
        self.assertEqual(status, 200)
        np.testing.assert_allclose(result['predictions_mw'], expected, rtol=0, atol=1e-12)
        self.assertEqual(self.request('GET', '/health')[1]['feature_columns'], self.columns)

    def test_bad_rows_rejected(self):
        for rows in [[], [{}], [{'lag_28d': 1}], [{'lag_28d': True, 'hour_sin_1': 1}],
                     [{'lag_28d': '1', 'hour_sin_1': 1}], [{'lag_28d': 1e309, 'hour_sin_1': 1}],
                     [{'lag_28d': 1, 'hour_sin_1': 1, 'extra': 1}],
                     [{'lag_28d': 1, 'hour_sin_1': 1}] * 101]:
            with self.subTest(rows=str(rows)[:80]):
                self.assertEqual(self.request('POST', '/predict', {'rows': rows})[0], 400)

    def test_json_and_body_contract(self):
        for raw in ['{', '{"rows": [], "rows": []}', '{"rows":[{"lag_28d":NaN,"hour_sin_1":0}]}',
                    '{"extra": []}', 'x' * (MAX_BODY_BYTES + 1)]:
            self.assertEqual(self.request('POST', '/predict', raw)[0], 400)
        self.assertEqual(self.request('POST', '/predict', '{}', {'Content-Type': 'text/plain'})[0], 400)
        self.assertEqual(self.request('GET', '/unknown')[0], 404)

    def test_missing_model_is_not_ready(self):
        with tempfile.TemporaryDirectory() as missing:
            unloaded = create_server(port=0, model_dir=missing)
            try:
                self.assertIsNone(unloaded.model)
            finally:
                unloaded.server_close()
        self.server.model = None
        self.assertEqual(self.request('GET', '/health')[0], 503)
        self.assertEqual(self.request('POST', '/predict', {'rows': [{}]})[0], 503)

    def test_tampered_artifact_rejected(self):
        path = Path(self.directory.name) / 'ridge-v1.npz'
        path.write_bytes(path.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_model(self.directory.name)

    def test_rag_input_validation_and_dispatch(self):
        for payload in [{'question': ''}, {'question': 'x' * 1001}, {'question': 'q', 'top_k': True},
                        {'question': 'q', 'top_k': 6}, {'question': 'q', 'file': '/tmp/private'}]:
            self.assertEqual(self.request('POST', '/ask', payload)[0], 400)
        import types
        fake = types.ModuleType('rag')
        fake.retrieve = lambda question, top_k: {'question': question, 'top_k': top_k, 'mode': 'retrieval'}
        fake.answer = lambda question, top_k: {'question': question, 'mode': 'retrieval_only', 'answer': None}
        with patch.dict(sys.modules, {'rag': fake}):
            self.assertEqual(self.request('POST', '/retrieve', {'question': 'How?', 'top_k': 2})[1]['top_k'], 2)
            self.assertEqual(self.request('POST', '/ask', {'question': 'How?'})[1]['mode'], 'retrieval_only')
            fake.answer = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError('private details'))
            self.assertEqual(self.request('POST', '/ask', {'question': 'How?'})[1], {'error': 'RAG service unavailable.'})


if __name__ == '__main__':
    unittest.main()
