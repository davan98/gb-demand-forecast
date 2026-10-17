"""Local HTTP inference over caller-supplied features; no live demand feed."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from model_artifact import load_model, predict_rows

MAX_BODY_BYTES = 128 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON keys are not allowed.')
        result[key] = value
    return result


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_args):
        pass  # Avoid logging submitted feature data.

    def reply(self, status, payload):
        data = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def do_GET(self):
        if self.path != '/health':
            return self.reply(404, {'error': 'Not found.'})
        ready = self.server.model is not None
        self.reply(200 if ready else 503, {'status': 'ready' if ready else 'not_ready',
            'model_version': 1 if ready else None,
            'feature_columns': self.server.metadata['feature_columns'] if ready else [],
            'training_cutoff_utc': self.server.metadata['training_cutoff_utc'] if ready else None})

    def read_json(self):
        if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1:
            raise ValueError('A single Content-Length header is required; transfer encoding is unsupported.')
        if self.headers.get_content_type() != 'application/json':
            raise ValueError('Content-Type must be application/json.')
        length = int(self.headers['Content-Length'])
        if not 0 < length <= MAX_BODY_BYTES:
            raise ValueError('Request body must contain 1 to 131072 bytes.')
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError('Incomplete request body.')
        return json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON number.')))

    def do_POST(self):
        if self.path not in ('/predict', '/retrieve', '/ask'):
            return self.reply(404, {'error': 'Not found.'})
        try:
            payload = self.read_json()
            if self.path in ('/retrieve', '/ask'):
                if not isinstance(payload, dict) or not {'question'} <= set(payload) <= {'question', 'top_k'}:
                    raise ValueError('Body requires question and optionally top_k.')
                question, top_k = payload['question'], payload.get('top_k', 3)
                if not isinstance(question, str) or not question.strip() or len(question) > 1000:
                    raise ValueError('question must contain 1 to 1000 characters.')
                if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
                    raise ValueError('top_k must be an integer between 1 and 5.')
                from rag import retrieve, answer
                result = (retrieve if self.path == '/retrieve' else answer)(question, top_k=top_k)
                return self.reply(200, result)
            if not isinstance(payload, dict) or set(payload) != {'rows'}:
                raise ValueError('Body must contain exactly one key: rows.')
            if self.server.model is None:
                return self.reply(503, {'error': 'Model unavailable. Run the training pipeline first.'})
            values = predict_rows(self.server.model, self.server.metadata, payload['rows'])
            self.reply(200, {'predictions_mw': values, 'model_version': 1,
                'training_cutoff_utc': self.server.metadata['training_cutoff_utc']})
        except (ValueError, UnicodeDecodeError, OverflowError, TimeoutError) as error:
            self.reply(400, {'error': str(error)})
        except (RuntimeError, ImportError):
            self.reply(503, {'error': 'RAG service unavailable.'})


def create_server(host='127.0.0.1', port=8000, model_dir=None):
    server = ThreadingHTTPServer((host, port), Handler)
    server.model, server.metadata = None, None
    try:
        server.model, server.metadata = load_model(model_dir or Path(__file__).resolve().parents[1] / 'models')
    except (OSError, ValueError, KeyError, TypeError):
        pass  # Health remains 503; restarting after training loads the artifact.
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'models')
    args = parser.parse_args()
    with create_server(args.host, args.port, args.model_dir) as server:
        print(f'Listening at http://{args.host}:{server.server_port}; ready={server.model is not None}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
