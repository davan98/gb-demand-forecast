"""Portable, pickle-free fitted ridge artifacts (schema version 1)."""
from pathlib import Path
import hashlib
import json
import numpy as np


def export_model(model, columns, training_cutoff, directory, *, alpha=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {'schema_version': 1, 'model_type': 'standardized_ridge',
                'feature_columns': list(columns), 'training_cutoff_utc': str(training_cutoff),
                'selected_alpha': alpha, 'prediction_unit': 'MW',
                'input_contract': 'Caller supplies historical-lag and calendar features; no live feed.'}
    arrays = {key: np.asarray(model[key], dtype=np.float64) for key in ('mean', 'scale', 'weights')}
    _validate(arrays, metadata)
    path = directory / 'ridge-v1.npz'
    np.savez(path, **arrays)
    metadata['artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / 'metadata-v1.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def _validate(model, metadata):
    columns = metadata.get('feature_columns')
    if (metadata.get('schema_version') != 1 or metadata.get('model_type') != 'standardized_ridge'
            or not isinstance(columns, list) or not columns
            or not all(isinstance(c, str) and c for c in columns) or len(set(columns)) != len(columns)
            or not metadata.get('training_cutoff_utc')):
        raise ValueError('Unsupported or invalid model metadata.')
    n = len(columns)
    for key, size in [('mean', n), ('scale', n), ('weights', n + 1)]:
        if model[key].shape != (size,) or not np.isfinite(model[key]).all():
            raise ValueError('Invalid model array: ' + key)
    if not (model['scale'] > 0).all():
        raise ValueError('Model scales must be positive.')


def load_model(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata-v1.json').read_text(encoding='utf-8'))
    path = directory / 'ridge-v1.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get('artifact_sha256'):
        raise ValueError('Model artifact checksum mismatch.')
    with np.load(path, allow_pickle=False) as archive:
        model = {key: archive[key].copy() for key in ('mean', 'scale', 'weights')}
    _validate(model, metadata)
    return model, metadata


def predict_rows(model, metadata, rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError('rows must contain between 1 and 100 feature objects.')
    columns = metadata['feature_columns']
    values = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(columns):
            raise ValueError('Each row must contain exactly the model feature keys.')
        for value in row.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('Feature values must be finite numbers, not booleans.')
        try:
            vector = np.asarray([row[c] for c in columns], dtype=np.float64)
        except (OverflowError, ValueError):
            raise ValueError('Feature values must be finite numbers.') from None
        if not np.isfinite(vector).all():
            raise ValueError('Feature values must be finite numbers.')
        values.append(vector)
    with np.errstate(over='ignore', invalid='ignore'):
        x = np.asarray(values)
        predictions = np.column_stack([np.ones(len(x)), (x - model['mean']) / model['scale']]) @ model['weights']
    if not np.isfinite(predictions).all():
        raise ValueError('Feature magnitudes produce non-finite predictions.')
    return predictions.tolist()
