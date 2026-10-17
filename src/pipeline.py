"""Download NESO data and evaluate a reproducible demand forecast.

Run from the repository root: python src/pipeline.py
Only the Python standard library, NumPy, pandas and Matplotlib are needed.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_artifact import export_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'
REPORT = ROOT / 'reports'
YEARS = (2022, 2023, 2024, 2025)
API = 'https://api.neso.energy/api/3/action/package_show?id=historic-demand-data'


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as response:
        package = json.load(response)['result']
    manifest = []
    for year in YEARS:
        resource = next(r for r in package['resources'] if r['name'] == f'Historic Demand Data {year}')
        path = DATA / f'demand_{year}.csv'
        if not path.exists():
            with urllib.request.urlopen(resource['url'], timeout=90) as response:
                path.write_bytes(response.read())
        manifest.append({'year': year, 'url': resource['url'], 'resource_id': resource['id'],
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'bytes': path.stat().st_size})
    (REPORT / 'data_manifest.json').write_text(json.dumps({
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'files': manifest
    }, indent=2), encoding='utf-8')
    return [DATA / f'demand_{y}.csv' for y in YEARS]


def prepare(paths):
    frames = [pd.read_csv(path, usecols=['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND']) for path in paths]
    raw = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(raw['SETTLEMENT_DATE'], format='mixed', dayfirst=True).dt.normalize()
    periods = pd.to_numeric(raw['SETTLEMENT_PERIOD'], errors='raise')
    if not periods.between(1, 50).all():
        raise ValueError('Settlement periods must lie between 1 and 50.')
    # Period numbers count elapsed half-hours from UK local midnight. Adding
    # elapsed time in UTC preserves both spring and autumn clock-change days.
    starts = dates.dt.tz_localize('Europe/London').dt.tz_convert('UTC')
    timestamps = starts + pd.to_timedelta((periods - 1) * 30, unit='min')
    if timestamps.duplicated().any():
        raise ValueError('Duplicate UTC settlement timestamps; inspect the source.')
    demand = pd.Series(pd.to_numeric(raw['ND'], errors='coerce').to_numpy(), index=pd.DatetimeIndex(timestamps), name='demand_mw').sort_index()
    nonpositive = int((demand <= 0).sum())
    demand = demand.mask(demand <= 0)
    complete = pd.date_range(demand.index.min(), demand.index.max(), freq='30min')
    missing_slots = len(complete.difference(demand.index))
    demand = demand.reindex(complete)
    counts = raw.groupby(dates).size()
    audit = {'source_rows': len(raw), 'missing_timestamps': missing_slots,
             'missing_or_invalid_demand': int(demand.isna().sum()),
             'nonpositive_demand': nonpositive,
             'days_by_period_count': {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
             'start_utc': str(demand.index.min()), 'end_utc': str(demand.index.max())}
    (REPORT / 'data_quality.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return demand


def features(demand):
    local = demand.index.tz_convert('Europe/London')
    frame = pd.DataFrame(index=demand.index)
    # NESO's historic feed is published 21 days in arrears. A 28-day minimum
    # lag leaves a buffer; a next-day lag would assume unavailable public data.
    for days in (28, 35, 42, 49):
        frame[f'lag_{days}d'] = demand.shift(days * 48)
    frame['recent_available_mean'] = demand.shift(28 * 48).rolling(48).mean()
    frame['recent_available_std'] = demand.shift(28 * 48).rolling(48).std()
    hour = local.hour + local.minute / 60
    for k in range(1, 5):
        for name, fn in [('sin', np.sin), ('cos', np.cos)]:
            frame[f'hour_{name}_{k}'] = fn(2 * np.pi * k * hour / 24)
    for k in (1, 2):
        frame[f'year_sin_{k}'] = np.sin(2 * np.pi * k * local.dayofyear / 365.25)
        frame[f'year_cos_{k}'] = np.cos(2 * np.pi * k * local.dayofyear / 365.25)
    for day in range(7):
        frame[f'weekday_{day}'] = (local.dayofweek == day).astype(float)
        frame[f'weekday_hour_sin_{day}'] = frame[f'weekday_{day}'] * frame['hour_sin_1']
        frame[f'weekday_hour_cos_{day}'] = frame[f'weekday_{day}'] * frame['hour_cos_1']
    frame['is_dst'] = [float(t.dst().total_seconds() != 0) for t in local]
    return frame


def fit_ridge(x, y, alpha):
    # Fit all preprocessing on the training split only.
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-9, 1, scale)
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(z.shape[1]) * alpha
    penalty[0, 0] = 0
    weights = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return {'mean': mean, 'scale': scale, 'weights': weights}


def predict(model, x):
    return np.column_stack([np.ones(len(x)), (x - model['mean']) / model['scale']]) @ model['weights']


def metrics(y, p):
    error = np.asarray(y) - np.asarray(p)
    return {'mae_mw': float(np.abs(error).mean()), 'rmse_mw': float(np.sqrt((error ** 2).mean())),
            'wape_percent': float(100 * np.abs(error).sum() / np.abs(y).sum())}


def main():
    demand = prepare(download())
    x = features(demand)
    dataset = x.assign(target=demand).dropna()
    year = dataset.index.tz_convert('Europe/London').year
    train, valid, test = (dataset.loc[mask] for mask in (year <= 2023, year == 2024, year == 2025))
    columns = list(x.columns)
    trials, models = [], {}
    for alpha in (0.1, 1, 10, 100, 1000):
        model = fit_ridge(train[columns].to_numpy(), train.target.to_numpy(), alpha)
        score = metrics(valid.target.to_numpy(), predict(model, valid[columns].to_numpy()))
        trials.append({'alpha': alpha, **score})
        models[alpha] = model
    selected = min(trials, key=lambda r: r['mae_mw'])['alpha']
    # Keep the 2024 calibration model and its training regime unchanged in test.
    # Intervals are empirical calibration bands, with no coverage guarantee
    # under temporal distribution shift.
    model = models[selected]
    export_model(model, columns, train.index.max(), ROOT / 'models', alpha=selected)
    validation_pred = predict(model, valid[columns].to_numpy())
    radius = float(np.quantile(np.abs(valid.target.to_numpy() - validation_pred), 0.9, method='higher'))
    predictions = pd.DataFrame({'actual_mw': test.target,
        'seasonal_28d_mw': test.lag_28d,
        'weekly_mean_mw': test[['lag_28d', 'lag_35d', 'lag_42d', 'lag_49d']].mean(axis=1),
        'ridge_mw': predict(model, test[columns].to_numpy())})
    predictions['lower_90_mw'] = predictions.ridge_mw - radius
    predictions['upper_90_mw'] = predictions.ridge_mw + radius
    scores = {name: metrics(predictions.actual_mw.to_numpy(), predictions[col].to_numpy())
              for name, col in [('seasonal_28d', 'seasonal_28d_mw'), ('weekly_mean', 'weekly_mean_mw'), ('ridge', 'ridge_mw')]}
    coverage = ((predictions.actual_mw >= predictions.lower_90_mw) & (predictions.actual_mw <= predictions.upper_90_mw)).mean()
    best_baseline = min(scores['seasonal_28d']['mae_mw'], scores['weekly_mean']['mae_mw'])
    result = {'train_rows': len(train), 'validation_rows': len(valid), 'test_rows': len(test),
              'selected_alpha': selected, 'validation_trials': trials, 'test_metrics': scores,
              'ridge_mae_improvement_vs_best_baseline_percent': 100 * (best_baseline - scores['ridge']['mae_mw']) / best_baseline,
              'empirical_interval_target': 0.9, 'test_interval_coverage': float(coverage),
              'interval_half_width_mw': radius, 'feature_columns': columns,
              'model_training_end': str(train.index.max()), 'test_start': str(test.index.min()), 'test_end': str(test.index.max())}
    (REPORT / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    predictions.to_csv(REPORT / 'test_predictions.csv', index_label='timestamp_utc')
    month = predictions.index.tz_convert('Europe/London').month
    monthly = pd.DataFrame({'month': month, 'ridge_abs_error_mw': np.abs(predictions.actual_mw - predictions.ridge_mw),
                            'baseline_abs_error_mw': np.abs(predictions.actual_mw - predictions.weekly_mean_mw)})
    monthly.groupby('month').mean().to_csv(REPORT / 'monthly_errors.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    sample = predictions.loc['2025-02-03':'2025-02-09']
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sample.index, sample.actual_mw / 1000, label='Actual', color='#172554', linewidth=1.7)
    ax.plot(sample.index, sample.ridge_mw / 1000, label='Ridge forecast', color='#0d9488', linewidth=1.3)
    ax.fill_between(sample.index, sample.lower_90_mw / 1000, sample.upper_90_mw / 1000, color='#0d9488', alpha=.14, label='Empirical 90% band')
    ax.set(title='GB National Demand | held-out week, February 2025', ylabel='Demand (GW)', xlabel='UTC')
    ax.legend(loc='upper right', ncol=3, frameon=False)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(REPORT / 'forecast_week.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['28-day seasonal', '4-week seasonal mean', 'Ridge regression'], [scores[n]['mae_mw'] for n in scores], color=['#94a3b8', '#64748b', '#0d9488'])
    ax.set(title='2025 holdout | lower error is better', ylabel='Mean absolute error (MW)')
    fig.tight_layout(); fig.savefig(REPORT / 'model_comparison.png', dpi=160); plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k not in ('feature_columns', 'validation_trials')}, indent=2))


if __name__ == '__main__':
    main()
