"""Build an offline HTML report from evaluated predictions, not a live forecast."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports'


def main():
    frame = pd.read_csv(REPORT / 'test_predictions.csv', parse_dates=['timestamp_utc']).set_index('timestamp_utc')
    daily = frame.resample('D').mean().round(2)
    rows = [[str(t.date()), *[float(v) for v in values]] for t, values in zip(daily.index, daily.to_numpy())]
    metrics = json.loads((REPORT / 'metrics.json').read_text())
    page = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GB demand | 2025 evaluation</title>
<style>
:root{font-family:system-ui,sans-serif;color:#172554;background:#f3f6fa}body{max-width:1120px;margin:auto;padding:36px 24px}h1{font-size:38px;letter-spacing:-1.5px;margin:12px 0}p{line-height:1.6;color:#475569}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:2px;color:#0f766e;font-weight:700}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.card,section{background:white;padding:24px;border:1px solid #dde5ee;border-radius:14px}.card strong{display:block;font-size:32px;margin:8px 0}.card small{color:#64748b}section{margin:20px 0}h2{font-size:19px;margin:0 0 14px}select{padding:9px 14px;border:1px solid #cbd5e1;border-radius:8px;background:white;color:#172554}canvas{width:100%;height:330px;display:block;margin-top:18px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #e2e8f0}th{color:#64748b}a{color:#0f766e}.legend{display:flex;gap:24px;flex-wrap:wrap;font-size:13px;margin-top:12px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}footer{font-size:12px;color:#64748b;margin:30px 0}@media(max-width:650px){.cards{grid-template-columns:1fr}body{padding:20px 14px}h1{font-size:30px}section{padding:16px}table{font-size:12px}}
</style>
<div class="eyebrow">Personal project / historical backtest</div>
<h1>Forecasting Britain's electricity demand</h1>
<p>Four years of official NESO data. Two seasonal baselines. One interpretable model.<br>Explore a full year of predictions that were held out from model training.</p>
<div class="cards"><div class="card">2025 prediction error<strong>2,120 MW</strong><small>Ridge mean absolute error</small></div><div class="card">Against the best baseline<strong>24.5% lower</strong><small>MAE over 17,520 half-hourly intervals</small></div><div class="card">Empirical interval coverage<strong>84.8%</strong><small>Below the nominal 90% target</small></div></div>
<section><h2>Actual and predicted demand</h2><label for="month">Show daily averages for </label><select id="month"><option value="all">All of 2025</option></select><canvas id="chart" role="img" aria-label="Daily average actual and forecast demand in gigawatts"></canvas><div class="legend"><span><i class="dot" style="background:#172554"></i>Actual</span><span><i class="dot" style="background:#0d9488"></i>Ridge</span><span><i class="dot" style="background:#94a3b8"></i>28-day seasonal</span></div><p id="period"></p><p><small>Daily averages smooth half-hourly errors. The headline metrics use every half-hour, not these averages. The linked CSV provides the accessible numeric chart data.</small></p><a href="test_predictions.csv">Download half-hourly predictions</a></section>
<section><h2>Model comparison · full 2025 holdout</h2><table><thead><tr><th>Model</th><th>MAE (MW)</th><th>RMSE (MW)</th><th>WAPE</th></tr></thead><tbody id="scores"></tbody></table></section>
<section><h2>How to interpret these results</h2><p>Training: 2022–2023 after lag warm-up. Validation: 2024. Test: 2025. The model stays fixed during the test year. Features use demand at least 28 days old because this public historic feed normally arrives 21 days late.</p><p>This is a backtest using retrospectively corrected data, not an operational forecasting service. Archived publication vintages are unavailable. Weather and holiday effects are not explicitly modelled. The uncertainty band undercovers in 2025 and needs further work.</p><p><a href="../README.md">Methodology and limitations</a> · <a href="metrics.json">Full metrics</a> · <a href="data_quality.json">Data-quality audit</a></p></section>
<footer>Supported by National Energy SO Open Data · Source: <a href="https://www.neso.energy/data-portal/historic-demand-data">NESO Historic Demand Data</a> · <a href="https://www.neso.energy/data-portal/neso-open-licence">NESO Open Licence</a><br>Independent analysis; not endorsed by NESO. Data snapshot retrieved 17 October 2026.</footer>
<script>
const rows=__ROWS__, metrics=__METRICS__;
const select=document.getElementById('month'), canvas=document.getElementById('chart');
for(let m=1;m<=12;m++){let option=document.createElement('option');option.value=String(m).padStart(2,'0');option.textContent=new Date(2025,m-1,1).toLocaleString('en-GB',{month:'long'});select.appendChild(option)}
const names={seasonal_28d:'28-day seasonal',weekly_mean:'4-week seasonal mean',ridge:'Ridge regression'};
document.getElementById('scores').innerHTML=Object.entries(metrics.test_metrics).map(([k,v])=>`<tr><td>${names[k]}</td><td>${v.mae_mw.toFixed(1)}</td><td>${v.rmse_mw.toFixed(1)}</td><td>${v.wape_percent.toFixed(2)}%</td></tr>`).join('');
function draw(){const data=rows.filter(r=>select.value==='all'||r[0].slice(5,7)===select.value),ctx=canvas.getContext('2d'),w=canvas.clientWidth,h=330,dpr=window.devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);const left=55,right=w-18,top=15,bottom=h-40;const vals=data.flatMap(r=>[r[1],r[2],r[4]]).map(v=>v/1000),lo=Math.floor(Math.min(...vals)/5)*5,hi=Math.ceil(Math.max(...vals)/5)*5;const x=i=>left+i*(right-left)/Math.max(1,data.length-1),y=v=>bottom-(v/1000-lo)/(hi-lo)*(bottom-top);ctx.font='12px system-ui';ctx.fillStyle='#64748b';for(let v=lo;v<=hi;v+=5){const py=y(v*1000);ctx.strokeStyle='#e2e8f0';ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(right,py);ctx.stroke();ctx.fillText(v+' GW',0,py+4)}for(const [col,color] of [[2,'#94a3b8'],[1,'#172554'],[4,'#0d9488']]){ctx.strokeStyle=color;ctx.lineWidth=col===4?2.3:1.6;ctx.beginPath();data.forEach((r,i)=>i?ctx.lineTo(x(i),y(r[col])):ctx.moveTo(x(i),y(r[col])));ctx.stroke()}ctx.fillStyle='#64748b';ctx.fillText(data[0][0],left,h-10);ctx.fillText(data[data.length-1][0],right-80,h-10);document.getElementById('period').textContent=`${data.length} daily averages · ${data[0][0]} to ${data[data.length-1][0]}`}
select.addEventListener('change',draw);window.addEventListener('resize',draw);draw();
</script></html>'''
    page = page.replace('__ROWS__', json.dumps(rows)).replace('__METRICS__', json.dumps(metrics))
    page = page.replace('2,120 MW', f"{metrics['test_metrics']['ridge']['mae_mw']:,.0f} MW")
    page = page.replace('24.5% lower', f"{metrics['ridge_mae_improvement_vs_best_baseline_percent']:.1f}% lower")
    page = page.replace('84.8%', f"{100 * metrics['test_interval_coverage']:.1f}%")
    page = page.replace('17,520 half-hourly intervals', f"{metrics['test_rows']:,} half-hourly intervals")
    (REPORT / 'dashboard.html').write_text(page, encoding='utf-8')
    print('Built reports/dashboard.html')


if __name__ == '__main__':
    main()
