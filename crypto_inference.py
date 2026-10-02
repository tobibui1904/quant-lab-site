"""Crypto hub: fresh prices first, cloud-cached learning in the background."""

import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="medium",
    css_file="../../theme.css",
    html_head_file="../../theme_head.html",
)


@app.cell
def _():
    from pathlib import Path
    import sys
    import os
    import marimo as mo
    import pandas as pd
    from dotenv import load_dotenv
    _root=Path(__file__).resolve().parents[2]
    for _path in (_root/'shared',Path(__file__).resolve().parent):
        if str(_path) not in sys.path: sys.path.insert(0,str(_path))
    load_dotenv(_root/'.env')
    from crypto_pipeline import get_service
    from crypto_execution import account_state, paper_client, portfolio_state, execute_service

    return (
        account_state,
        execute_service,
        get_service,
        mo,
        paper_client,
        pd,
        portfolio_state,
    )


@app.cell
def _(get_service):
    service=get_service()
    service.start(force=True)
    return (service,)


@app.cell
def _(mo):
    mo.md("""
    # Crypto research & paper trading
    MAPPO-LSTM · BTC/USD · ETH/USD · ETH/BTC · Alpaca market data

    Opening this desk or clicking Refresh fetches the latest available prices once.
    Automatic checkpoint updates also run Saturday at 12:10 AM New York time.
    Training includes the newest completed daily bar. Each daily decision is scored
    before the model learns that day's outcome; the newest 30 scored days form the
    test window and the preceding 60 form validation. Each pair qualifies with positive trading returns
    in both windows, at least one simulated trade in each, and no more than 20%
    drawdown. Training and evaluation use your zero-fee assumption.
    """)
    return


@app.cell
def _(mo):
    # Keep completion updates automatic without exposing a polling control.
    status_tick=mo.ui.refresh(default_interval=2)
    refresh_button=mo.ui.run_button(label='Refresh prices & learning')
    mo.hstack([refresh_button,status_tick.style({'display':'none'})],justify='start')
    return refresh_button, status_tick


@app.cell
def _(refresh_button, service):
    if refresh_button.value:
        service.start(force=True)
    return




@app.cell
def _(service, status_tick):
    status_tick.value
    service.poll()
    snapshot=service.snapshot()
    return (snapshot,)


@app.cell
def _(mo, pd, snapshot):
    _parts=[mo.md(f"**{snapshot['status'].capitalize()}** — {snapshot['message']}")]
    if snapshot.get('error'):
        _parts.append(mo.callout(snapshot['error'],kind='danger'))
    if snapshot.get('quotes_error'):
        _parts.append(mo.callout('Live quote refresh failed: '+snapshot['quotes_error'],kind='warn'))
    if snapshot.get('account_error'):
        _parts.append(mo.callout(snapshot['account_error'],kind='warn'))
    _quotes=[]
    for _symbol,_quote in snapshot['quotes'].items():
        _age=(pd.Timestamp.now(tz='UTC')-pd.Timestamp(_quote['timestamp'])).total_seconds()
        _quotes.append({'Pair':_symbol,'Bid':_quote['bid'],'Ask':_quote['ask'],
            'Quote time (UTC)':_quote['timestamp'],'Freshness':'Fresh' if 0<=_age<=60 else 'Stale',
            'Source':'Alpaca'})
    if _quotes: _parts.append(mo.ui.table(_quotes,selection=None))
    if snapshot.get('live_error'):
        _parts.append(mo.callout(snapshot['live_error']+'. Signals fall back to the last completed daily bar and paper execution is disabled.',kind='warn'))
    _parts.append(mo.md(f"Completed daily data is available through **{snapshot.get('bar_time','loading')}**. Signals recompute the current rolling window using the latest fetched Alpaca quote midpoint on opening, Refresh or Execute. Signal timestamps below come from the provider's quotes; they are separate from the checkpoint's training date. Today's unfinished bar is never a training outcome."))
    mo.vstack(_parts)
    return


@app.cell
def _(mo, snapshot):
    _report=snapshot.get('report')
    _parts=[]
    if _report:
        _parts.append(mo.md(f"### Model evaluation\nTraining through **{_report['trained_through']}** · Evaluated through **{_report['data_through']}**"))
        _parts.append(mo.md(f"Model observations begin **{_report.get('feature_start','unavailable')}** after indicator warm-up. Training uses continuous daily history; missing days are not filled with calculated prices."))
        _rows=[]
        for _symbol,_gate in _report.get('pair_eligibility',{}).items():
            for _window in ('validation','test'):
                _profit=_gate.get(_window+'_return'); _drawdown=_gate.get(_window+'_drawdown')
                _rows.append({'Pair':_symbol,'Window':_window.title(),'Profit currency':_gate['quote_currency'],
                    'Trading return':f'{_profit:.2%}' if _profit is not None else 'Unavailable',
                    'Max drawdown':f'{_drawdown:.2%}' if _drawdown is not None else 'Unavailable',
                    'Trades':_gate.get(_window+'_trades',0),'Pair eligible':'Yes' if _gate['eligible'] else 'No'})
        _parts.append(mo.ui.table(_rows,selection=None))
        _parts.append(mo.callout('Qualifying pairs can execute with the paper button; other pairs remain HOLD.' if _report['execution_ready'] else
            'No pair currently meets the positive trading-return checks. Recommendations remain HOLD.',
            kind='success' if _report['execution_ready'] else 'warn'))
        _parts.append(mo.md('These are historical out-of-sample results of the updating strategy: each decision was scored before training on its outcome. They are not a backtest of the final weights or a next-trade profit forecast. ETH/BTC profit is measured in BTC. Training assumes zero fees; the paper button fetches current bid/ask prices again.'))
    else:
        _parts.append(mo.md('### Model evaluation\nThe corrected model is preparing its first evaluation. Live prices remain available above.'))
    mo.vstack(_parts)
    return


@app.cell
def _(mo, snapshot):
    _live=snapshot.get('live_evaluation',[])
    _rows=[]
    for _result in _live:
        _valid=_result['status']=='provisional'
        _rows.append({'Pair':_result['symbol'],'Profit currency':_result['quote_currency'],
            'Test return through quote':f"{_result['test_return']:.2%}" if _valid else 'Unavailable',
            'Change since last daily close':f"{_result['since_completed_return']:.2%}" if _valid else 'Unavailable',
            'Valuation bid':_result.get('bid'), 'Quote time (UTC)':_result.get('quote_time','Unavailable'),
            'Quote age when fetched':f"{_result['quote_age_at_fetch_seconds']:.0f}s" if _valid else 'Unavailable',
            'Status':('Provisional' if _result['quote_fresh_at_fetch'] else 'Provisional · older quote') if _valid else _result['reason']})
    mo.vstack([mo.md('### Live-price evaluation · Provisional'),
        mo.ui.table(_rows,selection=None) if _rows else mo.md('Available after the current model and opening-price snapshot are ready.'),
        mo.md('The saved test portfolio is valued at the latest available Alpaca bids fetched on opening or Refresh. Returns extend the daily test from the same starting value; no new trades are simulated. ETH/BTC is valued directly in BTC. Older quotes are labeled with their age when fetched. This is an unrealized snapshot, not a completed-day test or a forecast of the next signal. It does not change training or paper eligibility; execution still requires fresh quotes.')])
    return


@app.cell
def _(mo, snapshot):
    _rows=[]
    for _signal in snapshot.get('signals',[]):
        _validation=_signal.get('validation_return'); _test=_signal.get('test_return')
        _rows.append({'Pair':_signal['symbol'],'Signal price time (UTC)':_signal.get('as_of'),
            'Signal price':_signal.get('reference_price'),'Price basis':_signal.get('price_basis'),
            'Model signal':_signal['action'],
            'Recommendation':_signal.get('recommendation','HOLD'),
            'Validation return':f'{_validation:.2%}' if _validation is not None else 'Unavailable',
            'Test return':f'{_test:.2%}' if _test is not None else 'Unavailable',
            'Profit currency':_signal.get('quote_currency'),'Reason':_signal.get('reason')})
    mo.vstack([mo.md('### BUY / HOLD / SELL recommendations'),
        mo.ui.table(_rows,selection=None) if _rows else mo.md('Signals appear after model evaluation finishes.'),
        mo.md('All three pairs use direct Alpaca data. The latest fetched bid/ask midpoint updates the current inference window, so signals can change within the day without retraining. Refreshing may still produce the same action. Signals use actual paper holdings when connected; otherwise they assume a cash-only portfolio. ETH/BTC buys spend available BTC; sells spend available ETH. Eligibility does not override stale quotes, broker minimums or insufficient balances.')])
    return


@app.cell
def _(mo, service, snapshot):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _frame=service.frame if snapshot.get('bar_time') else None
    _chart=None
    if _frame is not None:
        _tail=_frame.tail(90)
        _chart=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=['BTC/USD','ETH/USD','ETH/BTC'])
        for _row,_pair in enumerate(('btc','eth','etb'),1):
            _chart.add_trace(go.Scatter(x=_tail.index,y=_tail[f'{_pair}_close'],name=_pair.upper(),mode='lines'),row=_row,col=1)
        _chart.update_layout(template='plotly_dark',height=600,margin=dict(l=20,r=20,t=40,b=20),showlegend=False)
    mo.ui.plotly(_chart) if _chart is not None else mo.md('Daily history is loading.')
    return


@app.cell
def _(mo):
    execute_button=mo.ui.run_button(label='Execute eligible paper signals')
    mo.vstack([mo.md('### Paper execution\nEach click executes eligible BUY/SELL recommendations and reports why other pairs were skipped. It refetches the latest trades and recomputes signals first, then rechecks live quotes, balances, broker minimums and pending crypto orders. At most one order per symbol per UTC hour; no orders run automatically.'),execute_button])
    return (execute_button,)


@app.cell
def _(execute_button, execute_service, mo, service):
    mo.stop(not execute_button.value)
    try:
        _orders=execute_service(service)
        _output=mo.ui.table(_orders,selection=None) if _orders else mo.md('No eligible orders (hold, existing orders, or insufficient balance).')
    except Exception as _exc:
        _output=mo.callout(str(_exc),kind='danger')
    _output
    return


if __name__ == "__main__":
    app.run()
