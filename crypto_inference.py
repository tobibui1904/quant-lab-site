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

    Live quotes refresh independently of training. Learning uses completed daily
    bars; the newest 30 days are reserved for testing and the preceding 60 for
    validation. Each pair qualifies independently with positive trading returns
    in both windows, at least one simulated trade in each, and no more than 20%
    drawdown. Training and evaluation use your zero-fee assumption.
    """)
    return


@app.cell
def _(mo):
    status_tick=mo.ui.refresh(options=[2,5,10],default_interval=2,label='Refresh status')
    refresh_button=mo.ui.run_button(label='Refresh prices & learning')
    mo.hstack([refresh_button,status_tick],justify='start')
    return refresh_button, status_tick


@app.cell
def _(refresh_button, service):
    if refresh_button.value:
        service.start(force=True)
    return




@app.cell
def _(service, status_tick):
    status_tick.value
    service.start()
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
    _parts.append(mo.md(f"Completed daily bars through **{snapshot.get('bar_time','loading')}**. Quotes above are live observations, not training labels."))
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
        _parts.append(mo.md('These are historical held-out strategy returns, not predicted profits for the next trade. ETH/BTC profit is measured in BTC, excluding changes in BTC/USD. Training assumes zero fees and no fixed slippage surcharge; paper orders use current bid/ask prices.'))
    else:
        _parts.append(mo.md('### Model evaluation\nThe corrected model is preparing its first evaluation. Live prices remain available above.'))
    mo.vstack(_parts)
    return


@app.cell
def _(mo, snapshot):
    _rows=[]
    for _signal in snapshot.get('signals',[]):
        _validation=_signal.get('validation_return'); _test=_signal.get('test_return')
        _rows.append({'Pair':_signal['symbol'],'Model signal':_signal['action'],
            'Recommendation':_signal.get('recommendation','HOLD'),
            'Validation return':f'{_validation:.2%}' if _validation is not None else 'Unavailable',
            'Test return':f'{_test:.2%}' if _test is not None else 'Unavailable',
            'Profit currency':_signal.get('quote_currency'),'Reason':_signal.get('reason')})
    mo.vstack([mo.md('### BUY / HOLD / SELL recommendations'),
        mo.ui.table(_rows,selection=None) if _rows else mo.md('Signals appear after model evaluation finishes.'),
        mo.md('All three pairs use Alpaca market data directly. Signals use actual paper holdings when connected; otherwise they assume a cash-only portfolio. ETH/BTC buys spend available BTC; sells spend available ETH. Eligibility does not override stale quotes, broker minimums or insufficient balances.')])
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
    mo.vstack([mo.md('### Paper execution\nEach click executes eligible BUY/SELL recommendations and reports why other pairs were skipped. It rechecks live quotes, balances, broker minimums and pending orders. At most one order per symbol and completed bar; no orders run automatically.'),execute_button])
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
