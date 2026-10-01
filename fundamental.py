import marimo

__generated_with = "0.23.9"
app = marimo.App(css_file="../../theme.css", html_head_file="../../theme_head.html")


@app.cell
def _():
    with open('data/pair.txt', 'r') as f:
        content = f.read()
        lines = content.splitlines()
        pair = lines[0].split(" ")
    return (pair,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import requests
    from bs4 import BeautifulSoup
    import logging
    import calendar
    import time
    import yfinance as yf
    import json
    import re
    import sys
    import textwrap
    from datetime import datetime
    from typing import Any
    import ollama
    import marimo as mo

    # When stdout is a pipe rather than a console, Python encodes it with the
    # locale codec — cp1252 on this machine — and the report's ✓/✗/⚠/═ output
    # raises UnicodeEncodeError. Force UTF-8 on the real streams; marimo's
    # in-notebook capture stream has no reconfigure().
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

    return (
        Any,
        BeautifulSoup,
        calendar,
        datetime,
        json,
        logging,
        mo,
        np,
        ollama,
        pd,
        re,
        requests,
        sys,
        textwrap,
        time,
        yf,
    )


@app.cell
def _(mo):
    def page_header(title, section_number, subtitle=None):
        subtitle_html = f"""
          <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.1em; margin-top: 8px;">{subtitle}</div>
        """ if subtitle else ""

        return mo.Html(f"""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <div style="padding: 2rem 0 1.5rem;">
      <div style="display: flex; align-items: flex-start; gap: 16px;">
        <div style="padding-top: 6px;">
          <div style="width: 2px; height: 52px; background: #1D9E75;"></div>
        </div>
        <div>
          <div style="font-family: 'DM Mono', monospace; font-size: 11px; color: #1D9E75; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 6px;">Section {section_number}</div>
          <div style="font-family: 'DM Serif Display', serif; font-size: 42px; font-style: italic; font-weight: 400; line-height: 1.05; color: var(--color-text-primary);">{title}</div>
          {subtitle_html}
        </div>
      </div>
    </div>
    """)

    return (page_header,)


@app.cell
def _(mo):
    mo.Html("""
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

    <div style="padding: 2.5rem 0 2rem; text-align: center;">
      <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; border: 0.5px solid #1D9E75; margin-bottom: 1rem;">
        <i class="ti ti-chart-candle" style="font-size: 22px; color: #1D9E75;"></i>
      </div>
      <h1 style="font-family: 'DM Serif Display', serif; font-size: 36px; font-weight: 400; font-style: italic; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--color-text-primary);">Financial Analysis Strategy</h1>
      <p style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.18em; text-transform: uppercase; margin: 0 0 1.25rem;">Designed optimally for Options</p>
      <div style="display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: 11px; color: #0F6E56; background: #E1F5EE; padding: 4px 14px; border-radius: 999px;">
        <span style="width: 6px; height: 6px; border-radius: 50%; background: #1D9E75; display: inline-block;"></span>
        Live
      </div>
      <div style="margin-top: 1.5rem; width: 40px; height: 0.5px; background: var(--color-border-tertiary); margin-left: auto; margin-right: auto;"></div>
    </div>
    """)
    return


@app.cell
def _(mo, pair):
    mo.Html(f"""
    <div style="padding:2rem 0;display:flex;justify-content:center;">
      <div style="display:inline-flex;align-items:center;gap:0;
        background:var(--color-background-primary);
        border:0.5px solid var(--color-border-tertiary);
        border-radius:var(--border-radius-lg);
        overflow:hidden;">

        <div style="padding:18px 28px;display:flex;flex-direction:column;align-items:center;gap:6px;
          background:#EEEDFE;">
          <span style="font-size:11px;font-weight:500;letter-spacing:.1em;text-transform:uppercase;
            color:#534AB7;">leg A</span>
          <span style="font-size:26px;font-weight:500;color:#26215C;font-family:var(--font-mono);">{pair[0]}</span>
          <span style="font-size:11px;color:#534AB7;">long</span>
        </div>

        <div style="padding:18px 16px;display:flex;flex-direction:column;align-items:center;gap:4px;
          border-left:0.5px solid var(--color-border-tertiary);
          border-right:0.5px solid var(--color-border-tertiary);">
          <i class="ti ti-arrows-exchange" style="font-size:20px;color:var(--color-text-secondary);" aria-hidden="true"></i>
          <span style="font-size:10px;letter-spacing:.08em;color:var(--color-text-secondary);text-transform:uppercase;">pair</span>
        </div>

        <div style="padding:18px 28px;display:flex;flex-direction:column;align-items:center;gap:6px;
          background:#E1F5EE;">
          <span style="font-size:11px;font-weight:500;letter-spacing:.1em;text-transform:uppercase;
            color:#0F6E56;">leg B</span>
          <span style="font-size:26px;font-weight:500;color:#04342C;font-family:var(--font-mono);">{pair[1]}</span>
          <span style="font-size:11px;color:#0F6E56;">short</span>
        </div>

      </div>
    </div>
    """)
    return


@app.cell
def _(page_header):
    page_header("Crawling Financial Statements", "01")
    return


@app.cell
def _(BeautifulSoup, calendar, logging, mo, np, pair, pd, requests, time, yf):
    """
    Extract Latest 20 Quarterly (10-Q) Financial Statements from SEC EDGAR
    Returns pandas DataFrames instead of saving to Excel
    """


    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    pd.options.display.float_format = (
        lambda x: "{:,.0f}".format(x) if int(x) == x else "{:,.2f}".format(x)
    )

    # ============================================================================
    # CONFIGURATION - CHANGE YOUR EMAIL HERE
    # ============================================================================

    HEADERS = {"User-Agent": "buituannghia1904@gmail.com"}  # REQUIRED BY SEC

    # Statement name mappings - keys will match if contained in statement name
    STATEMENT_KEYS_MAP = {
        "balance_sheet": [
            "consolidated balance sheets",
            "balance sheets",
            "statements of financial position",
            "financial position",
            "condensed consolidated balance sheet",
            "consolidated balance sheet",
            "balance sheet",
        ],
        "income_statement": [
            "consolidated statements of operations",
            "consolidated results of operations",
            "statements of operations",
            "consolidated statements of income",
            "statements of income",
            "condensed consolidated statement of earnings",
            "consolidated statement of earnings",
            "statement of earnings",
            "statements of earnings",
            "consolidated statements of income and comprehensive income",
            "condensed consolidated statements of income and comprehensive income",
        ],
        "cash_flow_statement": [
            "consolidated statements of cash flows",
            "statements of cash flows",
            "cash flows",
            "condensed consolidated statement of cash flows",
            "consolidated statement of cash flows",
            "statement of cash flows",
        ],
    }

    # ============================================================================
    # CORE FUNCTIONS
    # ============================================================================

    def cik_matching_ticker(ticker, headers=HEADERS):
        """Convert ticker to CIK number."""
        ticker = ticker.upper().replace(".", "-")
        ticker_json = requests.get(
            "https://www.sec.gov/files/company_tickers.json", headers=headers
        ).json()

        for company in ticker_json.values():
            if company["ticker"] == ticker:
                cik = str(company["cik_str"]).zfill(10)
                return cik
        raise ValueError(f"Ticker {ticker} not found")


    def get_submission_data_for_ticker(ticker, headers=HEADERS):
        """Get submission data for a ticker."""
        cik = cik_matching_ticker(ticker)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        company_json = requests.get(url, headers=headers).json()
        return pd.DataFrame(company_json["filings"]["recent"])


    def get_10q_filings(ticker, headers=HEADERS):
        """Get 10-Q filings for a ticker."""
        filings_df = get_submission_data_for_ticker(ticker, headers)
        return filings_df[filings_df["form"] == "10-Q"]


    def get_facts(ticker, headers=HEADERS):
        """Retrieve company facts for label mapping."""
        cik = cik_matching_ticker(ticker)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        return requests.get(url, headers=headers).json()


    def get_label_dictionary(ticker, headers=HEADERS):
        """Get human-readable labels for XBRL tags."""
        facts = get_facts(ticker, headers)
        us_gaap_data = facts["facts"]["us-gaap"]
        return {fact: details["label"] for fact, details in us_gaap_data.items()}


    def get_statement_file_names(ticker, accession_number, headers=HEADERS):
        """Get available statement file names from FilingSummary.xml."""
        try:
            cik = cik_matching_ticker(ticker)
            base_link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}"
            filing_summary_link = f"{base_link}/FilingSummary.xml"

            response = requests.get(filing_summary_link, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "lxml-xml")
            statement_files = {}

            for report in soup.find_all("Report"):
                html_file = report.find("HtmlFileName")
                xml_file = report.find("XmlFileName")

                file_name = html_file.text if html_file else (xml_file.text if xml_file else None)
                if not file_name:
                    continue

                short_name = report.find("ShortName")
                long_name = report.find("LongName")

                if short_name and long_name and "Statement" in long_name.text:
                    statement_files[short_name.text.lower()] = file_name

            return statement_files
        except Exception as e:
            logging.error(f"Error getting filing summary: {e}")
            return {}


    def get_statement_soup(ticker, accession_number, statement_name, headers=HEADERS):
        """Get BeautifulSoup object for a specific statement."""
        cik = cik_matching_ticker(ticker)
        base_link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}"

        statement_files = get_statement_file_names(ticker, accession_number, headers)

        if not statement_files:
            raise ValueError("No FilingSummary.xml found")

        # Find matching statement - try exact match first, then partial match
        statement_link = None
        search_keys = STATEMENT_KEYS_MAP.get(statement_name.lower(), [])

        # Try exact match first
        for possible_key in search_keys:
            file_name = statement_files.get(possible_key.lower())
            if file_name:
                statement_link = f"{base_link}/{file_name}"
                break

        # If no exact match, try partial match (key contained in statement name)
        if not statement_link:
            for possible_key in search_keys:
                for available_name, file_name in statement_files.items():
                    if possible_key.lower() in available_name.lower():
                        statement_link = f"{base_link}/{file_name}"
                        logging.info(f"Partial match: '{possible_key}' found in '{available_name}'")
                        break
                if statement_link:
                    break

        if not statement_link:
            available = list(statement_files.keys())
            raise ValueError(f"Could not find {statement_name}. Available: {available}")

        response = requests.get(statement_link, headers=headers)
        response.raise_for_status()

        if statement_link.endswith(".xml"):
            return BeautifulSoup(response.content, "lxml-xml", from_encoding="utf-8")
        else:
            return BeautifulSoup(response.content, "lxml")


    def standardize_date(date: str) -> str:
        """Standardize date strings."""
        for abbr, full in zip(calendar.month_abbr[1:], calendar.month_name[1:]):
            date = date.replace(abbr, full)
        return date


    def keep_numbers_only(mixed_string: str):
        """Filter string to keep only numbers and decimals."""
        return "".join(filter(lambda x: x in "1234567890.", mixed_string))


    def get_dates_from_statement(soup: BeautifulSoup) -> pd.DatetimeIndex:
        """Extract dates from statement headers."""
        table_headers = soup.find_all("th", {"class": "th"})
        dates = [str(th.div.string) for th in table_headers if th.div and th.div.string]
        dates = [standardize_date(date).replace(".", "") for date in dates]
        return pd.to_datetime(dates)


    def extract_data_from_statement(soup):
        """Extract financial data from HTML statement."""
        columns = []
        values_set = []
        date_index = get_dates_from_statement(soup)

        for table in soup.find_all("table"):
            unit_multiplier = 1
            special_case = False

            table_header = table.find("th")
            if table_header:
                header_text = table_header.get_text()
                if "in Thousands" in header_text:
                    unit_multiplier = 1
                elif "in Millions" in header_text:
                    unit_multiplier = 1000
                if "unless otherwise specified" in header_text:
                    special_case = True

            for row in table.select("tr"):
                onclick_elements = row.select("td.pl a, td.pl.custom a")
                if not onclick_elements:
                    continue

                onclick_attr = onclick_elements[0]["onclick"]
                column_title = onclick_attr.split("defref_")[-1].split("',")[0]
                columns.append(column_title)

                values = [np.nan] * len(date_index)

                for i, cell in enumerate(row.select("td.text, td.nump, td.num")):
                    if "text" in cell.get("class"):
                        continue

                    value = keep_numbers_only(
                        cell.text.replace("$", "").replace(",", "")
                        .replace("(", "").replace(")", "").strip()
                    )

                    if value:
                        value = float(value)
                        if special_case:
                            value /= 1000
                        else:
                            if "nump" in cell.get("class"):
                                values[i] = value * unit_multiplier
                            else:
                                values[i] = -value * unit_multiplier

                values_set.append(values)

        return columns, values_set, date_index


    def process_statement(ticker, accession_number, statement_name, headers=HEADERS):
        """Process a single financial statement."""
        try:
            soup = get_statement_soup(ticker, accession_number, statement_name, headers)
            columns, values, dates = extract_data_from_statement(soup)

            transposed_values = list(zip(*values))
            df = pd.DataFrame(transposed_values, columns=columns, index=dates)

            if not df.empty:
                df = df.T

                # Deduplicate row index (XBRL concept tags)
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='first')]

                # Deduplicate date columns
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]

                # Normalise column names to plain "YYYY-MM-DD" strings
                df.columns = [
                    c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c)
                    for c in df.columns
                ]

                return df
            return None
        except Exception as e:
            logging.error(f"Error processing statement: {e}")
            return None


    def rename_statement(statement, label_dict):
        """Rename XBRL tags to human-readable labels."""
        statement.index = statement.index.map(
            lambda x: label_dict.get(x.split("_", 1)[-1], x)
        )
        return statement


    # ============================================================================
    # COMBINE HELPER - safe against duplicate index labels
    # ============================================================================

    def combine_statements(quarters_data, stmt_key):
        """
        Combine a statement across quarters using the first (most recent) column
        of each quarter's DataFrame. Safe against duplicate index labels.
        """
        series_dict = {}
        for q in quarters_data:
            df = q.get(stmt_key)
            if df is not None and not df.empty:
                col_name = f"Q{q['quarter']}_{q['report_date']}"
                series_dict[col_name] = df.iloc[:, 0]

        if not series_dict:
            return None

        try:
            # pd.concat(axis=1) does NOT reindex, so duplicate index labels survive
            combined = pd.concat(series_dict.values(), axis=1)
            combined.columns = list(series_dict.keys())
            return combined
        except Exception as e:
            logging.error(f"combine_statements failed for '{stmt_key}': {e}")
            return None


    # ============================================================================
    # DEBUG FUNCTIONS
    # ============================================================================

    def check_available_filings(ticker):
        """Debug function to see available 10-Q filings with detailed timing"""
        cik = cik_matching_ticker(ticker)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = requests.get(url, headers=HEADERS).json()

        company_name = data.get('name', 'Unknown')
        filings_df = pd.DataFrame(data["filings"]["recent"])
        quarterly_filings = filings_df[filings_df["form"] == "10-Q"]

        print(f"\nCompany: {company_name}")
        print(f"CIK: {cik}")
        print(f"Total 10-Q filings available: {len(quarterly_filings)}")
        print(f"\nAvailable 10-Q filings for {ticker}:")
        print("="*90)

        for idx, row in quarterly_filings.head(10).iterrows():
            print(f"Report Date: {row['reportDate']} | Filed: {row['filingDate']} | Time: {row['acceptanceDateTime']}")
            print(f"  Accession: {row['accessionNumber']}")

        return quarterly_filings


    def check_statement_names(ticker, accession_number):
        """Debug function to see what statements are available in a filing"""
        print(f"\nChecking statements in filing {accession_number}...")
        print("="*70)

        acc_num = accession_number.replace("-", "")
        statement_files = get_statement_file_names(ticker, acc_num)

        if statement_files:
            print("\nAvailable statements:")
            for name, filename in statement_files.items():
                print(f"  - {name}")
                print(f"    File: {filename}")
        else:
            print("❌ No FilingSummary.xml found or no statements available")

        return statement_files


    # ============================================================================
    # YFINANCE FALLBACK
    # ============================================================================

    def extract_yf_quarters(ticker, n=20):
        stock = yf.Ticker(ticker)

        income   = stock.quarterly_income_stmt
        balance  = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cash_flow

        # Guard: if any statement is completely empty, return empty structure
        if income.empty and balance.empty and cashflow.empty:
            print(f"⚠️  yfinance returned no data for {ticker}")
            return {
                "quarters":                    [],
                "income_statement_latest":     None,
                "balance_sheet_latest":        None,
                "cash_flow_statement_latest":  None,
            }

        # Trim to n columns each, then use the smallest common count
        income   = income.iloc[:,   :n] if not income.empty   else income
        balance  = balance.iloc[:,  :n] if not balance.empty  else balance
        cashflow = cashflow.iloc[:, :n] if not cashflow.empty else cashflow

        num_quarters = min(
            income.shape[1]   if not income.empty   else 0,
            balance.shape[1]  if not balance.empty  else 0,
            cashflow.shape[1] if not cashflow.empty else 0,
            n,
        )

        if num_quarters == 0:
            print(f"⚠️  yfinance returned no usable quarters for {ticker}")
            return {
                "quarters":                    [],
                "income_statement_latest":     None,
                "balance_sheet_latest":        None,
                "cash_flow_statement_latest":  None,
            }

        quarters_data = []
        for i in range(num_quarters):
            # Use income date as canonical; fall back to balance or cashflow
            if not income.empty:
                date = income.columns[i]
            elif not balance.empty:
                date = balance.columns[i]
            else:
                date = cashflow.columns[i]

            quarters_data.append({
                "quarter":             i + 1,
                "report_date":         date.strftime("%Y-%m-%d"),
                "income_statement":    income.iloc[:,   [i]] if not income.empty   else None,
                "balance_sheet":       balance.iloc[:,  [i]] if not balance.empty  else None,
                "cash_flow_statement": cashflow.iloc[:, [i]] if not cashflow.empty else None,
            })

        return {
            "quarters":                    quarters_data,
            "income_statement_latest":     income   if not income.empty   else None,
            "balance_sheet_latest":        balance  if not balance.empty  else None,
            "cash_flow_statement_latest":  cashflow if not cashflow.empty else None,
        }


    # ============================================================================
    # MAIN FUNCTION
    # ============================================================================

    def extract_latest_4_quarters(ticker, show_filing_times=False):
        """
        Extract all three financial statements for the latest 20 quarters.

        Returns:
            dict with keys:
                'quarters'                  - list of per-quarter dicts
                'balance_sheet_latest'      - combined DataFrame (rows=line items, cols=quarters)
                'income_statement_latest'   - combined DataFrame
                'cash_flow_statement_latest'- combined DataFrame
        """
        print("\n" + "="*70)
        print(f"Extracting Latest 20 Quarterly (10-Q) Statements for {ticker.upper()}")
        print("="*70)

        # ── 1. Fetch filing index ────────────────────────────────────────────────
        try:
            # Fetch canonical us-gaap facts ONCE: used both for human labels and
            # for the deterministic ratio engine (see compute_ratios downstream).
            company_facts = get_facts(ticker)
            us_gaap_facts = company_facts.get("facts", {}).get("us-gaap", {})
            label_dict = {tag: d["label"] for tag, d in us_gaap_facts.items()}

            cik = cik_matching_ticker(ticker)
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            data = requests.get(url, headers=HEADERS).json()

            filings_df = pd.DataFrame(data["filings"]["recent"])
            filings = filings_df[filings_df["form"] == "10-Q"]

            if filings.empty:
                print(f"❌ No 10-Q filings found for {ticker}, falling back to yfinance...")
                return extract_yf_quarters(ticker, n=20)

            print(f"\nCompany: {data.get('name', 'Unknown')}")
            print(f"Found {len(filings)} total 10-Q filings")
            print(f"Extracting the 20 most recent...\n")

            num_to_extract = min(20, len(filings))
            filings = filings.head(num_to_extract).iloc[::-1].reset_index(drop=True)

        except Exception as e:
            print(f"\n⚠️  Could not fetch filing index for {ticker}: {e}")
            print("Falling back to yfinance...")
            return extract_yf_quarters(ticker, n=20)

        # ── 2. Fetch each quarter's statements ───────────────────────────────────
        quarters_data = []

        for idx, row in filings.iterrows():
            acc_num         = row["accessionNumber"].replace("-", "")
            report_date     = row["reportDate"]
            filing_date     = row["filingDate"]
            acceptance_time = row["acceptanceDateTime"]

            print(f"\n{'='*70}")
            print(f"Quarter {len(quarters_data) + 1}/{num_to_extract}:")
            print(f"Report Date: {report_date}, Filing Date: {filing_date}")
            if show_filing_times:
                print(f"Acceptance Time: {acceptance_time}")
            print('='*70)

            quarter_data = {
                "quarter":              len(quarters_data) + 1,
                "report_date":          report_date,
                "filing_date":          filing_date,
                "acceptance_datetime":  acceptance_time,
                "accession_number":     row["accessionNumber"],
            }

            for stmt_name in ["balance_sheet", "income_statement", "cash_flow_statement"]:
                print(f"\n  Retrieving {stmt_name.replace('_', ' ').title()}...", end=" ")
                time.sleep(0.11)  # SEC rate limit: 10 req/s

                try:
                    df = process_statement(ticker, acc_num, stmt_name)
                    if df is not None and not df.empty:
                        renamed_df = rename_statement(df, label_dict)
                        quarter_data[stmt_name] = renamed_df
                        print(f"✓ ({renamed_df.shape[0]} items × {renamed_df.shape[1]} periods)")
                    else:
                        quarter_data[stmt_name] = None
                        print("✗ Empty result")
                except Exception as e:
                    quarter_data[stmt_name] = None
                    error_msg = str(e)[:100]
                    print(f"✗ Error: {error_msg}")
                    logging.error(f"Full error for {stmt_name} [{report_date}]: {e}")

            quarters_data.append(quarter_data)

        # ── 3. Print summary ─────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print('='*70)

        for q in quarters_data:
            print(f"\nQuarter {q['quarter']}: {q['report_date']} (Filed: {q['filing_date']})")
            if show_filing_times:
                print(f"  Accepted: {q['acceptance_datetime']}")
            for stmt_name in ['balance_sheet', 'income_statement', 'cash_flow_statement']:
                df = q.get(stmt_name)
                if df is not None:
                    print(f"  {stmt_name.replace('_', ' ').title()}: {df.shape} (rows × columns)")
                else:
                    print(f"  {stmt_name.replace('_', ' ').title()}: Not available")

        print(f"\n✓ Successfully extracted {len(quarters_data)} quarters")

        # ── 4. Combine into wide DataFrames ──────────────────────────────────────
        # NOTE: this is intentionally OUTSIDE the try/except so any error here
        # is visible rather than silently returning None.
        print(f"\n{'='*70}")
        print("Creating Combined Latest Period DataFrames...")
        print('='*70)

        return {
            'quarters':                    quarters_data,
            'balance_sheet_latest':        combine_statements(quarters_data, 'balance_sheet'),
            'income_statement_latest':     combine_statements(quarters_data, 'income_statement'),
            'cash_flow_statement_latest':  combine_statements(quarters_data, 'cash_flow_statement'),
            # Canonical (un-renamed) us-gaap facts — the deterministic ratio engine
            # reads these standardized tags instead of per-company display labels.
            'us_gaap_facts':               us_gaap_facts,
        }


    # ============================================================================
    # USAGE
    # ============================================================================

    stock1_results = extract_latest_4_quarters(pair[0], show_filing_times=True)
    stock2_results = extract_latest_4_quarters(pair[1], show_filing_times=True)

    def make_tab(results, ticker):
        if results is None:
            return mo.md(f"❌ Could not load data for {ticker}")
        def tbl(df):
            return mo.ui.table(df) if df is not None else mo.md("_Not available_")
        return mo.vstack([
            mo.md("### Income Statement"),  tbl(results["income_statement_latest"]),
            mo.md("### Balance Sheet"),     tbl(results["balance_sheet_latest"]),
            mo.md("### Cash Flow"),         tbl(results["cash_flow_statement_latest"]),
        ])

    mo.ui.tabs({
        pair[0]: make_tab(stock1_results, pair[0]),
        pair[1]: make_tab(stock2_results, pair[1]),
    })
    return stock1_results, stock2_results


@app.cell
def _(
    Any,
    datetime,
    json,
    ollama,
    pair,
    pd,
    re,
    stock1_results,
    sys,
    textwrap,
):
    """
    analyst_pipeline.py
    ====================
    SEC XBRL → Ollama LLM → Structured JSON → Rich Terminal Report
    """

    # ─────────────────────────────────────────────────────────────────────────────
    # 1.  CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────────

    MODEL             = "gpt-oss:120b-cloud"
    TEMPERATURE       = 0.0
    MAX_OUTPUT_TOKENS = 131072
    MAX_RETRIES       = 3
    SEED              = 42          # fixed seed → reproducible runs
    DEBUG             = True


    # ─────────────────────────────────────────────────────────────────────────────
    # 1b. DETERMINISTIC RATIO ENGINE  (computed in code — never by the LLM)
    # ─────────────────────────────────────────────────────────────────────────────
    #
    # Filers use different us-gaap tags for the same economic concept, so each
    # ratio input maps to a PRIORITY LIST of canonical tags. We try them in order
    # and return None when none are present — e.g. a bank has no AssetsCurrent, so
    # its current_ratio is correctly null rather than fabricated.

    RATIO_ALIASES = {
        "revenue":             ["RevenueFromContractWithCustomerExcludingAssessedTax",
                                "Revenues",
                                "RevenueFromContractWithCustomerIncludingAssessedTax",
                                "SalesRevenueNet", "RevenuesNetOfInterestExpense",
                                "InterestAndDividendIncomeOperating", "PremiumsEarnedNet"],
        "cost_of_revenue":     ["CostOfGoodsAndServicesSold", "CostOfRevenue",
                                "CostOfGoodsSold", "CostOfServices"],
        "gross_profit":        ["GrossProfit"],
        "operating_income":    ["OperatingIncomeLoss",
                                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
        "net_income":          ["NetIncomeLoss", "ProfitLoss"],
        "current_assets":      ["AssetsCurrent"],
        "current_liabilities": ["LiabilitiesCurrent"],
        "inventory":           ["InventoryNet"],
        "total_assets":        ["Assets"],
        "total_liabilities":   ["Liabilities"],
        "equity":              ["StockholdersEquity",
                                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        "interest_expense":    ["InterestExpense", "InterestExpenseDebt",
                                "InterestAndDebtExpense", "InterestExpenseBorrowings",
                                "InterestExpenseNonoperating"],
        "cfo":                 ["NetCashProvidedByUsedInOperatingActivities",
                                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        "capex":               ["PaymentsToAcquirePropertyPlantAndEquipment",
                                "PaymentsToAcquireProductiveAssets", "PaymentsToAcquireRealEstate"],
    }

    # Interest-bearing debt is summed from components; filers split it many ways.
    DEBT_SHORT = ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings",
                  "OtherShortTermBorrowings", "NotesPayableCurrent", "CommercialPaper",
                  "LongTermDebtAndCapitalLeaseObligationsCurrent"]
    DEBT_LONG  = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
                  "LongTermDebt", "LongTermNotesPayable", "OtherLongTermDebtNoncurrent",
                  "ConvertibleDebtNoncurrent"]
    DEBT_TOTAL = ["DebtLongtermAndShorttermCombinedAmount",
                  "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"]

    QUARTER_MIN, QUARTER_MAX = 80, 100      # day-count window for a fiscal quarter


    def _usd_points(facts, tag):
        return ((facts.get(tag) or {}).get("units", {}) or {}).get("USD") or []


    def _duration_days(point):
        s, e = point.get("start"), point.get("end")
        return (datetime.fromisoformat(e) - datetime.fromisoformat(s)).days if s and e else None


    def _resolve_points(facts, concept, flow: bool):
        """Alias tag with the MOST RECENT data (not merely the first one present)."""
        best, best_end = [], ""
        for tag in RATIO_ALIASES.get(concept, []):
            pts = [p for p in _usd_points(facts, tag) if bool(p.get("start")) == flow]
            if pts:
                end = max(p["end"] for p in pts)
                if end > best_end:
                    best, best_end = pts, end
        return best


    def _anchor_period(facts):
        """Latest quarter-end that has a true quarterly (not YTD) flow reported."""
        for concept in ("revenue", "net_income"):
            q = [p for p in _resolve_points(facts, concept, True)
                 if QUARTER_MIN <= (_duration_days(p) or 0) <= QUARTER_MAX]
            if q:
                return max(p["end"] for p in q)
        return None


    def _quarterly_value(facts, concept, period_end):
        """Quarterly value at an EXACT period end - never mixes YTD with quarterly."""
        pts = [p for p in _resolve_points(facts, concept, True)
               if QUARTER_MIN <= (_duration_days(p) or 0) <= QUARTER_MAX
               and p["end"] == period_end]
        return sorted(pts, key=lambda p: p.get("filed", ""))[-1]["val"] if pts else None


    def _instant_value(facts, concept, period_end):
        pts = [p for p in _resolve_points(facts, concept, False) if p["end"] <= period_end]
        return sorted(pts, key=lambda p: (p["end"], p.get("filed", "")))[-1]["val"] if pts else None


    def _common_window(facts, concept_a, concept_b):
        """Latest window where BOTH flows share an identical (start, end).

        10-Q cash flows are cumulative YTD, so a ratio of two flows is only valid
        when both cover the same window. Returns (val_a, val_b, days).
        """
        a = {(p["start"], p["end"]): p for p in
             sorted(_resolve_points(facts, concept_a, True), key=lambda p: p.get("filed", ""))}
        b = {(p["start"], p["end"]): p for p in
             sorted(_resolve_points(facts, concept_b, True), key=lambda p: p.get("filed", ""))}
        shared = sorted(set(a) & set(b), key=lambda w: w[1])
        if not shared:
            return None, None, None
        w = shared[-1]
        return a[w]["val"], b[w]["val"], _duration_days(a[w])


    def _sum_tags(facts, tags, period_end):
        total = None
        for tag in tags:
            pts = [p for p in _usd_points(facts, tag)
                   if not p.get("start") and p["end"] <= period_end]
            if pts:
                total = (total or 0) + sorted(
                    pts, key=lambda p: (p["end"], p.get("filed", "")))[-1]["val"]
        return total


    def _total_debt(facts, period_end, total_liabilities):
        """Interest-bearing debt, or None when it cannot be resolved reliably.

        Some filers (e.g. Ford) tag debt in a CUSTOM extension outside us-gaap, so
        summing us-gaap components yields only a trivial fragment. We return None
        rather than a misleading number - liabilities_to_equity covers leverage.
        """
        total = _sum_tags(facts, DEBT_TOTAL, period_end)
        if total is None:
            long_term = _sum_tags(facts, DEBT_LONG, period_end)
            if long_term is None:
                return None
            total = (_sum_tags(facts, DEBT_SHORT, period_end) or 0) + long_term
        # Plausibility guard: debt under 2% of total liabilities means we almost
        # certainly missed custom-tagged debt.
        if total_liabilities and total < 0.02 * total_liabilities:
            return None
        return total


    def detect_company_profile(facts) -> str:
        """Business archetype - decides which ratios are even meaningful."""
        has = lambda t: t in facts
        if any(has(t) for t in ("PremiumsEarnedNet", "LiabilityForFuturePolicyBenefits")):
            return "insurer"
        if any(has(t) for t in ("Deposits", "DepositsLiabilities",
                                "InterestAndDividendIncomeOperating")):
            return "bank"
        if any(has(t) for t in ("RealEstateInvestmentPropertyNet",
                                "RealEstateInvestmentPropertyAtCost")):
            return "reit"
        if has("AssetsCurrent") and has("LiabilitiesCurrent"):
            return "commercial"
        return "general"


    def compute_ratios(facts: dict) -> dict:
        """Key ratios computed deterministically from canonical us-gaap facts.

        Works across company types: a bank has no current ratio, a REIT no gross
        margin - those come back null instead of fabricated. Returns the
        {name: {value, unit, interpretation}} shape the renderers expect.
        """
        if not facts:
            return {}
        period_end = _anchor_period(facts)
        if not period_end:
            return {}

        rev = _quarterly_value(facts, "revenue", period_end)
        oi = _quarterly_value(facts, "operating_income", period_end)
        ni = _quarterly_value(facts, "net_income", period_end)
        gp = _quarterly_value(facts, "gross_profit", period_end)
        if gp is None:                                  # derive when untagged
            cor = _quarterly_value(facts, "cost_of_revenue", period_end)
            gp = (rev - cor) if (rev is not None and cor is not None) else None

        ca = _instant_value(facts, "current_assets", period_end)
        cl = _instant_value(facts, "current_liabilities", period_end)
        inv = _instant_value(facts, "inventory", period_end)
        ta = _instant_value(facts, "total_assets", period_end)
        tl = _instant_value(facts, "total_liabilities", period_end)
        eq = _instant_value(facts, "equity", period_end)
        if tl is None and ta is not None and eq is not None:
            tl = ta - eq        # many filers (e.g. Walmart) never tag Liabilities
        debt = _total_debt(facts, period_end, tl)

        cfo, capex, fcf_days = _common_window(facts, "cfo", "capex")
        oi_w, ie_w, _ = _common_window(facts, "operating_income", "interest_expense")

        div = lambda a, b: round(a / b, 4) if (a is not None and b) else None
        pct = lambda x: round(x * 100, 2) if x is not None else None
        quick_num = (ca - inv) if (ca is not None and inv is not None) else ca

        raw = {
            "current_ratio":         (div(ca, cl),        "x"),
            "quick_ratio":           (div(quick_num, cl), "x"),
            "gross_margin":          (pct(div(gp, rev)),  "%"),
            "operating_margin":      (pct(div(oi, rev)),  "%"),
            "net_margin":            (pct(div(ni, rev)),  "%"),
            "return_on_assets":      (pct(div(ni, ta)),   "%"),
            "return_on_equity":      (pct(div(ni, eq)),   "%"),
            "debt_to_equity":        (div(debt, eq),      "x"),
            "liabilities_to_equity": (div(tl, eq),        "x"),
            "interest_coverage":     (div(oi_w, ie_w),    "x"),
            "asset_turnover":        (div(rev, ta),       "x"),
            "free_cash_flow":        ((cfo - capex) if (cfo is not None and capex is not None)
                                      else None,          "$"),
            "fcf_period_days":       (fcf_days,           "days"),
        }
        return {name: {"value": v, "unit": u, "interpretation": None}
                for name, (v, u) in raw.items()}


    def compute_trends(facts: dict) -> dict:
        """Quarter-over-quarter and year-over-year growth, computed in code."""
        if not facts:
            return {}

        def series(concept):
            pts = [p for p in _resolve_points(facts, concept, True)
                   if QUARTER_MIN <= (_duration_days(p) or 0) <= QUARTER_MAX]
            by_end = {}
            for p in sorted(pts, key=lambda x: x.get("filed", "")):
                by_end[p["end"]] = p["val"]     # latest restatement wins
            return sorted(by_end.items())

        rev, ni = series("revenue"), series("net_income")

        def growth(s, back):
            if len(s) < back + 1:
                return None
            cur, prior = s[-1][1], s[-1 - back][1]
            return round((cur - prior) / abs(prior) * 100, 2) if prior else None

        rev_map, ni_map = dict(rev), dict(ni)

        def margin_at(idx):
            if len(rev) < abs(idx):
                return None
            end = rev[idx][0]
            r, n = rev_map.get(end), ni_map.get(end)
            return (n / r * 100) if (r and n is not None) else None

        cur_m = margin_at(-1)
        prior_m = margin_at(-5) if len(rev) >= 5 else None

        return {
            "period_end":           (rev[-1][0] if rev else None),
            "company_profile":      detect_company_profile(facts),
            "revenue_qoq_pct":      growth(rev, 1),
            "revenue_yoy_pct":      growth(rev, 4),
            "net_income_qoq_pct":   growth(ni, 1),
            "net_income_yoy_pct":   growth(ni, 4),
            "net_margin_delta_pp":  (round(cur_m - prior_m, 2)
                                     if (cur_m is not None and prior_m is not None) else None),
            "quarters_available":   len(rev),
        }


    # ─────────────────────────────────────────────────────────────────────────────
    # 2.  SYSTEM PROMPT
    # ─────────────────────────────────────────────────────────────────────────────

    SYSTEM_PROMPT = """\
    You are a senior Wall Street equity analyst with 20+ years of experience
    and deep expertise in listed options and fundamental analysis.

    You will receive structured financial data extracted from SEC XBRL filings.
    The data may include balance sheet, income statement, and cash flow facts
    across one or more reporting periods.

    STRICT RULES:
    - Labels may vary across companies — infer meaning from context.
    - Do NOT assume row order or column meaning.
    - Do NOT fabricate or estimate missing values; mark them null.
    - Only use information explicitly present in the supplied data.
    - All monetary values should be normalised to millions USD.

    YOUR TASK:
    1. Assess overall financial health with a confidence rating.
    2. Do NOT compute, recompute, or adjust any ratio. The input contains a
       "precomputed_ratios" object whose values were calculated in code from the
       canonical XBRL tags and are AUTHORITATIVE. For "key_ratios", copy each
       name, value and unit VERBATIM from precomputed_ratios — including nulls
       (a null means the concept is not filed for this business model, e.g. a
       bank has no current ratio; leave it null and do not estimate it). Your
       ONLY job for ratios is to write the short "interpretation" string.
    3. Describe trends using the "precomputed_trends" object (growth rates and
       margin deltas already calculated in code). Do NOT compute your own growth
       percentages — narrate direction, magnitude and what they imply.
       "precomputed_trends.company_profile" tells you the business archetype
       (commercial | bank | insurer | reit | general). Tailor the analysis to it:
       a bank or REIT legitimately has NO current ratio or gross margin, so treat
       those nulls as "not applicable to this business model" rather than as
       missing data or a red flag. Note that "fcf_period_days" states the window
       free_cash_flow covers (10-Q cash flows are year-to-date, not quarterly).
    4. Write a detailed analyst-grade narrative covering:
       - Revenue and earnings trajectory
       - Margin profile and direction
       - Balance sheet strength (liquidity, solvency)
       - Cash flow quality and capital allocation
       - Any red flags or standout positives
       - Overall investment thesis in one sentence
    5. List key risks that could derail the thesis.
    6. Recommend exactly ONE best listed-option strategy for the next quarter.
       Choose the single strategy that best fits the fundamental outlook and
       implied-volatility environment. Explain in detail why this strategy was
       chosen over alternatives (e.g. why not a simple long call, why not a
       put spread, etc.).

    OUTPUT FORMAT — CRITICAL:
    - Respond ONLY with a single valid JSON object.
    - Do NOT include any text before or after the JSON.
    - Do NOT use markdown code fences (no ```json or ```).
    - Do NOT include comments inside the JSON.
    - Every string value must use double quotes.
    - Use null (not "null", not None) for missing values.

    JSON schema:
    {
      "company": "<ticker or name if identifiable, else null>",
      "report_date": "<YYYY-MM-DD of most recent period, else null>",

      "financial_health": {
        "rating": "Strong | Moderate | Weak",
        "confidence": "High | Medium | Low",
        "summary": "<2-3 sentence headline rationale>"
      },

      "key_ratios": {
        "<ratio_name>": {
          "value": 0.0,
          "unit": "x | % | $ | days | ratio",
          "interpretation": "<short phrase>"
        }
      },

      "trends": ["<concise bullet covering direction and magnitude>"],

      "analysis": {
        "revenue_and_earnings": "<2-3 sentences>",
        "margins":              "<2-3 sentences>",
        "balance_sheet":        "<2-3 sentences>",
        "cash_flow":            "<2-3 sentences>",
        "red_flags":            "<2-3 sentences, or null if none>",
        "investment_thesis":    "<single punchy sentence>"
      },

      "risks": ["<bullet>"],

      "option_strategy": {
        "bias": "Bullish | Bearish | Neutral | Hedge",
        "name": "<strategy name>",
        "structure": "<specific legs, strikes, expiry target>",
        "max_profit": "<description>",
        "max_loss": "<description>",
        "ideal_iv_environment": "rising | falling | elevated | low",
        "rationale": "<2-3 sentences grounded in the analysis above>",
        "why_best": "<detailed explanation of why this strategy beats alternatives such as long calls, put spreads, straddles, etc.>"
      }
    }
    """


    # ─────────────────────────────────────────────────────────────────────────────
    # 3.  DATA BUILDER
    # ─────────────────────────────────────────────────────────────────────────────

    def statement_to_dict(df: pd.DataFrame) -> dict:
        if df is None:
            return {}

        """DataFrame (concepts × periods) → nested plain dict, NaNs dropped."""
        result = {}
        for concept in df.index:
            values = df.loc[concept].dropna()
            if len(values) > 0:
                result[str(concept)] = {
                    str(period): float(value)
                    for period, value in values.items()
                }
        return result


    def build_financial_package(stock_results: dict) -> dict:
        return {
            "balance_sheet":       statement_to_dict(stock_results["balance_sheet_latest"]),
            "income_statement":    statement_to_dict(stock_results["income_statement_latest"]),
            "cash_flow_statement": statement_to_dict(stock_results["cash_flow_statement_latest"]),
            # Authoritative, code-computed ratios. The LLM must copy these values
            # verbatim and only add interpretations (see SYSTEM_PROMPT rule 2).
            # Empty when facts are unavailable (e.g. the yfinance fallback path).
            "precomputed_ratios":  compute_ratios(stock_results.get("us_gaap_facts", {})),
            # Code-computed growth/trend metrics + the company archetype, so the
            # LLM narrates direction instead of doing its own arithmetic.
            "precomputed_trends":  compute_trends(stock_results.get("us_gaap_facts", {})),
        }


    # ─────────────────────────────────────────────────────────────────────────────
    # 4.  JSON EXTRACTION HELPERS
    # ─────────────────────────────────────────────────────────────────────────────

    def _extract_text(resp: Any) -> str:
        if hasattr(resp, "response"):
            return resp.response or ""
        if isinstance(resp, dict):
            return (
                resp.get("response")
                or (resp.get("message") or {}).get("content")
                or ""
            )
        return str(resp)

    def _escape_interior_quotes(text: str) -> str:
        """
        Scan a JSON-ish string and escape any bare double-quotes that appear
        *inside* a string value (i.e. not at the boundaries of a key or value).
        Strategy: tokenise into JSON structural chars vs string content and
        re-escape any " found inside a string that isn't already escaped.
        """
        result = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]

            # ── Enter a JSON string ────────────────────────────────────────────
            if ch == '"':
                result.append('"')
                i += 1
                # consume until we hit an unescaped closing "
                while i < n:
                    c = text[i]
                    if c == '\\':
                        # keep escape sequence intact
                        result.append(c)
                        i += 1
                        if i < n:
                            result.append(text[i])
                            i += 1
                    elif c == '"':
                        # closing quote — but is what follows structural or content?
                        # look ahead: skip whitespace, then check for :  ,  }  ]
                        j = i + 1
                        while j < n and text[j] in ' \t\r\n':
                            j += 1
                        if j >= n or text[j] in ':,}]':
                            # legitimate closing quote
                            result.append('"')
                            i += 1
                            break
                        else:
                            # interior quote — escape it
                            result.append('\\"')
                            i += 1
                    else:
                        result.append(c)
                        i += 1
            else:
                result.append(ch)
                i += 1

        return "".join(result)


    def _scrub_json(raw: str) -> str:
        text = raw.lstrip("\ufeff\u200b\u200c\u200d")

        # Normalize Unicode lookalikes
        text = text.replace("\u201c", '"').replace("\u201d", '"')   # " "  → "
        text = text.replace("\u2018", "'").replace("\u2019", "'")   # ' '  → '
        text = text.replace("\u2011", "-").replace("\u2010", "-")
        text = text.replace("\u2012", "-").replace("\u2013", "-")
        text = text.replace("\u2014", "-")
        text = text.replace("\u00ad", "-")
        text = text.replace("\u2026", "...")

        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```", "", text)

        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return text.strip()
        text = text[start : end + 1]

        text = re.sub(r"//[^\n]*", "", text)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        text = re.sub(r"(?<![\\\"'])'([^'\\]*)'(?![\\\"'])", r'"\1"', text)

        # ── Re-escape any interior bare double-quotes ──────────────────────────
        text = _escape_interior_quotes(text)

        return text.strip()


    def _try_parse(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            pos  = exc.pos
            snip = text[max(0, pos - 40) : pos + 40].replace("\n", "↵")
            raise json.JSONDecodeError(
                f"{exc.msg} — near: …{snip}…", exc.doc, exc.pos
            ) from None


    # ─────────────────────────────────────────────────────────────────────────────
    # 5.  LLM CALL
    # ─────────────────────────────────────────────────────────────────────────────

    def ask_ollama(
        financial_package: dict,
        question: str,
        model: str             = MODEL,
        temperature: float     = TEMPERATURE,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        retries: int           = MAX_RETRIES,
    ) -> dict:
        prompt = (
            f"Financial Data:\n{json.dumps(financial_package, indent=2)}\n\n"
            f"Question:\n{question}"
        )

        last_error: Exception = RuntimeError("No attempts made")
        last_raw:   str       = ""

        for attempt in range(1, retries + 1):
            print(f"  ⏳ Attempt {attempt}/{retries}  (temp={temperature:.1f}) …")
            try:
                resp = ollama.generate(
                    model=model,
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    stream=False,
                    format="json",          # constrain output to valid JSON
                    options={
                        "temperature": temperature,
                        "num_predict": max_output_tokens,
                        "seed": SEED,       # deterministic across runs
                    },
                )

                raw      = _extract_text(resp)
                last_raw = raw

                if DEBUG:
                    preview = raw[:300].replace("\n", " ")
                    print(f"  [DEBUG] raw preview → {preview!r}")

                clean  = _scrub_json(raw)
                parsed = _try_parse(clean)

                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected a JSON object, got {type(parsed)}")

                return parsed

            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                # NOTE: temperature is deliberately NOT raised on retry — bumping
                # it made every retry less consistent, which is the opposite of
                # what we want. format="json" is what fixes malformed output.
                last_error = exc
                print(f"  ⚠  Parse error on attempt {attempt}: {exc}")

            except Exception as exc:
                last_error = exc
                print(f"  ✗  Ollama error on attempt {attempt}: {exc}")
                break

        raise ValueError(
            f"ask_ollama failed after {retries} attempts.\n"
            f"Last error : {last_error}\n"
            f"Last raw   :\n{last_raw}"
        )


    # ─────────────────────────────────────────────────────────────────────────────
    # 6.  RICH TERMINAL RENDERER
    # ─────────────────────────────────────────────────────────────────────────────

    class C:
        RESET  = "\033[0m"
        BOLD   = "\033[1m"
        DIM    = "\033[2m"
        GREEN  = "\033[92m"
        YELLOW = "\033[93m"
        RED    = "\033[91m"
        BLUE   = "\033[94m"
        CYAN   = "\033[96m"
        WHITE  = "\033[97m"


    WIDTH = 68


    def _rule(char: str = "─") -> str:
        return char * WIDTH


    def _section(title: str) -> None:
        print(f"\n{C.BOLD}{title}{C.RESET}")
        print(f"{C.DIM}{_rule()}{C.RESET}")


    def _wrap(text: str, indent: int = 4) -> str:
        prefix = " " * indent
        return textwrap.fill(
            str(text), width=WIDTH, initial_indent=prefix, subsequent_indent=prefix
        )


    _HEALTH_COLOR = {"Strong": C.GREEN, "Moderate": C.YELLOW, "Weak": C.RED}
    _BIAS_COLOR   = {"Bullish": C.GREEN, "Bearish": C.RED,
                     "Neutral": C.YELLOW, "Hedge": C.BLUE}
    _IV_ICON      = {"rising": "↑ rising", "falling": "↓ falling",
                     "elevated": "▲ elevated", "low": "▼ low"}


    def print_analyst_report(result: dict, ticker: str = None) -> None:
        company = ticker.upper() if ticker else (result.get("company") or "Unknown Company")
        report_date = result.get("report_date") or datetime.today().strftime("%Y-%m-%d")
        generated   = datetime.now().strftime("%Y-%m-%d %H:%M")

        print(f"\n{C.BOLD}{C.WHITE}{'═' * WIDTH}{C.RESET}")
        print(f"{C.BOLD}{C.WHITE}  EQUITY ANALYST REPORT  ·  {company}{C.RESET}")
        print(f"{C.BOLD}{C.WHITE}{'═' * WIDTH}{C.RESET}")
        print(f"{C.DIM}  Period: {report_date}   Generated: {generated}{C.RESET}")

        # ── Financial health ──────────────────────────────────────────────────────
        health  = result.get("financial_health") or {}
        rating  = health.get("rating", "N/A")
        conf    = health.get("confidence", "N/A")
        summary = health.get("summary", "")
        hc      = _HEALTH_COLOR.get(rating, C.RESET)

        _section("Financial Health")
        print(f"  Rating     : {hc}{C.BOLD}{rating}{C.RESET}   "
              f"{C.DIM}(Confidence: {conf}){C.RESET}")
        if summary:
            print(_wrap(summary))

        # ── Key ratios ────────────────────────────────────────────────────────────
        ratios = result.get("key_ratios") or {}
        if ratios:
            _section("Key Ratios")
            col_w = 28
            for name, data in ratios.items():
                if not isinstance(data, dict):
                    continue
                val    = data.get("value")
                unit   = data.get("unit", "")
                interp = data.get("interpretation", "")

                if val is None:
                    val_str = f"{C.DIM}n/a{C.RESET}"
                elif unit == "%":
                    val_str = f"{float(val):.1f}{unit}"
                elif unit in ("x", "ratio", "days"):
                    val_str = f"{float(val):.2f} {unit}"
                else:
                    val_str = f"{float(val):,.1f}"

                note = f"  {C.DIM}— {interp}{C.RESET}" if interp else ""
                print(f"  {name:<{col_w}} {C.BOLD}{val_str}{C.RESET}{note}")

        # ── Trends ────────────────────────────────────────────────────────────────
        trends = result.get("trends") or []
        if trends:
            _section("Trends")
            for t in trends:
                print(_wrap(f"• {t}"))

        # ── Detailed analysis ─────────────────────────────────────────────────────
        analysis = result.get("analysis") or {}
        if analysis:
            _section("Detailed Analysis")

            subsections = [
                ("Revenue & Earnings", "revenue_and_earnings"),
                ("Margins",            "margins"),
                ("Balance Sheet",      "balance_sheet"),
                ("Cash Flow",          "cash_flow"),
                ("Red Flags",          "red_flags"),
            ]

            for label, key in subsections:
                text = analysis.get(key)
                if not text:
                    continue
                print(f"\n  {C.BOLD}{label}{C.RESET}")
                print(_wrap(text))

            thesis = analysis.get("investment_thesis")
            if thesis:
                print(f"\n  {C.BOLD}{C.CYAN}▶  {thesis}{C.RESET}")

        # ── Risks ─────────────────────────────────────────────────────────────────
        risks = result.get("risks") or []
        if risks:
            _section("Key Risks")
            for r in risks:
                print(_wrap(f"⚠  {r}"))

        # ── Best option strategy ──────────────────────────────────────────────────
        s = result.get("option_strategy") or {}
        if s:
            _section("Best Option Strategy  (next quarter)")
            bias     = s.get("bias", "")
            bc       = _BIAS_COLOR.get(bias, C.RESET)
            iv_raw   = (s.get("ideal_iv_environment") or "").lower()
            iv_label = _IV_ICON.get(iv_raw, iv_raw or "n/a")

            print(f"\n  {C.BOLD}{s.get('name', '')}{C.RESET}  [{bc}{bias}{C.RESET}]")
            print(f"     {C.DIM}Structure      :{C.RESET} {s.get('structure', '')}")
            print(f"     {C.DIM}Max profit     :{C.RESET} {s.get('max_profit') or 'n/a'}")
            print(f"     {C.DIM}Max loss       :{C.RESET} {s.get('max_loss') or 'n/a'}")
            print(f"     {C.DIM}IV environment :{C.RESET} {iv_label}")

            if s.get("rationale"):
                print(f"\n  {C.BOLD}Rationale{C.RESET}")
                print(_wrap(s["rationale"]))

            if s.get("why_best"):
                print(f"\n  {C.BOLD}Why this over alternatives{C.RESET}")
                print(_wrap(s["why_best"]))

        print(f"\n{C.DIM}{_rule()}{C.RESET}\n")


    # ─────────────────────────────────────────────────────────────────────────────
    # 7.  OPTIONAL: SAVE RAW JSON
    # ─────────────────────────────────────────────────────────────────────────────

    def save_report(result: dict, path: str = "analyst_report.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  ✓ JSON saved → {path}")


    # ─────────────────────────────────────────────────────────────────────────────
    # 8.  MAIN PIPELINE FUNCTION
    # ─────────────────────────────────────────────────────────────────────────────

    def run_analysis(stock_results: dict,
                     ticker: str     = None,
                     save_json: bool = False,
                     filename: str   = "analyst_report.json") -> dict:

        print("\n  ⏳ Building financial package …")
        package = build_financial_package(stock_results)

        if ticker:
            package["company_ticker"] = ticker.upper()

        question = (
            "Assess the company's overall financial health in detail. "
            "Identify key trends and risks across all available periods. "
            "Recommend exactly ONE best listed-option strategy for next quarter "
            "and explain why it beats alternatives."
        )

        try:
            result = ask_ollama(package, question)
        except ValueError as exc:
            print(f"\n  ✗ Analysis failed:\n{exc}", file=sys.stderr)
            return {}

        # force company name — don't trust the LLM to identify it
        result["company"] = ticker.upper() if ticker else (result.get("company") or "Unknown Company")

        # Authoritative override: ratio VALUES always come from code, never the
        # LLM — we keep only its interpretation text. This guarantees the numbers
        # shown match the filings even if the model ignores SYSTEM_PROMPT rule 2.
        computed = package.get("precomputed_ratios") or {}
        if computed:
            llm_ratios = result.get("key_ratios") or {}
            result["key_ratios"] = {
                name: {
                    "value":          data["value"],
                    "unit":           data["unit"],
                    "interpretation": (llm_ratios.get(name) or {}).get("interpretation"),
                }
                for name, data in computed.items()
            }

        print_analyst_report(result, ticker=ticker)   # ← pass ticker

        if save_json:
            save_report(result, path=filename)

        return result

    # ─────────────────────────────────────────────────────────────────────────────
    # 9.  ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────────

    if __name__ == "__main__":
        result1 = run_analysis(stock1_results, ticker = pair[0], save_json=True, filename="data/report_stock1.json")
    return result1, run_analysis


@app.cell
def _(pair, run_analysis, stock2_results):
    result2 = run_analysis(stock2_results, ticker = pair[1], save_json=True, filename="data/report_stock2.json")
    return (result2,)


@app.cell
def _(page_header):
    page_header("AI Analysis", "02")
    return


@app.cell
def _(mo, pair, pd, result1, result2):
    def render_report(result: dict) -> mo.Html:
        if not result:
            return mo.callout(mo.md("No result available"), kind="warn")

        company     = result.get("company", "Unknown")
        report_date = result.get("report_date", "")

        # ── Financial Health ──────────────────────────────────────────────
        health  = result.get("financial_health", {})
        rating  = health.get("rating", "N/A")
        rating_color = {"Strong": "success", "Moderate": "warn", "Weak": "danger"}.get(rating, "info")

        # ── Key Ratios table ──────────────────────────────────────────────
        ratios = result.get("key_ratios", {})
        ratio_rows = [
            {"Ratio": name, "Value": d.get("value"), "Unit": d.get("unit",""), "Interpretation": d.get("interpretation","")}
            for name, d in ratios.items() if isinstance(d, dict)
        ]

        # ── Trends ───────────────────────────────────────────────────────
        trends = result.get("trends", [])

        # ── Analysis ─────────────────────────────────────────────────────
        analysis = result.get("analysis", {})

        # ── Risks ────────────────────────────────────────────────────────
        risks = result.get("risks", [])

        # ── Option Strategy ───────────────────────────────────────────────
        s    = result.get("option_strategy", {})
        bias = s.get("bias", "")
        bias_color = {"Bullish": "success", "Bearish": "danger", "Neutral": "warn", "Hedge": "info"}.get(bias, "info")

        return mo.vstack([
            mo.md(f"# {company}  ·  {report_date}"),

            mo.callout(mo.md(f"**{rating}** — {health.get('summary','')}"), kind=rating_color),

            mo.md("## Key Ratios"),
            mo.ui.table(pd.DataFrame(ratio_rows)),

            mo.md("## Trends"),
            mo.vstack([mo.md(f"- {t}") for t in trends]),

            mo.md("## Analysis"),
            mo.vstack([
                mo.md(f"**Revenue & Earnings:** {analysis.get('revenue_and_earnings','')}"),
                mo.md(f"**Margins:** {analysis.get('margins','')}"),
                mo.md(f"**Balance Sheet:** {analysis.get('balance_sheet','')}"),
                mo.md(f"**Cash Flow:** {analysis.get('cash_flow','')}"),
                mo.md(f"**Red Flags:** {analysis.get('red_flags','')}"),
                mo.callout(mo.md(f"▶ {analysis.get('investment_thesis','')}"), kind="info"),
            ]),

            mo.md("## Key Risks"),
            mo.vstack([mo.md(f"- {r}") for r in risks]),

            mo.md("## Option Strategy"),
            mo.callout(mo.md(
                f"**{s.get('name','')}** [{bias}]\n\n"
                f"- Structure: {s.get('structure','')}\n"
                f"- Max Profit: {s.get('max_profit','')}\n"
                f"- Max Loss: {s.get('max_loss','')}\n"
                f"- IV Environment: {s.get('ideal_iv_environment','')}\n\n"
                f"**Rationale:** {s.get('rationale','')}\n\n"
                f"**Why best:** {s.get('why_best','')}"
            ), kind=bias_color),
        ])


    tabs = mo.ui.tabs({
        pair[0]: render_report(result1),
        pair[1]: render_report(result2),
    })
    return (tabs,)


@app.cell
def _(mo, tabs):
    import pathlib
    pathlib.Path(".fundamental_done").touch()

    mo.vstack([
        tabs,
        mo.md("✅ Fundamental analysis complete.")
    ])
    return


if __name__ == "__main__":
    app.run()
