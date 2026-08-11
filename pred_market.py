import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    css_file="theme.css",
    html_head_file="theme_head.html",
)


@app.cell
def _():
    # cell 1 - imports & setup
    import marimo as mo
    import asyncio
    import threading
    import json
    import jsonschema
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from contextlib import AsyncExitStack
    from ollama import chat
    import traceback

    MODEL = "gemma4:cloud"

    # The one profile this notebook is allowed to touch. Every tool exposes an
    # `account` param defaulting to "default"; the model must never pick it, or a
    # hallucinated name silently creates a new empty profile instead of erroring.
    ACCOUNT = "default"

    server_params = StdioServerParameters(
        command=r"C:\Users\buitu\AppData\Roaming\Python\Python311\Scripts\pm-trader-mcp.exe"
    )

    def mcp_tool_to_ollama(tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }

    return (
        ACCOUNT,
        AsyncExitStack,
        ClientSession,
        MODEL,
        asyncio,
        chat,
        json,
        jsonschema,
        mcp_tool_to_ollama,
        mo,
        server_params,
        stdio_client,
        threading,
    )


@app.cell
def _(mo):
    # Display-only house hero, matching the other notebooks. Fonts and the
    # Tabler icon webfont are loaded globally via theme_head.html.
    mo.Html("""
    <div style="padding: 2.5rem 0 2rem; text-align: center;">
      <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; border: 0.5px solid #1D9E75; margin-bottom: 1rem;">
        <i class="ti ti-crystal-ball" style="font-size: 22px; color: #1D9E75;"></i>
      </div>
      <h1 style="font-family: 'DM Serif Display', serif; font-size: 36px; font-weight: 400; font-style: italic; margin: 0 0 6px; letter-spacing: -0.01em; color: var(--color-text-primary);">Prediction Market Strategy</h1>
      <p style="font-family: 'DM Mono', monospace; font-size: 11px; color: var(--color-text-secondary); letter-spacing: 0.18em; text-transform: uppercase; margin: 0 0 1.25rem;">Polymarket paper desk &middot; LLM broker</p>
      <div style="display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: 11px; color: #0F6E56; background: #E1F5EE; padding: 4px 14px; border-radius: 999px;">
        <span style="width: 6px; height: 6px; border-radius: 50%; background: #1D9E75; display: inline-block;"></span>
        Live
      </div>
      <div style="margin-top: 1.5rem; width: 40px; height: 0.5px; background: var(--color-border-tertiary); margin-left: auto; margin-right: auto;"></div>
    </div>
    """)
    return


@app.cell
def _(
    AsyncExitStack,
    ClientSession,
    asyncio,
    mcp_tool_to_ollama,
    server_params,
    stdio_client,
    threading,
):
    # cell 2 - persistent MCP actor thread (survives Marimo cell re-runs)
    class MCPManager:
        def __init__(self, server_params):
            self.server_params = server_params
            self._loop: asyncio.AbstractEventLoop | None = None
            self._thread: threading.Thread | None = None
            self._session = None
            self._ready = threading.Event()
            self._stop_event: asyncio.Event | None = None
            self._error: Exception | None = None
            self.tools = []
            self.tools_by_name = {}

        def start(self):
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            if not self._ready.wait(timeout=30):
                raise RuntimeError("MCP server did not become ready in time")
            if self._error:
                raise self._error

        def _run_loop(self):
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._main())
            except Exception as e:
                self._error = e
                self._ready.set()  # unblock start() so it can raise

        async def _main(self):
            self._stop_event = asyncio.Event()
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(self.server_params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                self.tools = (await session.list_tools()).tools
                self.tools_by_name = {t.name: t for t in self.tools}
                self._ready.set()
                await self._stop_event.wait()

        async def call_tool_async(self, name, args, timeout=60):
            fut = asyncio.run_coroutine_threadsafe(
                self._session.call_tool(name, args), self._loop
            )
            return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=timeout)

        def is_alive(self) -> bool:
            return bool(self._thread and self._thread.is_alive())

        def stop(self):
            if self._loop and self._stop_event:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            if self._thread:
                self._thread.join(timeout=5)

    try:
        mcp_manager.stop()
    except NameError:
        pass

    mcp_manager = MCPManager(server_params)
    mcp_manager.start()
    ollama_tools = [mcp_tool_to_ollama(t) for t in mcp_manager.tools]
    print(f"[mcp] connected, {len(ollama_tools)} tools loaded: {[t.name for t in mcp_manager.tools]}")
    return mcp_manager, ollama_tools


@app.cell
def _(
    ACCOUNT,
    MODEL,
    asyncio,
    chat,
    json,
    jsonschema,
    mcp_manager,
    ollama_tools,
):
    # cell 3 - model function for mo.ui.chat

    SYSTEM_PROMPT = {
        "role": "system",
        "content": (
            "You are a trading assistant with access to live Polymarket tools via MCP. "
            "CRITICAL RULES:\n"
            "1. NEVER invent, guess, retype, or round market data, prices, IDs, or volumes. "
            "A verified data table will be shown to the user automatically — you do not "
            "need to reproduce numbers yourself. Refer to markets by name/slug only.\n"
            "2. If a tool result is prefixed with 'TOOL ERROR', 'TOOL TIMEOUT', or "
            "'INVALID ARGUMENTS', tell the user the call failed and why — do not proceed "
            "as if you have data.\n"
            "3. Only call tools with arguments matching their actual schema. Do not assume "
            "a tool supports parameters (like sorting or filtering) unless its schema says so.\n"
            "4. If no available tool can do what the user asked, say so plainly instead of "
            "fabricating output.\n"
            "5. You MUST obtain any market, price, volume, or account figure by calling a "
            "tool in THIS conversation. Your training data is stale and Polymarket markets "
            "change constantly — markets you 'remember' (e.g. 2024 election markets) are gone. "
            "NEVER answer a data question from memory; if you haven't called a tool this turn, "
            "call one before answering."
        ),
    }


    # ---------- deterministic, non-LLM data handling ----------

    def _try_parse_json(text: str):
        """Best-effort JSON parse of a tool's raw text content. Returns None if not JSON."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None


    def _fmt_num(v):
        """Format a number preserving real precision -- never rounds, never invents."""
        if isinstance(v, float):
            # keep full precision, strip trailing zeros but don't round the value
            s = f"{v:.6f}".rstrip("0").rstrip(".")
            return s if s else "0"
        return str(v)


    def _json_to_markdown_table(parsed) -> str | None:
        """
        Deterministically render tool JSON as a markdown table.
        Handles the common shapes: {"ok":.., "data":[...]} , a bare list[dict], or a dict.
        Returns None if the shape isn't tabular (caller falls back to raw text).
        """
        rows = None
        if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
            rows = parsed["data"]
        elif isinstance(parsed, list):
            rows = parsed
        elif isinstance(parsed, dict):
            rows = [parsed]

        if not rows or not all(isinstance(r, dict) for r in rows):
            return None

        # union of keys across rows, in first-seen order, capped so tables stay readable
        cols = []
        for r in rows:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        # prioritize the most useful columns if present, keep the rest after
        preferred = [c for c in ("question", "slug", "outcomes", "outcome_prices",
                                  "volume", "liquidity", "closed", "end_date") if c in cols]
        rest = [c for c in cols if c not in preferred]
        cols = preferred + rest

        def cell(v):
            if isinstance(v, float):
                return _fmt_num(v)
            if isinstance(v, list):
                return ", ".join(_fmt_num(x) if isinstance(x, float) else str(x) for x in v)
            if v is None:
                return ""
            return str(v)

        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = "\n".join(
            "| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |"
            for r in rows
        )
        return "\n".join([header, sep, body])


    def _extract_content(result):
        """
        Pull text out of an MCP CallToolResult.
        Returns (content_str_for_llm, verified_markdown_table_or_None).
        The markdown table is built directly from raw JSON -- the LLM never touches it.
        """
        parts = []
        for block in result.content:
            btype = getattr(block, "type", None)
            if hasattr(block, "text"):
                parts.append(block.text)
            elif btype:
                parts.append(f"[unsupported content type from tool: {btype}]")
            else:
                parts.append(f"[unrecognized content block: {block!r}]")
        raw_text = "\n".join(parts) if parts else "[tool returned no content]"

        if getattr(result, "isError", False):
            return f"TOOL ERROR: {raw_text}", None

        parsed = _try_parse_json(raw_text)
        table = _json_to_markdown_table(parsed) if parsed is not None else None
        return raw_text, table


    def _validate_args(tool, args: dict) -> str | None:
        schema = getattr(tool, "inputSchema", None)
        if not schema:
            return None
        try:
            jsonschema.validate(instance=args, schema=schema)
            return None
        except jsonschema.ValidationError as e:
            return f"INVALID ARGUMENTS for {tool.name}: {e.message}"


    def _is_error_content(content: str) -> bool:
        return content.startswith(("TOOL ERROR", "TOOL TIMEOUT", "INVALID ARGUMENTS"))


    async def pm_trader_agent(messages, config):
        history = [SYSTEM_PROMPT] + [{"role": m.role, "content": m.content} for m in messages]

        max_tool_hops = 8
        last_tool_had_error = False
        last_verified_table = None   # <- ground truth, built in code, never touched by the LLM
        last_verified_source = None

        for hop in range(max_tool_hops):
            if not mcp_manager.is_alive():
                return "[error: MCP server connection has died — restart the notebook/kernel]"

            resp = await asyncio.to_thread(
                chat,
                model=MODEL,
                messages=history,
                tools=ollama_tools,
                options={"temperature": 0},
            )
            msg = resp.message
            history.append(msg)

            if not msg.tool_calls:
                final = msg.content or ""

                if last_tool_had_error and not any(
                    w in final.lower() for w in ("error", "fail", "couldn't", "could not", "unable")
                ):
                    print(f"[WARNING] previous tool call errored but final reply doesn't acknowledge it: {final[:300]}")

                # Always append the deterministic, code-generated table -- this is the
                # part of the answer that is guaranteed to match the tool's raw output.
                if last_verified_table:
                    final = (
                        f"{final}\n\n"
                        f"**Verified data from `{last_verified_source}`** "
                        f"(rendered directly from the tool result, not retyped by the model):\n\n"
                        f"{last_verified_table}"
                    )
                return final

            last_tool_had_error = False

            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass

                tool = mcp_manager.tools_by_name.get(name)

                # Pin the account. A hallucinated name here doesn't error -- the
                # server creates that profile on demand, so a wrong value reads an
                # empty account (or worse, trades/resets into a phantom one).
                if tool is not None and isinstance(args, dict):
                    props = (getattr(tool, "inputSchema", None) or {}).get("properties") or {}
                    if "account" in props:
                        if args.get("account") not in (None, ACCOUNT):
                            print(f"[guard] overriding account={args['account']!r} -> {ACCOUNT!r}")
                        args["account"] = ACCOUNT

                if tool is None:
                    content = f"TOOL ERROR: no such tool '{name}'. Available tools: {list(mcp_manager.tools_by_name)}"
                    table = None
                else:
                    validation_error = _validate_args(tool, args)
                    if validation_error:
                        content, table = validation_error, None
                    else:
                        try:
                            result = await mcp_manager.call_tool_async(name, args)
                            content, table = _extract_content(result)
                        except asyncio.TimeoutError:
                            content = (
                                f"TOOL TIMEOUT: {name} did not respond in time. "
                                f"No data was retrieved — do not guess or fabricate a value."
                            )
                            table = None
                        except Exception as e:
                            content = f"TOOL ERROR calling {name}: {e}"
                            table = None

                if _is_error_content(content):
                    last_tool_had_error = True
                elif table:
                    last_verified_table = table
                    last_verified_source = name

                print(f"[hop {hop}] [tool call] {name}({args}) -> {content[:2000]}")

                history.append({
                    "role": "tool",
                    "content": content,
                    "tool_name": name,
                })

        return "[stopped: hit tool-call limit for this turn]"

    return (pm_trader_agent,)


@app.cell
def _(mcp_manager, mo):
    # one natural-language example per tool (used by the chat suggestions too)
    EXAMPLES = {
        "init_account":       "Initialize my account with $10,000",
        "get_balance":        "Check my account balance",
        "reset_account":      "Reset my account (wipes all history)",
        "search_markets":     "Search for markets about the Fed",
        "list_markets":       "Show the top 10 markets by volume",
        "get_market":         "Show details for will-xrp-reach-2pt4-in-july-2026",
        "get_order_book":     "Show the order book",
        "get_tags":           "List all market categories",
        "get_markets_by_tag": "Show markets in the politics category",
        "get_event":          "Show the event for the 2026 World Cup winner",
        "watch_prices":       "Watch live prices",
        "buy":                "Buy $100 of YES",
        "sell":               "Sell 50 shares of YES",
        "portfolio":          "Show my open positions",
        "history":            "Show my recent trade history",
        "place_limit_order":  "Place a limit buy",
        "list_orders":        "List my pending limit orders",
        "cancel_order":       "Cancel limit order #3",
        "cancel_all_orders":  "Cancel all my limit orders",
        "check_orders":       "Check and fill my pending limit orders",
        "stats":              "Show my performance stats",
        "backtest":           "Backtest a strategy from a CSV of prices",
        "resolve":            "Resolve my position",
        "resolve_all":        "Resolve all my closed markets",
        "stats_card":         "Generate my shareable stats card",
        "leaderboard_entry":  "Generate my leaderboard entry",
        "share_content":      "Create shareable content for X (Twitter)",
        "pk_card":            "Compare account",
        "leaderboard_card":   "Show the top 10 leaderboard",
        "pk_battle":          "Run a PK battle between two strategies",
    }

    GROUPS = [
        ("💰 Account & setup",        ["init_account", "get_balance", "reset_account"]),
        ("🔎 Market discovery",       ["search_markets", "list_markets", "get_market",
                                        "get_order_book", "get_tags", "get_markets_by_tag",
                                        "get_event", "watch_prices"]),
        ("📈 Trading (market orders)",["buy", "sell"]),
        ("📂 Positions & history",    ["portfolio", "history"]),
        ("🎯 Limit orders",           ["place_limit_order", "list_orders", "cancel_order",
                                        "cancel_all_orders", "check_orders"]),
        ("📊 Performance & backtest", ["stats", "backtest"]),
        ("✅ Resolution",             ["resolve", "resolve_all"]),
        ("🏆 Sharing / social / PK",  ["stats_card", "leaderboard_entry", "share_content",
                                        "pk_card", "leaderboard_card", "pk_battle"]),
    ]

    _by_name = {t.name: t for t in mcp_manager.tools}

    def _desc(name):
        t = _by_name.get(name)
        return t.description.strip().splitlines()[0] if t and t.description else ""

    def _table(names):
        rows = ["| Command | What it does | Try saying |", "| --- | --- | --- |"]
        for n in names:
            if n in _by_name:
                rows.append(f"| `{n}` | {_desc(n)} | *{EXAMPLES.get(n, '')}* |")
        return mo.md("\n".join(rows))

    _accordion = mo.accordion({label: _table(names) for label, names in GROUPS})

    instructions_ui = mo.vstack([
        mo.md(
            f"""
            # 🧠 Polymarket Paper-Trading Assistant
            Chat in plain English — the assistant picks the right tool from **{len(mcp_manager.tools)} available**.
            Data is **real & live**; fills are **simulated**. Your account, trades and positions
            persist in `~/.pm-trader/default/paper.db` across kernel restarts.

            **Typical flow:**&nbsp; 🔎 discover a market → inspect it + its **order book** →
            📈 buy/sell (or 🎯 place a limit order, then `check_orders`) →
            📂 track your **portfolio** → ✅ **resolve** when it closes → 📊 review **stats**.

            > ⚠️ Say **“reset my account”** only if you truly want to wipe all history.
            """
        ),
        mo.md("### 📖 All commands — click a category to expand"),
        _accordion,
    ])
    instructions_ui
    return (EXAMPLES,)


@app.cell
def _(EXAMPLES, mo, pm_trader_agent):
    chat_ui = mo.ui.chat(
            pm_trader_agent,
            prompts=list(EXAMPLES.keys()),   # one clickable suggestion per tool
        )
    chat_ui
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
