# scf-agent

Supply Chain Finance (SCF) AI Agent system built with **LangChain** and **LangGraph**.

The platform ingests supplier and macroeconomic data, scores supplier credit
risk, runs dynamic-discounting / early-payment IRR simulations, and exposes a
natural-language ReAct agent (Anthropic Claude) so finance teams can ask
questions like *"Which of my top-50 suppliers are at risk this quarter, and
what early-payment discount maximizes IRR?"*.

---

## 1. Project Overview

`scf-agent` is composed of four loosely coupled layers:

| Layer            | Responsibility                                                                |
| ---------------- | ----------------------------------------------------------------------------- |
| **Pipeline**     | Ingest raw supplier / macro data, engineer features, persist processed sets.  |
| **Models**       | XGBoost-based supplier risk scorer with versioned artifacts.                  |
| **Optimization** | IRR / NPV engine for early-payment-discount decisioning vs. WACC.             |
| **Agent**        | LangGraph ReAct agent (Claude Sonnet 4) that calls the layers above as tools. |
| **API**          | FastAPI surface exposing suppliers, decisions, and the agent endpoint.        |
| **Dashboard**    | Streamlit UI for risk tables, IRR charts, and a chat panel.                   |

```
              ┌─────────────┐
              │  Streamlit  │  ← analyst UI
              └──────┬──────┘
                     │ HTTP
              ┌──────▼──────┐
              │   FastAPI   │  ← /suppliers /decisions /agent
              └──┬──────┬───┘
                 │      │
         ┌───────▼─┐  ┌─▼──────────────┐
         │  Tools  │  │ Risk / IRR /    │
         │ (Lang-  │◄►│ Pipeline modules│
         │ Graph)  │  └─────────────────┘
         └─────────┘
```

---

## 2. Setup with `uv`

This project uses [uv](https://github.com/astral-sh/uv) as its package /
environment manager. Python **3.11+** is required.

```bash
# 1. install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. clone and enter the repo
git clone <your-repo-url> scf-agent
cd scf-agent

# 3. create the env and install all locked dependencies
uv sync

# 4. configure secrets
cp .env.example .env
# edit .env: ANTHROPIC_API_KEY, FRED_API_KEY, etc.
```

All commands below assume `uv run` to ensure the project virtual environment
is active.

---

## 3. Running the components

### 3a. Data pipeline

Ingest raw sources, build features, and write processed parquet files into
`data/processed/`.

```bash
uv run python -m scf_agent.pipeline.ingest
uv run python -m scf_agent.pipeline.preprocess
uv run python -m scf_agent.pipeline.features
```

### 3b. FastAPI service

```bash
uv run uvicorn scf_agent.api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- Swagger UI:  http://localhost:8000/docs
- Suppliers:   `GET /suppliers/`
- Decisions:   `POST /decisions/irr`
- Agent:       `POST /agent/ask`

### 3c. Streamlit dashboard

```bash
uv run streamlit run dashboard/app.py
```

The dashboard expects the FastAPI service to be reachable at
`http://localhost:8000`.

### 3d. Tests

```bash
uv run pytest
```

---

## 4. LangChain Agent Architecture

The natural-language agent lives in [`scf_agent/agent/`](scf_agent/agent) and
is built with **LangGraph's prebuilt ReAct agent** on top of
**`langchain-anthropic`'s** `ChatAnthropic` (`claude-sonnet-4-20250514`).

```
              ┌──────────────────────────┐
   user  ─►   │     ask(question)        │   ← async entrypoint
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │   LangGraph ReAct agent  │
              │   (LLM + tools + memory) │
              └────┬─────────────────┬───┘
                   │                 │
        ┌──────────▼──────┐   ┌──────▼───────────────┐
        │ ChatAnthropic   │   │ ConversationBuffer-  │
        │ Claude Sonnet 4 │   │ WindowMemory (k=10)  │
        └─────────────────┘   └──────────────────────┘
                   │
        ┌──────────▼─────────────────────────────────┐
        │            Tools (LangChain @tool)         │
        │  get_supplier_risk_score(supplier_name)    │
        │  run_irr_simulation(days, pct, value)      │
        │  get_at_risk_suppliers()                   │
        │  get_macro_indicators()                    │
        └────────────────────────────────────────────┘
```

- **Tools** ([`scf_agent/agent/tools.py`](scf_agent/agent/tools.py)) are thin
  `@tool`-decorated wrappers around the risk scorer, IRR engine, and macro
  ingest. They are intentionally side-effect-free and JSON-serializable so the
  agent can reason over their outputs.
- **Memory** ([`scf_agent/agent/memory.py`](scf_agent/agent/memory.py)) uses
  `ConversationBufferWindowMemory(k=10)` so the last 10 turns of the
  conversation are kept in context — enough for an analyst's working session
  without bloating tokens.
- **Agent** ([`scf_agent/agent/nl_agent.py`](scf_agent/agent/nl_agent.py))
  composes the LLM, the four tools, and the memory into a LangGraph ReAct
  graph and exposes a single async `ask(question: str) -> str` function used
  by the API router.

---

## 5. Free API Sources

| Source                                                                                                                     | What we use it for                                                          | Auth          |
| -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------- |
| [**FRED** (Federal Reserve Economic Data)](https://fred.stlouisfed.org/docs/api/fred/)                                     | Macro indicators (Fed funds rate, CPI, industrial production) for features. | `FRED_API_KEY` |
| [**DataCo Smart Supply Chain dataset**](https://data.mendeley.com/datasets/8gx2fvg2k6/5)                                   | Historical supplier / shipment data for training the risk scorer.           | none           |
| [**Anthropic API** (Claude Sonnet 4)](https://docs.anthropic.com/)                                                         | LLM that drives the LangGraph ReAct agent.                                  | `ANTHROPIC_API_KEY` |

All three are either free or have a generous free tier suitable for development.

---

## License

MIT
