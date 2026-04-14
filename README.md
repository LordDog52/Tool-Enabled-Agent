# Tool-Enabled Agent (FastAPI + Ollama + PostgreSQL)

This project implements a tool-enabled agent API that:
- Calls an Ollama model (`/api/generate`)
- Lets the model invoke tools (database query + external API simulation)
- Runs a guardrail moderation pass before returning final answers
## Flowchart
![Agent AI Workflow - Frame 1](https://github.com/user-attachments/assets/99de4182-572d-4fa8-bc1a-ea6ba1921720)

## Features

- FastAPI endpoint: `POST /ollama/generate`
- Tool calling loop with retry limit
- Structured PostgreSQL query tool with schema-aware validation
- External API simulation tool with timeout + retry
- Guardrail classification using `Qwen/Qwen3Guard-Gen-0.6B`
- Seeded internal dataset under schema `intern_task`

## Project Structure

```text
src/
  main.py                       # FastAPI app, startup/shutdown, endpoint
  agent/
    agent.py                    # agent() + agent_loop()
    manual_decision.py          # Tool decision manually
    parser.py                   # parses <tool_call> blocks from model output
    prompt_template.py          # prompt rendering with tool schemas
  tools/
    prompt_guard.py             # prompt_guard (Huggingface)
    tool1.py                    # Structured_Data_Query_Tool (PostgreSQL)
    tool2.py                    # External_API_Simulation_Tool (HTTP)
    tool3.py                    # Guardrail_Evaluation_Tool (Huggingface)
  services/
    db_init.py                  # executes SQL seed file
  schemas/
    internal_database_seed.sql  # schema + tables + seed data
tests/
  agent.py                      # integration-style pytest cases
```

## Requirements

- Python 3.10+
- PostgreSQL (local or container)
- Ollama running at `http://localhost:11434`
- Local availability of guardrail model: `Qwen/Qwen3Guard-Gen-0.6B`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
DB_USER=langchain
DB_PASSWORD=langchain
DB_HOST=localhost
DB_PORT=6024
DB_NAME=langchain
```

## Quick Start

1. Start PostgreSQL (example with Docker):

```bash
docker run --name pgvector-container \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain \
  -p 6024:5432 \
  -d pgvector/pgvector:pg16
```

2. Initialize database schema and seed data:

```bash
python src/services/db_init.py
```

3. Start Ollama (example with Docker):

```bash
docker run -d --name ollama-docker -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
docker exec -it ollama-docker ollama pull qwen3:1.7b
```

4. Run the API:

```bash
uvicorn src.main:app --reload
```

The app docs are available at `http://127.0.0.1:8000/`.

## API Usage

Endpoint:

- `POST /ollama/generate`
- Form fields:
  - `question` (required)
  - `model` (optional, default: `qwen3:1.7b`)
  - `thinking` (optional, default: `false`)

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/ollama/generate" \
  -F "model=qwen3:1.7b" \
  -F "question=What is the SLA for Premium Support?" \
  -F "thinking=false"
```

## Tools

### 1) `Structured_Data_Query_Tool`

- Queries `intern_task.<table>` with validated filters
- Validates table names, columns, and value types against live schema metadata
- Supports scalar equality and `ANY(...)` for array columns

### 2) `External_API_Simulation_Tool`

- Safe HTTP wrapper (`GET/POST/PUT/DELETE`)
- Timeout and retry support
- Returns structured success/failure payloads

### 3) `Guardrail_Evaluation_Tool`

- Uses Hugging Face Transformers with local files
- Classifies output as `Safe`, `Unsafe`, or `Controversial`
- Helps enforce refusal behavior for unsafe content

## Seeded Database Tables

The SQL seed file creates and populates:

- `intern_task.dataset_metadata`
- `intern_task.policies`
- `intern_task.policy_rules`
- `intern_task.sla_lookup`
- `intern_task.accounts`
- `intern_task.system_status`

## Running Tests

Run all tests:

```bash
pytest -s tests/agent.py
```

Notes:
- Tests are integration-heavy and require running PostgreSQL + Ollama.
- Guardrail tests require the `Qwen/Qwen3Guard-Gen-0.6B` model available locally.

## Logging

- Logging config: `src/agent.conf`
- Log output file: `src/logging/agent.log`
- The log file is cleared on app startup.

## Known Operational Constraints

- Ollama endpoint is expected at `localhost:11434`.
- Guardrail model loading uses `local_files_only=True`; missing local model files will fail.
- If the selected model lacks required capabilities (tools or thinking), the API returns a capability error message.
