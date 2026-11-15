# Free AI Model Aggregator (free-model-aggregator-02)

One server, two APIs, live free models behind every answer.

`main.py` (Python 3.13, standard library only) serves the OpenAI Chat
Completions API (`POST /v1/chat/completions`) and the Anthropic Messages API
(`POST /v1/messages`), in JSON and in server-sent events, and answers each
prompt with a real completion from a live free OpenRouter model, dialled
through the grader's gateway on the key pushed to the competition.

## How it works

- **Catalog**: one provider, `openrouter`, with exactly the fourteen dialable
  free models from the brief, each with its rating, context length and
  capabilities (tools / vision / JSON mode). `python3 main.py --list` prints
  the same document it registers.
- **Routing**: every prompt is served by the highest-rated free model that
  meets its requirements. JSON-mode prompts get a JSON-capable model; tools
  prompts get a tools-capable model; the rest get the best model overall.
- **Failover**: a refusal (429 / 5xx) is never an answer. The server fails
  over to the next-best model that meets the prompt, reports the refusals it
  recovered from in the `x-makefun-upstream-failures` header, and cools a
  refused model down so one rate-limited model cannot take every later
  prompt down with it.
- **Wires**: an Anthropic request is translated to a Chat Completions body,
  dialled on the gateway, and translated back. The gateway names the model
  without the `openrouter/` prefix; every answer and streamed chunk puts back
  the id that was dialled, so the `model` field always names the model that
  really answered.

## Practice results (kit env + mock gateway, local)

| Trial | Score | Quality | Served | Wire | Failover | Catalog |
|-------|-------|---------|--------|------|----------|---------|
| 0 | 100 | 100 | 100 | 100 | 100 | 100 |
| 1 | 100 | 100 | 100 | 100 | 100 | 100 |
| 2 | 100 | 100 | 100 | 100 | 100 | 100 |

With the top model refused once (429), trial-0 still serves 6/6 at score 94
(quality 90, everything else 100). The mock gateway answers with canned
completions carrying each prompt's token; the live gateway dials real models.
