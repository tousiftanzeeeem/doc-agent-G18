# Example agent trace (A3 — fill this)

One full run of your agent on a single query: the query, each tool call in order
(with what came back), the grounding/abstention decision, and the final cited answer.
Add more trace files here if you like (one per interesting query).

- **Query:**
- **verifiable / judged / abstention:**

## Steps (tool calls in order)
| # | Tool | Input | What came back |
|---|------|-------|----------------|
| 1 | retrieve | | |
| 2 | rerank | | |
| 3 | read_page | | |
| 4 | extract | | |
| 5 | cite | | |

- **Loop / re-search (if any):** did the agent decide evidence was insufficient and retrieve again? why?
- **Grounding decision:** grounded (cite a page) OR abstained ("insufficient evidence") — and why.

## Final answer
- **Answer:**
- **Citation (page/chunk):**
- **Grounded:** true / false   **Confidence:**
