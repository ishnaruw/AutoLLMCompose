You decompose a user goal into ordered subtasks.

User goal:
{user_goal}

Instructions:
1) Break the goal into 2 to 5 clear subtasks that could reasonably be handled by external APIs.
2) Each subtask must describe an API-backed capability, such as fetch, search, retrieve, check, scan, send, book, create, update, calculate, or summarize using a model/API.
3) Order the subtasks in a logical execution sequence.
4) Preserve the user's intent, but do not create standalone subtasks for local workflow logic.
5) Treat objects already supplied by the user goal, such as "given domains", "provided URLs", selected items, or local state, as inputs. Do not add a separate API subtask to fetch those inputs unless the user explicitly names an external source/catalog service.
6) For delivery subtasks, lead with the delivery channel and keep the payload generic enough for API retrieval, e.g. "Send an email with selected results" instead of making the payload domain the main phrase.
7) Do not create separate API subtasks for internal/local decisions, aggregation, scoring, logging, formatting, final recommendations, or policy decisions unless the user explicitly asks for an external API call for that action.

Return strict JSON in this format:

{
  "subtasks": [
    {
      "id": 1,
      "description": "short phrase describing this subtask"
    }
  ]
}

Rules:
- Do not invent APIs or parameters.
- Do not include anything outside the JSON object.
- Number subtasks starting from 1.
- Do not create standalone subtasks for formatting, UI display, user selection, sorting, ranking, comparing, aggregating, combining results, dashboard updates, or policy decisions.
- Do not create standalone API subtasks for internal/local decisions, aggregation, scoring, logging, formatting, final recommendations, or policy decisions unless the user explicitly asks for an external API call for that action.
- For phrases like "local decision", "blocking decision", "decide whether to block", "final decision", "record decision", and "update policy", keep the decision inside planner rationale or inside the nearest relevant API step. Do not create a separate API subtask.
- Do not create standalone subtasks for fetching a configuration, inventory, monitoring list, or user-provided input unless the goal explicitly asks to call a specific external inventory/catalog API.
- Do not create standalone subtasks for returning, handing off, or sending results to an unnamed downstream/local service. Fold that into the nearest API-backed scan/check step, or omit it if it is only local application behavior.
- Fold local workflow logic into the nearest API-backed subtask description.
- If a goal includes internal logic between API calls, mention it briefly inside the related API-backed subtask instead of making it its own subtask.
- If a goal needs to compare already-fetched values against stored baselines, thresholds, or previous local records, treat that comparison as local workflow logic and fold it into the related fetch/check subtask.
- If a subtask is internal-only and has no clear external API action, merge it into the nearest previous API-backed subtask or remove it before retrieval.
- Prefer API-facing verbs over UI/internal verbs.

Good:
- "Retrieve nearby venues using location search APIs"
- "Fetch pricing or availability details for selected venues"
- "Fetch current or historical product pricing via price/product APIs; compare locally against stored baselines to detect drops"
- "Check provided domains for risk using domain or threat-intelligence APIs"
- "Check the URL for adult-content risk using a content-classification API"
- "Send an email with selected results using an email delivery API"
- "Send selected results via SMS"

Bad:
- "Fetch the list of domains to monitor via a configuration API"
- "Compare pricing plans"
- "Check fetched prices against stored baseline values using a price-monitoring/check API"
- "Present results to the user"
- "Compose SMS digest"
- "Combine scan results and decide whether to block"
- "Record the blocking decision through a security-policy update API"
- "Send aggregated scan results to the downstream blocking service"
