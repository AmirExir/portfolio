---
name: streamlit-production
description: Refactor, debug, and deploy reliable Streamlit applications with caching, modular architecture, secure configuration, and clear user feedback.
---

# Production Streamlit

## Use this skill when

Use this skill for Streamlit architecture, performance, caching, deployment, state management, error handling, secrets, or UI integration with ML, RAG, or power-system workflows.

## Required workflow

1. Identify the expensive resources and computations.
2. Separate UI code from domain and data logic.
3. Use `st.cache_resource` for stable clients, models, and indexes.
4. Use `st.cache_data` for deterministic serializable results.
5. Define cache invalidation based on content or configuration versions.
6. Validate all user inputs.
7. Show actionable errors without leaking secrets or stack traces.
8. Test core logic outside Streamlit.
9. Verify deployment dependencies and startup behavior.

## Architecture

Prefer:

```text
app.py
src/
├── services/
├── domain/
├── data/
├── ui/
└── config.py
```

The main app file should orchestrate pages and presentation, not contain the entire application.

## Performance rules

- Do not reload models or vector stores on every rerun.
- Do not re-embed unchanged documents.
- Batch API and database operations.
- Use pagination or lazy loading for large results.
- Avoid caching objects that cannot be safely serialized unless using resource caching.

## State rules

- Use `st.session_state` only for user-session state.
- Initialize keys consistently.
- Avoid hidden state transitions.
- Make reset behavior explicit.

## Security

- Use Streamlit secrets or environment variables.
- Never commit credentials.
- Do not display raw exception details to end users.
- Sanitize filenames and uploads.
- Validate file type and size.

## Completion format

Report:

- Architecture changes
- Caching behavior
- Error handling
- Deployment checks
- Remaining performance risks
