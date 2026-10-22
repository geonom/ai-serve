# Identity building blocks for apps (docs/identity.md v1)

One function each — "who is using this app?" — so no app writes its own token checking.
Copy the single file into the app (they are small and versioned via `__version__` / `VERSION`):

| Language | File | Needs | Tests |
|---|---|---|---|
| Python | `python/llama_serve_identity.py` | `PyJWT[crypto]` | `tests/unit/test_identity_sdk.py` (repo root, `venv/bin/python3 -m pytest`) |
| TypeScript (Node) | `ts/llamaServeIdentity.ts` | `jose` | `ts/llamaServeIdentity.test.ts` (`cd ts && npm install && npx vitest run`) |

Both implement the same rules: a valid ES256 token from the door names the person; no header
from loopback = the owner (`LLAMA_SERVE_OWNER_SUB`); everything else — no header from
elsewhere, or a header that doesn't verify, also from loopback — raises `Unauthenticated`
(answer 401). Configuration comes from the environment the dashboard sets:
`LLAMA_SERVE_OWNER_SUB`, `LLAMA_SERVE_IDENTITY_JWKS`, `LLAMA_SERVE_IDENTITY_ISSUER`.

When changing the rules, change both files and both test suites together.
