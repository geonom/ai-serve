# Integrating an app into ai-serve's Apps & Services page

(Besides manifest-described apps, the tab lists **built-in** services the backend runs itself — currently ComfyUI. They are opened and exposed the same way; start/stop go through the backend's own code.)

Any local app can show up in the dashboard's **Apps & Services** page (status, start, stop, open, log, and who can reach it via Tailscale). The app needs **no code changes** — it describes itself in one file in its project root.

## 1. Put the project under a service root

The dashboard scans every **direct child** of its service roots (`SERVICE_ROOTS`, default `./apps` and `./plugins` in the ai-serve folder) for a `llama-serve.service.json`. Hidden and symlinked directories are skipped.

A project elsewhere can be added in ai-serve-ui's optional `services.json`; the same file can hide projects:

```json
{
  "include": ["/opt/some-app"],
  "exclude": ["storytime", "~/projects/old-project"]
}
```

`exclude` takes service ids or absolute project paths. No restart needed — both files are re-read on every request.

## 2. Write `llama-serve.service.json`

```json
{
  "$schema": "http://localhost:8030/api/services/manifest-schema",
  "id": "storytime",
  "name": "StoryTime",
  "description": "Interactive stories with chat, images and TTS via the ai-serve-ui gateway.",
  "command": ["./run.sh"],
  "env": {"STORYTIME_PORT": "{port}"},
  "port": 8008,
  "bind": "127.0.0.1",
  "match": ["main.py"],
  "open_path": "/"
}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Stable id: lowercase letters, digits, `-` (≤40). Used in URLs, logs and the exposure settings. |
| `name` | yes | Display name (≤60). |
| `description` | no | One line for the card (≤300). |
| `command` | yes | argv, **no shell**. `command[0]` must be an executable script inside the project, written `./<path>` (symlinks pointing outside are refused). `{port}` is replaced by `port`. |
| `env` | no | Extra environment for the start (`{port}` substituted). `PATH`, `LD_*`, `BASH_ENV`, `ENV`, `PYTHONSTARTUP`, `PYTHONHOME`, `NODE_OPTIONS`, `PERL5OPT`, … are refused. |
| `port` | yes | Port the app listens on when started from the dashboard (1024–65535; not 8030 or ComfyUI's). Unique across services. |
| `bind` | no | `127.0.0.1` (default), `localhost` or `::1` — **loopback only**. Who else may reach the app is chosen in the dashboard (section 4), never by the app; a manifest with `0.0.0.0`/`::` is refused. Make the app's host configurable (env var) and set it here via `env`. |
| `match` | yes | Command-line fragments that identify the app's own processes (e.g. `main.py`, `benchy serve`, `munin.app`). Used to find running copies — also ones started in a terminal — and to make sure a stop only hits the app. Be specific: helper scripts in the same directory must **not** match. |
| `open_path` | no | Path the "Open" link and the tailnet URL point to (default `/`). |
| `health_path` | no | If set, a GET there must answer < 500 for the instance to count as healthy. |
| `tailnet_port` | no | HTTPS port on the tailnet when exposed (default `port + 10000`). 443, 8443 and 10000 are refused: Funnel could publish those. |
| `category` | no | Group on the "Apps & Services" page: `ai`, `stories`, `learning`, `knowledge`, `tools` (default), `integrations`. |
| `icon` | no | One of `speaker mic image book branch crown waves bubbles magnifier archive bars cloud chat trend code music video globe puzzle game camera pen map heart star tool sun home`. The dashboard draws it; without one, the first letter of `name` is shown. |
| `screenshot` | no | `./path` to a PNG/JPEG/WebP inside the project (at most 5 MB). Shown on the app's card and when the app is pinned to the overview. SVG is refused (it can carry script); a path leaving the project (also via symlink) is refused, and re-checked each time it is served. |
| `identity` | no | How the app handles people (`docs/identity.md`): `none` = keeps no per-person data, `single` = one person's data — only the owner may use it (default), `multi` = separates data per person. A `single` app can't be exposed to the whole tailnet. |
| `maturity` | no | How far the app is: `alpha` (works, still rough), `beta` (usable, being tested), `stable` (fully tested). Shown as a pill in the dashboard and as a badge on the door's overview page for the people it is shared with. Missing = not rated. |
| `ui` | no | `false` for services without a page of their own (APIs such as tts-serve): the dashboard shows no Open/share button and refuses Tailscale exposure for them. Default `true`. |
| `status_path` | no | If set, a GET on the app's own port there answers `{"facts": [{"label": "...", "value": "..."}]}`; the overview shows at most 6 of them as plain text (label ≤ 40, value ≤ 80 characters; numbers and booleans are turned into text, anything else is dropped). |

Unknown fields are an **error** (typos like `comand` would otherwise be silently ignored); keys starting with `_` or `$` are allowed for comments and `$schema`.

What the start does: runs `command` in the project directory, in its own session (it survives a dashboard restart), output to ai-serve-ui's `logs/service-<id>.log`. The environment is the dashboard's minus its own secrets and Python environment (`PYTHONPATH`, `VIRTUAL_ENV`, its venv in `PATH`) — your script sets up its own interpreter. A start is refused while anything listens on `port`.

## 3. Check it

Open *Apps & Services*, or:

```bash
curl -s localhost:8030/api/services | python3 -m json.tool
```

A broken manifest appears under `errors` with the reason; a valid one under `services` with `source` (which root found it). Validate against the schema in your editor with the `$schema` line above, or:

```bash
curl -s localhost:8030/api/services/manifest-schema > /tmp/schema.json
python3 -c "import json,jsonschema; jsonschema.validate(json.load(open('llama-serve.service.json')), json.load(open('/tmp/schema.json')))"
```

The schema checks shape; only the dashboard can check file-system facts (script exists and is executable, port not reserved or taken by another service).

## 4. Reachability via Tailscale

Only with `NETWORK_PROVIDER=tailscale`; tried on the developer's tailnet (the *Only my Tailscale user* level live,
*Whole tailnet* through unit tests only).

Per service, the card's **Reachable for** chooses:

| Level | Effect |
|---|---|
| Only this machine | Nothing beyond `localhost` (default). |
| Whole tailnet | `tailscale serve` (tailnet only) on `https://<machine>.ts.net:<tailnet_port>` → the app. Every tailnet member the ACLs allow — **including devices other users shared into the tailnet**. |
| Only my Tailscale user | Same URL, but through an owner gate in the dashboard that passes only requests carrying the machine owner's `Tailscale-User-Login` (set by Tailscale, not forgeable by clients). |

There is no public level; publishing to the internet is only possible for the dashboard's own curated `/public` API. Changing a level is allowed on this machine or from one of the owner's tailnet devices — not from the LAN. Choices are stored in `service_exposure.json` and re-applied when the dashboard starts.

**The dashboard is the only authority over reachability.** Consequently:
- a `tailscale serve` root route that points at a service's port but wasn't created by the dashboard is **removed** when levels are applied (path mounts on the same port stay);
- a running instance that listens beyond loopback (e.g. started by hand with `HOST=0.0.0.0`) is flagged red on its card — stop it and start it from the dashboard.
