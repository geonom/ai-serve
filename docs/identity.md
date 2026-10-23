# Identity standard for ai-serve-ui apps (v1)

How an app learns **who** is using it, and how it keeps every person's data separate —
so the owner can let people they know use selected apps from anywhere, without any app
building its own login. Decided 2026-10-17 (user: apps shareable per person, "egal über
welchen Service per Konto-Login / SSO / selbst angebotene User Identification").

Status: **v1 draft.** In this release: the manifest field `identity` and its guard, the building blocks in
`sdk/identity/` (unit-tested), and the Spielwiese as a `multi` app. **The door itself – an `access` plugin with the
login, the per-app policy and the signed header – is not included**; it runs on the developer's own setup (a small
VPS + Nextcloud as login) and isn't general enough to publish yet. Without it, only this machine (the owner) and the
owner's Tailscale devices reach the apps.

## 1. The pieces

```
browser ──► door (Pomerium, on this machine) ──► app (127.0.0.1:<port>)
              │  login via the identity provider (the owner's Nextcloud, OIDC)
              │  per-app policy: who may use which app
              └► adds X-Llama-Serve-Identity (signed) to every request it lets through
```

- **Identity provider (IdP):** where people log in. Planned: the owner's Nextcloud as OpenID
  Connect provider (accounts, passkeys, 2FA live there). Replaceable — apps never talk to it.
- **Door:** the only way into an app from outside this machine. Checks the login, applies the
  per-app policy from the dashboard, strips anything a client sends in the headers below, and
  adds the signed identity. *(planned)*
- **App:** listens on 127.0.0.1 only (already required by the service manifest) and trusts
  identity **only** in the form described here.

Apps never see passwords, never run a login flow, never call the IdP.

## 2. What an app receives

### 2.1 `X-Llama-Serve-Identity` *(planned)*

With Pomerium as the door the same signed assertion arrives as `X-Pomerium-Jwt-Assertion`, with
the app's **public host** (e.g. `sonne.apps.example.org`) as `aud`; the dashboard/manifest sets
`LLAMA_SERVE_IDENTITY_AUDIENCE` to that host and `LLAMA_SERVE_IDENTITY_JWKS` to
`https://<host>/.well-known/pomerium/jwks.json`. The SDKs accept both header names — only the
signature by the door's key counts.

A JWT (JWS, ES256) issued by the door for this request:

| Claim | Meaning |
|---|---|
| `sub` | **The person.** Stable, opaque string, assigned by the IdP. Never changes for the same person, even if their name, e-mail or linked logins change. The only key an app may store data under. |
| `name` | Display name (may change — never use as a key). |
| `email` | Optional; may be absent. |
| `picture` | Optional avatar URL. |
| `groups` | List of group names (e.g. `["familie"]`) for roles inside the app. |
| `aud` | The app's manifest `id` — reject tokens for another app. |
| `iss`, `iat`, `exp` | Issuer and a short lifetime (minutes). Reject expired tokens. |

Verify the signature against the keys at the URL in the environment variable
`LLAMA_SERVE_IDENTITY_JWKS` (set by the dashboard when it starts the app). Reject on any
failure — wrong signature, wrong `aud`, expired, missing `sub`.

### 2.2 No identity header

| Request comes from | Who it is |
|---|---|
| loopback (127.0.0.1 / ::1) without the header | **the owner** — someone at this machine. Use `sub` = `LLAMA_SERVE_OWNER_SUB` (env, set by the dashboard; `owner` until the owner's IdP account is linked). The owner's data is therefore the same whether they come via `localhost` or log in through the door. |
| loopback, but brought by Tailscale Serve or the dashboard's gate (they connect from loopback too) | `Tailscale-User-Login` = the owner's login (`LLAMA_SERVE_OWNER_TAILSCALE_LOGIN`, env, set by the dashboard) → the owner; any other login → nobody; `Tailscale-Funnel-Request` → nobody. Serve sets these headers itself and strips client-supplied copies. |
| anywhere else without a valid header | nobody — answer **401**. |

**Only when the request names this machine** (since 2026-10-23): without a door identity, `Host` must be an IP
address, `localhost`, a name in `LLAMA_SERVE_LOCAL_HOSTS` (this machine's names and addresses, comma-separated — set
by the dashboard when it starts the app; a manifest can't change it) or the app's public name
(`LLAMA_SERVE_IDENTITY_AUDIENCE`), and an `Origin`, if present, must be one of those too — else nobody (401).
Otherwise any web page could be "the owner": one that points its own name at 127.0.0.1 (DNS rebinding — the browser
then sends `Host` = its name, often without an `Origin`), or one posting to the app from elsewhere (foreign `Origin`).
Both SDKs do this (`_this_machine` / `thisMachine`).

### 2.3 Legacy: Tailscale headers

Until an app has moved behind the door, apps reached via the dashboard's Tailscale levels
receive `Tailscale-User-Login` (set by Tailscale Serve, which strips client-supplied
copies). Apps may keep using it in the meantime, as Munin does — but link it to an internal
user, never store data under the login string itself, so switching to `sub` later is a
one-line mapping.

## 3. What an app must do

1. **Key everything that belongs to a person by `sub`** — a `user_id` column, or a folder
   `data/users/<sub>/`. Never by name or e-mail.
2. **Keep shared things explicit.** Anything visible to more than one person (shared
   projects, a common library) is marked as shared, with its members listed by `sub`.
3. **Migrate existing data to the owner.** On first start with this standard, everything that
   exists so far belongs to `LLAMA_SERVE_OWNER_SUB`.
4. **Per person: export and delete.** A person's data can be listed and removed by `sub`
   (people leave; the owner should be able to honour that).
5. **Pass the person on to the gateway** when calling it on someone's behalf (header
   `X-Llama-Serve-User: <sub>`), so usage can be counted and limited per person. *(planned)*
6. **Never trust identity from anywhere else** — query parameters, cookies set by the app
   itself for identity, or the convenience headers below without the JWT.

Convenience headers `X-Llama-Serve-User` (= `sub`) and `X-Llama-Serve-User-Name` *(planned)*
are for logging and display only.

7. **Include `sdk/door/door-session.js` as the first script** of every page behind the door
   (copy it into the app; serve it like the app's other static files; a Vite app puts it in
   `src/` or `public/`). When the door login runs out, a same-origin `fetch` would otherwise
   fail and the app would need a reload, losing typed text, the reading position and ticked
   boxes (owner, 2026-10-20). The script renews the login in the background — a hidden,
   script-less frame to `/.door-renew` while Nextcloud still knows the person, else a small
   "Anmelden" notice that opens the login in a popup — and then sends the request again; the
   page is never reloaded. It checks the login on `/.door-probe` (no app serves it; 404 from
   the app = signed in, redirect/401 from the door = not). A service worker must leave
   `/.door-*` alone. Not covered: XMLHttpRequest, EventSource, WebSocket. Tests:
   `node --test sdk/door/door-session.test.mjs`. Used by Spielwiese, StoryTime, Sonnenrechner,
   Liruna, Detektiv, Kristallhelden; after changing the SDK, copy it into each again.

## 4. The manifest field `identity`

Every app says in its `llama-serve.service.json` how it handles people:

| Value | Meaning | May be used by other people? |
|---|---|---|
| `none` | Keeps no per-person data on the server (stateless, or data stays in the browser). | yes |
| `single` | Keeps data, but not separated by person — whoever uses it sees and changes everything. | **no** — only the owner |
| `multi` | Separates data by person as in section 3 (or, until it moves behind the door, by the Tailscale login as in 2.3). | yes |

Missing field = `single`: an app is assumed to keep one person's data until it says
otherwise. The dashboard enforces it: a `single` app can't be exposed to anyone but the
owner — today that means the Tailscale level „Ganzes Tailnet" is refused for it (other
users' shared-in devices would reach it); later the door refuses any policy that names
someone other than the owner.

## 5. Checklist for moving an app to `multi`

- [ ] Find every place the app stores data (database tables, files, browser storage synced to
      the server) and add the `sub` key.
- [ ] Read the identity as in section 2 — copy the reference function from `sdk/identity/`
      (Python: `llama_serve_identity.identify`, TypeScript: `identify` from
      `llamaServeIdentity.ts`) instead of writing token checks yourself.
- [ ] Migrate existing data to `LLAMA_SERVE_OWNER_SUB`.
- [ ] Shared data marked explicitly; per-person export/delete.
- [ ] Tests: two people never see each other's data; no header from outside → 401;
      loopback without header → owner.
- [ ] Set `"identity": "multi"` in the manifest.
- [ ] **Vite apps:** put `sdk/vite/servePolicy.js` first in `plugins` (`servePolicyPlugin({...})`). Vite's dev
      server hands out *any* file under the project root -- found 2026-10-17: TrioLingo served
      `data/users/*/state.json` and `scripts/deploy.env` (FTP credentials). Keep per-person data out of
      what the server can reach in any case.
