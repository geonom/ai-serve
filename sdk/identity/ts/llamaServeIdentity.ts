/**
 * Who is using this app? -- reference implementation of docs/identity.md (v1) in
 * ai-serve-ui, the TypeScript twin of sdk/identity/python/llama_serve_identity.py.
 * Copy this single file into an app; it needs `jose`.
 *
 *   const person = await identify(req.headers, req.socket.remoteAddress, { appId: "pv-calc" });
 *   person.sub       // the key for everything this person owns
 *   person.isOwner   // the owner of this machine
 *
 * Rules (security-relevant, tests in llamaServeIdentity.test.ts):
 * - X-Llama-Serve-Identity (or Pomerium's X-Pomerium-Jwt-Assertion) present -> must be an ES256
 *   JWT that verifies against the door's keys (LLAMA_SERVE_IDENTITY_JWKS URL, or keys passed in),
 *   aud = LLAMA_SERVE_IDENTITY_AUDIENCE (the app's public host) or else this app's id, unexpired,
 *   expected issuer (LLAMA_SERVE_IDENTITY_ISSUER, if set), with a sub. Anything that doesn't
 *   verify is nobody -- also on loopback: a bad header never falls back to the owner.
 * - no header from loopback -> the owner (LLAMA_SERVE_OWNER_SUB, set by the dashboard) -- unless
 *   Tailscale Serve brought it: with Tailscale-User-Login it is the owner only if that is
 *   LLAMA_SERVE_OWNER_TAILSCALE_LOGIN; with Tailscale-Funnel-Request it is nobody.
 * - no header from anywhere else -> nobody (Unauthenticated -> answer 401).
 */
import { createLocalJWKSet, createRemoteJWKSet, decodeProtectedHeader, jwtVerify, type JSONWebKeySet } from "jose";

export const VERSION = "1";
export const HEADER = "x-llama-serve-identity";
/** Pomerium (the door) sends its signed assertion under its own name; only the signature counts. */
export const DOOR_HEADERS = [HEADER, "x-pomerium-jwt-assertion"];

export class Unauthenticated extends Error {}

export interface Person {
  sub: string;
  name: string;
  email: string | null;
  picture: string | null;
  groups: string[];
  via: "door" | "local" | "tailscale";
  isOwner: boolean;
}

export interface IdentifyOptions {
  appId: string;
  jwks?: JSONWebKeySet;
  jwksUrl?: string;
  issuer?: string;
  ownerSub?: string;
}

type Headers = Record<string, string | string[] | undefined> | Headers_;
type Headers_ = { get(name: string): string | null };

const remoteSets = new Map<string, ReturnType<typeof createRemoteJWKSet>>();

function header(headers: Headers, name: string = HEADER): string | undefined {
  if (typeof (headers as Headers_).get === "function") return (headers as Headers_).get(name) ?? undefined;
  for (const [k, v] of Object.entries(headers as Record<string, string | string[] | undefined>)) {
    if (k.toLowerCase() === name) return Array.isArray(v) ? v[0] : v;
  }
  return undefined;
}

export function isLoopback(peer: string | undefined | null): boolean {
  if (!peer) return false;
  const p = peer.replace(/^\[|\]$/g, "").replace(/^::ffff:/i, "");
  return p === "::1" || /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(p);
}

function isIp(value: string): boolean {
  const v = value.replace(/^\[|\]$/g, "");
  return /^\d{1,3}(\.\d{1,3}){3}$/.test(v) || (v.includes(":") && /^[0-9a-f:.]+$/i.test(v));
}

function localNames(): Set<string> {
  const names = new Set(["localhost"]);
  for (const n of (process.env.LLAMA_SERVE_LOCAL_HOSTS || "").split(",")) if (n.trim()) names.add(n.trim().toLowerCase());
  const aud = (process.env.LLAMA_SERVE_IDENTITY_AUDIENCE || "").trim().toLowerCase();
  if (aud) names.add(aud.split("://").pop()!.split("/")[0].split(":")[0]);
  return names;
}

/** Without a door identity, Host and Origin must name this machine – else any web page could act as the owner
 * (DNS rebinding: Host = the page's name; a foreign page: its Origin). Found 2026-10-23. */
function thisMachine(headers: Headers): void {
  const names = localNames();
  const host = header(headers, "host");
  if (host) {
    let h = host.trim().toLowerCase();
    h = h.startsWith("[") && h.includes("]") ? h.slice(1, h.indexOf("]")) : (h.split(":").length === 2 ? h.split(":")[0] : h);
    if (!isIp(h) && !names.has(h)) throw new Unauthenticated(`unknown host name '${h}' (add it to LLAMA_SERVE_LOCAL_HOSTS if it is this machine)`);
  }
  const origin = header(headers, "origin");
  if (origin !== undefined) {
    let oh = "";
    try { oh = origin === "null" ? "" : new URL(origin).hostname.replace(/^\[|\]$/g, "").toLowerCase(); } catch { oh = ""; }
    if (!oh || !(names.has(oh) || isLoopback(oh))) throw new Unauthenticated(`request from a foreign page (Origin '${origin}')`);
  }
}

export async function identify(headers: Headers, peer: string | undefined | null, opts: IdentifyOptions): Promise<Person> {
  const owner = opts.ownerSub || process.env.LLAMA_SERVE_OWNER_SUB || "owner";
  let raw: string | undefined;
  for (const name of DOOR_HEADERS) {
    raw = header(headers, name);
    if (raw !== undefined) break;
  }
  if (raw === undefined) {
    if (!isLoopback(peer)) throw new Unauthenticated("no identity");
    thisMachine(headers);
    // Tailscale Serve (and the dashboard's owner gate) also connect from loopback: such a request
    // is someone on the tailnet, not someone at this machine. Serve sets these headers itself
    // and strips client-supplied copies.
    if (header(headers, "tailscale-funnel-request") !== undefined) throw new Unauthenticated("from the internet without identity");
    const tsLogin = header(headers, "tailscale-user-login");
    if (tsLogin !== undefined) {
      const ownerLogin = process.env.LLAMA_SERVE_OWNER_TAILSCALE_LOGIN;
      if (!ownerLogin || tsLogin !== ownerLogin) throw new Unauthenticated("tailnet user who is not the owner");
      return { sub: owner, name: "", email: null, picture: null, groups: [], via: "tailscale", isOwner: true };
    }
    return { sub: owner, name: "", email: null, picture: null, groups: [], via: "local", isOwner: true };
  }
  const url = opts.jwksUrl || process.env.LLAMA_SERVE_IDENTITY_JWKS;
  let keys;
  if (opts.jwks) keys = createLocalJWKSet(opts.jwks);
  else if (url) {
    if (!remoteSets.has(url)) remoteSets.set(url, createRemoteJWKSet(new URL(url), { cacheMaxAge: 300_000 }));
    keys = remoteSets.get(url)!;
  } else throw new Unauthenticated("no identity keys configured (LLAMA_SERVE_IDENTITY_JWKS)");

  const issuer = opts.issuer || process.env.LLAMA_SERVE_IDENTITY_ISSUER || undefined;
  let payload;
  try {
    if (decodeProtectedHeader(raw).alg !== "ES256") throw new Error("algorithm not allowed");
    ({ payload } = await jwtVerify(raw, keys, {
      // the door puts the app's public host into aud -- set via the environment
      algorithms: ["ES256"], audience: process.env.LLAMA_SERVE_IDENTITY_AUDIENCE || opts.appId, issuer, clockTolerance: 30,
      requiredClaims: ["exp", "aud", "sub"],
    }));
  } catch (e) {
    throw new Unauthenticated(`identity does not verify: ${(e as Error).message}`);
  }
  const sub = payload.sub;
  if (typeof sub !== "string" || !sub) throw new Unauthenticated("identity without sub");
  const groups = Array.isArray(payload.groups) ? payload.groups.map(String) : [];
  return {
    sub, name: String(payload.name ?? ""), email: typeof payload.email === "string" ? payload.email : null,
    picture: typeof payload.picture === "string" ? payload.picture : null, groups, via: "door", isOwner: sub === owner,
  };
}
