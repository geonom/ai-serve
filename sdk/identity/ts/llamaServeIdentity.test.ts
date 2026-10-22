// Same rules as ai-serve-ui tests/unit/test_identity_sdk.py (docs/identity.md section 2).
import { exportJWK, generateKeyPair, SignJWT, type JSONWebKeySet } from "jose";
import { beforeAll, describe, expect, it } from "vitest";
import { identify, isLoopback, Unauthenticated } from "./llamaServeIdentity";

const APP = "pv-calc";
const ISS = "https://apps.example";
let key: CryptoKey, other: CryptoKey, jwks: JSONWebKeySet;

beforeAll(async () => {
  const kp = await generateKeyPair("ES256");
  key = kp.privateKey;
  other = (await generateKeyPair("ES256")).privateKey;
  jwks = { keys: [{ ...(await exportJWK(kp.publicKey)), kid: "k1", alg: "ES256", use: "sig" }] };
});

async function token(over: Record<string, unknown> = {}, signer?: CryptoKey) {
  const now = Math.floor(Date.now() / 1000);
  const claims: Record<string, unknown> = { sub: "u-anna", name: "Anna", groups: ["familie"], aud: APP, iss: ISS, iat: now, exp: now + 300, ...over };
  for (const k of Object.keys(claims)) if (claims[k] === undefined) delete claims[k];
  return new SignJWT(claims).setProtectedHeader({ alg: "ES256", kid: "k1" }).sign(signer ?? key);
}

const who = (h: Record<string, string>, peer = "100.64.0.5") => identify(h, peer, { appId: APP, jwks, issuer: ISS, ownerSub: "u-owner" });

describe("identify", () => {
  it("names the person from a valid token", async () => {
    const p = await who({ "X-Llama-Serve-Identity": await token() });
    expect([p.sub, p.name, p.groups, p.via, p.isOwner]).toEqual(["u-anna", "Anna", ["familie"], "door", false]);
  });

  it("owner on loopback without header", async () => {
    for (const peer of ["127.0.0.1", "::1", "::ffff:127.0.0.1"]) {
      const p = await who({}, peer);
      expect([p.sub, p.via, p.isOwner]).toEqual(["u-owner", "local", true]);
    }
  });

  it("owner logging in through the door is the owner", async () => {
    expect((await who({ "x-llama-serve-identity": await token({ sub: "u-owner" }) })).isOwner).toBe(true);
  });

  it("works with fetch-style Headers", async () => {
    const h = new Headers({ "X-Llama-Serve-Identity": await token() });
    expect((await identify(h, "100.64.0.5", { appId: APP, jwks, issuer: ISS })).sub).toBe("u-anna");
  });

  it("no header from elsewhere is nobody", async () => {
    await expect(who({})).rejects.toBeInstanceOf(Unauthenticated);
  });

  const bad: Record<string, () => Promise<string>> = {
    "other key": () => token({}, other),
    "other app": () => token({ aud: "storytime" }),
    expired: () => token({ exp: Math.floor(Date.now() / 1000) - 120 }),
    "wrong issuer": () => token({ iss: "https://evil.example" }),
    "no sub": () => token({ sub: undefined }),
    tampered: async () => (await token()) + "x",
    unsigned: async () => {
      const b64 = (o: object) => Buffer.from(JSON.stringify(o)).toString("base64url");
      return `${b64({ alg: "none" })}.${b64({ sub: "u-anna", aud: APP, iss: ISS, exp: Math.floor(Date.now() / 1000) + 60 })}.`;
    },
    garbage: async () => "not-a-token",
  };
  for (const [name, make] of Object.entries(bad)) {
    it(`bad token (${name}) is nobody, even from loopback`, async () => {
      for (const peer of ["100.64.0.5", "127.0.0.1"]) {
        await expect(who({ "X-Llama-Serve-Identity": await make() }, peer)).rejects.toBeInstanceOf(Unauthenticated);
      }
    });
  }

  it("header without keys is nobody", async () => {
    const t = await token();
    delete process.env.LLAMA_SERVE_IDENTITY_JWKS;
    await expect(identify({ "X-Llama-Serve-Identity": t }, "100.64.0.5", { appId: APP })).rejects.toBeInstanceOf(Unauthenticated);
  });

  it("loopback detection", () => {
    expect(["127.0.0.1", "127.5.0.1", "::1", "[::1]", "::ffff:127.0.0.1"].every(isLoopback)).toBe(true);
    expect(["100.64.0.5", "192.168.1.40", "", undefined, "127.0.0.1.evil", "::ffff:10.0.0.1"].some((p) => isLoopback(p as string))).toBe(false);
  });

  describe("through Tailscale Serve (connects from loopback too)", () => {
    it("owner's login is the owner", async () => {
      process.env.LLAMA_SERVE_OWNER_TAILSCALE_LOGIN = "owner@github";
      const p = await who({ "Tailscale-User-Login": "owner@github" }, "127.0.0.1");
      expect([p.isOwner, p.sub, p.via]).toEqual([true, "u-owner", "tailscale"]);
    });
    it("other tailnet user is not the owner", async () => {
      process.env.LLAMA_SERVE_OWNER_TAILSCALE_LOGIN = "owner@github";
      await expect(who({ "Tailscale-User-Login": "friend@example.com" }, "127.0.0.1")).rejects.toBeInstanceOf(Unauthenticated);
    });
    it("unknown owner login means nobody", async () => {
      delete process.env.LLAMA_SERVE_OWNER_TAILSCALE_LOGIN;
      await expect(who({ "Tailscale-User-Login": "owner@github" }, "127.0.0.1")).rejects.toBeInstanceOf(Unauthenticated);
    });
    it("funnel is never the owner", async () => {
      process.env.LLAMA_SERVE_OWNER_TAILSCALE_LOGIN = "owner@github";
      await expect(who({ "Tailscale-Funnel-Request": "?1" }, "127.0.0.1")).rejects.toBeInstanceOf(Unauthenticated);
    });
  });

  describe("door = Pomerium", () => {
    it("accepts X-Pomerium-Jwt-Assertion", async () => {
      expect((await who({ "X-Pomerium-Jwt-Assertion": await token() })).sub).toBe("u-anna");
    });
    it("audience from the environment", async () => {
      process.env.LLAMA_SERVE_IDENTITY_AUDIENCE = "sonne.apps.example";
      try {
        expect((await who({ "X-Pomerium-Jwt-Assertion": await token({ aud: "sonne.apps.example" }) })).sub).toBe("u-anna");
        await expect(who({ "X-Pomerium-Jwt-Assertion": await token() })).rejects.toBeInstanceOf(Unauthenticated);
      } finally {
        delete process.env.LLAMA_SERVE_IDENTITY_AUDIENCE;
      }
    });
    it("forged assertion is nobody, even from loopback", async () => {
      await expect(who({ "X-Pomerium-Jwt-Assertion": await token({}, other) }, "127.0.0.1")).rejects.toBeInstanceOf(Unauthenticated);
    });
  });
});
