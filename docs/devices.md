# Adding devices (other machines as compute)

ai-serve can use other machines of yours – a MacBook, a second PC, a Raspberry Pi, an Android phone with Termux – as
extra compute for language models. The machine running ai-serve stays the **coordinator**: apps keep talking to it,
and it sends a request to a device only when it can't serve the model itself (no room on its GPUs, or the model file
only exists on the device).

The machines talk over a private network. This guide uses **Tailscale**; WireGuard, Nebula, NetBird, ZeroTier and
Headscale are meant to work too (see the end).

> **Tried so far:** a Linux PC with NVIDIA (also with a Vulkan and a CPU build of llama.cpp) and a MacBook with Apple
> Silicon (Metal build, autostart as a LaunchAgent, budget changed from the dashboard). **Not tried yet:** Raspberry Pi,
> Android/Termux, AMD/ROCm builds, and anything without Tailscale – those paths exist in the code and have unit tests,
> but no real device has run them.

## 1. Tailscale on both machines

1. Install Tailscale on the coordinator and on the new device (<https://tailscale.com/download>) and sign in **with
   the same account** on both. Devices of other people shared into your tailnet are refused on purpose.
2. On the coordinator, add to `ai-serve.local.env` (in the ai-serve folder; create it from `ai-serve.example.env` if
   it doesn't exist):

   ```bash
   NETWORK_PROVIDER=tailscale
   ```

3. `./ai-serve restart`

From now on your own Tailscale devices may use the dashboard and the API too (`http://<coordinator>:8030`); everything
else on your local network still needs `LLAMA_SERVE_API_KEY`.

## 2. Add the device

1. In the dashboard: **Compute & models → Add a machine**. It shows a one-time command (the code works once and
   expires after 15 minutes).
2. Run that command **on the new device** (needs Python 3.9 or newer – on a Mac `python3` comes with the Xcode command
   line tools, on Termux `pkg install python`). It
   - downloads the agent (`agent.py`, a single file, standard library only) from the coordinator,
   - checks its SHA-256,
   - registers the device with the code and writes its settings to `~/.config/llama-serve-node/config.json`.
3. Let the agent start with the device:

   ```bash
   python3 agent.py --install-service
   ```

   macOS: a LaunchAgent; Linux: a `systemd --user` service (on a machine nobody logs into, also run
   `loginctl enable-linger $USER`); Android: Termux:Boot (not tried yet). `--uninstall-service` undoes it.

The device now appears under **Compute & models** with its accelerators.

## 3. llama.cpp and models on the device

On the device's card, under **Manage: llama.cpp, models, files**:

- **Install llama.cpp** – the coordinator picks the right prebuilt release for the device (Metal on Apple Silicon,
  CUDA/ROCm/Vulkan on Linux, CPU otherwise); the agent downloads it from GitHub and checks its SHA-256.
- **Send model** – a model from the coordinator or a Hugging Face repo + file. The device downloads from whichever
  source is faster, checked against one SHA-256. Nothing is ever deleted automatically: if a model doesn't fit, the
  device says so and lists its downloads; delete them yourself with **Delete**.
- **Budget** – how much memory ai-serve may use there. Default: all of a dedicated GPU, **half of the RAM** on
  machines where GPU and system share it (Apple Silicon, CPU-only). Change it any time.

You don't have to send models by hand: if a request needs a model the coordinator can't run, it copies it to the
best-fitting device in the background and answers `503` with `Retry-After` meanwhile.

## Remove a device

**Remove device** on its card. The agent keeps running on the device until you stop it there
(`python3 agent.py --uninstall-service`); it just isn't used any more.

## When something doesn't work

| What you see | Why / what to do |
|---|---|
| "Adding devices needs a private network …" | Step 1 is missing: `NETWORK_PROVIDER=tailscale` and a restart. |
| The command can't reach the coordinator | Is Tailscale up on both (`tailscale status`)? Same account? Open the dashboard through its Tailscale name/IP and try again. |
| `403` on enroll | The code was used already or expired – make a new one. Or the device isn't on your tailnet / belongs to someone else. |
| The device shows up but stays offline | Is the agent running (`--install-service`, or start `python3 agent.py` by hand)? Its log: `~/Library/Logs/llama-serve-node.log` (macOS), `journalctl --user -u llama-serve-node` (Linux). |
| "no llama.cpp" when starting a model there | Install llama.cpp on the device first (step 3). |

Updating the agent: download `agent.py` again from `http://<coordinator>:8030/api/cluster/agent.py` and restart it.

## Without Tailscale (not tried on real machines yet)

Any overlay network where an address belongs to exactly one device (WireGuard, Nebula, NetBird, ZeroTier,
Headscale) is meant to work with `NETWORK_PROVIDER=static` on the coordinator and a `network.json` next to it
(`NETWORK_CONFIG_FILE` to put it elsewhere):

```json
{
  "networks": ["10.8.0.0/24"],
  "owner": "me",
  "hostname": "10.8.0.1",
  "peers": {"10.8.0.2": "me", "10.8.0.3": "me"}
}
```

`networks`: the overlay's private ranges; `owner`: who owns the coordinator; `hostname`: the coordinator's overlay
address (used in the one-time command); `peers`: which overlay address is whose – only addresses mapped to the owner
may use the dashboard and enroll. On the device, the agent's config needs the same view of the network:

```json
"network": {"provider": "static", "networks": ["10.8.0.0/24"], "peers": {"10.8.0.1": "me"}}
```

and `--listen <the device's overlay address>` when enrolling (there is no `tailscale ip` to ask).
