#!/usr/bin/env python3
"""Speaker-similarity analysis over voices/casting/pool60/ (+ optionally the
original 12-voice casting set), so near-duplicate VoiceDesign samples can be
grouped and only one representative kept per cluster.

Embeddings come from Chatterbox's own bundled speaker/voice encoder
(Resemblyzer-derived d-vector network, ve.pt from the already-downloaded
ResembleAI/chatterbox HF snapshot) -- run under venv-chatterbox (has
torch/librosa already; no new install needed). This is a *different* model
from the TTS engines themselves, used purely as a fixed acoustic-similarity
metric, not tied to any one engine's own conditioning.

Clustering: agglomerative, cosine distance, average linkage, distance
threshold chosen from the empirical pairwise-distance distribution (see
printed histogram) rather than a guessed constant.

Output: voices/casting/pool60/similarity.json --
  - "embeddings": not stored (large/opaque), only derived results
  - "clusters": list of {members: [slug...], centroid_slug} for clusters with >1 member
  - "singletons": slugs that didn't cluster with anything
  - "nearest_neighbors": {slug: [(other_slug, cosine_sim), ...]} top-5 per voice
"""
import glob
import json
import os
import sys

POOL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "casting", "pool60")
VE_CKPT = os.path.expanduser("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots")


def find_ve_ckpt():
    for root, _, files in os.walk(VE_CKPT):
        if "ve.pt" in files:
            return os.path.join(root, "ve.pt")
    raise FileNotFoundError("ve.pt not found under the ResembleAI/chatterbox HF cache -- run Chatterbox once first.")


def main():
    import numpy as np
    import librosa
    import torch
    from chatterbox.models.voice_encoder import VoiceEncoder

    ckpt = find_ve_ckpt()
    print(f"Loading voice encoder from {ckpt}", file=sys.stderr)
    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    ve.eval()

    wav_paths = sorted(glob.glob(os.path.join(POOL_DIR, "*_en.wav")))
    if not wav_paths:
        print(f"No wavs found in {POOL_DIR}", file=sys.stderr)
        sys.exit(1)

    # Pre-resample to the encoder's own rate (16kHz, S3_SR) via librosa.load's
    # default soxr backend -- embeds_from_wavs()'s own internal resample step
    # hardcodes res_type="kaiser_fast", which needs the optional `resampy`
    # package (not installed here); passing already-16kHz audio in makes it
    # skip that branch entirely (sample_rate == self.hp.sample_rate).
    slugs, wavs = [], []
    for p in wav_paths:
        slug = os.path.basename(p).replace("_en.wav", "")
        y, sr = librosa.load(p, sr=16000, mono=True)
        slugs.append(slug)
        wavs.append((y, sr))

    print(f"Embedding {len(slugs)} clips...", file=sys.stderr)
    embeds = []
    for slug, (y, sr) in zip(slugs, wavs):
        e = ve.embeds_from_wavs([y], sample_rate=sr, as_spk=True)
        embeds.append(e)
    embeds = np.stack(embeds)  # (N, E)
    norm = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    sim = norm @ norm.T  # cosine similarity matrix
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    # Empirical distribution of off-diagonal pairwise distances, to pick a
    # threshold instead of guessing one.
    iu = np.triu_indices(len(slugs), k=1)
    off_diag = dist[iu]
    print("Pairwise cosine-distance stats: "
          f"min={off_diag.min():.3f} p10={np.percentile(off_diag,10):.3f} "
          f"median={np.median(off_diag):.3f} p90={np.percentile(off_diag,90):.3f} "
          f"max={off_diag.max():.3f}", file=sys.stderr)

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Threshold: a FIXED cosine-distance cutoff (0.137), not a percentile of
    # this run's own pairwise-distance distribution. Percentile-of-this-batch
    # was tried first and is a methodological bug: "closest 10% of pairs" is
    # always ~10% of pairs *by construction*, regardless of how much genuine
    # redundancy is actually in the pool -- confirmed live when adding the
    # 60-voice OmniVoice batch on top of the original 60 Qwen voices made the
    # p10 threshold loosen enough to produce a 22-member cluster and only 4
    # singletons out of 120, versus 11 clusters/17 singletons out of 60 on
    # the Qwen-only run. A fixed threshold makes cluster membership mean the
    # same acoustic-distance thing regardless of pool size or composition,
    # so results are comparable across reruns as the pool grows. 0.137 is
    # that original Qwen-only run's p10 value, kept as the calibration point.
    threshold = 0.137
    Z = linkage(squareform(dist, checks=False), method="average")
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")

    clusters = {}
    for slug, cid in zip(slugs, cluster_ids):
        clusters.setdefault(int(cid), []).append(slug)

    grouped = [members for members in clusters.values() if len(members) > 1]
    singletons = [members[0] for members in clusters.values() if len(members) == 1]

    # Nearest neighbors per voice (top 5), for browsing even outside formal clusters.
    nn = {}
    for i, slug in enumerate(slugs):
        order = np.argsort(dist[i])
        top = [(slugs[j], float(sim[i, j])) for j in order if j != i][:5]
        nn[slug] = top

    out = {
        "threshold_distance": threshold,
        "n_voices": len(slugs),
        "clusters": [{"members": m} for m in grouped],
        "singletons": singletons,
        "nearest_neighbors": nn,
    }
    out_path = os.path.join(POOL_DIR, "similarity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n{len(grouped)} clusters (>1 member), {len(singletons)} singletons out of {len(slugs)} voices.")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
