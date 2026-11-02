"""How long a starting llama-server may take (backend/model_manager.py LoadWatch, 2026-10-24): a cold 11.5 GB file
took 240 s just to be read from disk, and the old fixed 60 s gave up every time -- wait while the load progresses."""
from backend.model_manager import LoadWatch, _load_progress


def test_a_slow_but_progressing_load_is_waited_for():
    w = LoadWatch(0)
    for t in range(1, 300):
        assert w.verdict(t, (t * 50_000_000, 100)) is None   # reading ~50 MB/s for 5 minutes


def test_a_stalled_load_gives_up_after_a_minute_without_progress():
    w = LoadWatch(0)
    assert w.verdict(10, (1, 1)) is None
    assert w.verdict(69, (1, 1)) is None
    why = w.verdict(70, (1, 1))
    assert why and "no loading progress for 60s" in why


def test_unknown_progress_counts_as_none():
    w = LoadWatch(0)
    assert w.verdict(59, None) is None and w.verdict(60, None)


def test_there_is_an_upper_limit():
    w = LoadWatch(0)
    for t in range(1, LoadWatch.MAX_S, 30):
        assert w.verdict(t, (t * 10**9, 0)) is None
    assert "within" in w.verdict(LoadWatch.MAX_S, (LoadWatch.MAX_S * 10**9, 0))


def test_a_trickle_of_reads_is_no_progress():
    # 2026-10-26, V-Noveller: the external SSD reset over USB every ~40 s; after each reset a few MB came through,
    # which kept resetting the stall clock -- llama-server sat in D state for 3 min until it was unloaded by hand
    w = LoadWatch(0)
    for t in range(1, 70):
        why = w.verdict(t, (47_000_000 + (t // 40) * 2_000_000, 1442))
        if why:
            break
    assert t == 61 and "disk reads stalled" in why and "2 MB" in why


def test_log_output_counts_as_progress():
    w = LoadWatch(0)
    for t in range(1, 200):
        assert w.verdict(t, (5_000_000, 1000 + t)) is None    # e.g. warmup lines while nothing is read


def test_progress_of_this_process(tmp_path):
    import os
    log = tmp_path / "x.log"
    log.write_text("abc")
    read, size = _load_progress(os.getpid(), str(log))
    assert size == 3 and read >= 0
    assert _load_progress(2 ** 22 + 12345, str(tmp_path / "missing")) == (0, 0)
