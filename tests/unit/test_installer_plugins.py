"""installer/plugins.py -- `./ai-serve plugin add <git-url>`: only remote repositories, a plain folder name, nothing
overwritten, the manifest checked before the plugin counts."""
import json

import pytest

from installer import plugins as p


@pytest.mark.parametrize("url", ["https://github.com/someone/my-plugin", "https://github.com/someone/my-plugin.git",
                                 "https://git.example.org:8443/group/sub/repo", "git@github.com:someone/repo.git",
                                 "ssh://git@host.example/owner/repo.git"])
def test_good_urls(url):
    assert p.check_url(url) == url


@pytest.mark.parametrize("url", ["", "/home/me/repo", "./repo", "~/repo", "file:///etc", "file://host/x",
                                 "--upload-pack=touch /tmp/x", "-c core.sshCommand=x", "https://github.com/a/b c",
                                 "https://github.com/a/../b", "ext::sh -c touch% /tmp/pwned", "http://github.com/a/b",
                                 "https://github.com/a/b\n", "ftp://host/repo"])
def test_refused_urls(url):
    with pytest.raises(p.PluginError):
        p.check_url(url)


def test_names():
    assert p.name_for("https://github.com/someone/My-Plugin.git") == "my-plugin"
    assert p.name_for("git@github.com:someone/repo.git", name="other") == "other"
    with pytest.raises(p.PluginError):
        p.name_for("https://github.com/someone/repo", name="../evil")


def _fake_clone(manifest):
    def clone(url, dest):
        dest.mkdir()
        (dest / "run.sh").write_text("#!/bin/sh\n")
        (dest / "run.sh").chmod(0o755)
        if manifest is not None:
            (dest / "llama-serve.service.json").write_text(json.dumps(manifest))
    return clone


GOOD = {"id": "kokoro", "name": "Kokoro", "command": ["./run.sh"], "port": 8077, "match": ["kokoro"],
        "ui": False, "identity": "none", "provides": {"tts": {"catalog": "/v1/engines", "generate": "/v1/generate"}}}


def test_add_checks_the_manifest(tmp_path):
    info = p.add("https://github.com/x/kokoro-plugin", tmp_path, clone=_fake_clone(GOOD))
    assert (info["id"], info["provides"], info["setup"]) == ("kokoro", ["tts"], False)
    assert (tmp_path / "kokoro-plugin" / "llama-serve.service.json").is_file()
    with pytest.raises(p.PluginError):     # never over an existing folder
        p.add("https://github.com/x/kokoro-plugin", tmp_path, clone=_fake_clone(GOOD))


def test_a_clone_without_valid_manifest_is_removed_again(tmp_path):
    with pytest.raises(p.PluginError):
        p.add("https://github.com/x/not-a-plugin", tmp_path, clone=_fake_clone(None))
    assert not (tmp_path / "not-a-plugin").exists()
    with pytest.raises(p.PluginError):
        p.add("https://github.com/x/bad", tmp_path, clone=_fake_clone({**GOOD, "command": ["/bin/sh"]}))
    assert not (tmp_path / "bad").exists()


def test_listing(tmp_path):
    p.add("https://github.com/x/kokoro-plugin", tmp_path, clone=_fake_clone(GOOD))
    (tmp_path / "junk").mkdir()
    rows = {r["dir"].rsplit("/", 1)[-1]: r for r in p.listing(tmp_path)}
    assert rows["kokoro-plugin"]["id"] == "kokoro" and "error" in rows["junk"]
