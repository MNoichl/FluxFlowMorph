import sys
from types import ModuleType, SimpleNamespace

from flowmorph_klein.environment import redact_secrets, resolve_hf_token


def test_hugging_face_token_is_redacted():
    secret = "hf_" + "AbCd1234" * 4
    output = redact_secrets(f"failure URL?token={secret}")
    assert secret not in output
    assert "redacted" in output


def test_colab_secret_timeout_falls_back_to_interactive_login(monkeypatch):
    class TimeoutException(Exception):
        pass

    def timed_out_secret(_key):
        raise TimeoutException("Secrets are available only in the Colab UI")

    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.userdata = SimpleNamespace(get=timed_out_secret)
    google.colab = colab

    hub = ModuleType("huggingface_hub")
    hub.login = lambda *, skip_if_logged_in: None
    hub.get_token = lambda: "hf_interactive_test_token"

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    authentication = resolve_hf_token()

    assert authentication.source == "interactive_login"
    assert authentication.token == "hf_interactive_test_token"
