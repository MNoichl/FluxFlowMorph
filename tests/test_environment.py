from flowmorph_klein.environment import redact_secrets


def test_hugging_face_token_is_redacted():
    secret = "hf_" + "AbCd1234" * 4
    output = redact_secrets(f"failure URL?token={secret}")
    assert secret not in output
    assert "redacted" in output

