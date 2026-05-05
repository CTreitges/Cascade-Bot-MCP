"""Plan v5 R7 — Federation Smoke."""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cascade.federation import (
    SyncFile,
    SyncManifest,
    build_manifest,
    hash_file,
    issue_token,
    scan_pii,
    strip_pii,
    verify_token,
)


def passed(label):
    print(f"  ✅ {label}")


def fail(label, why):
    print(f"  ❌ {label}: {why}")
    raise SystemExit(1)


def test_pii_scan_and_strip():
    print("\n[1] PII scan + strip")
    text = """
    api_key=sk-ant-abc123def456ghi789jkl012mno345pq
    GITHUB_PAT=ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAA1234
    plain content here
    password = "supersecret123"
    """
    scan = scan_pii(text)
    assert scan.has_pii
    kinds = [k for k, _ in scan.findings]
    print(f"     findings: {kinds}")
    assert "anthropic" in kinds
    assert "github_pat" in kinds
    stripped = strip_pii(text)
    assert "sk-ant-" not in stripped
    assert "ghp_" not in stripped
    assert "<REDACTED>" in stripped
    passed("scan + strip funktional")


def test_token_roundtrip():
    print("\n[2] token issue + verify roundtrip")
    secret = b"shared-key-known-only-to-trusted-machines-1234"
    tok = issue_token(
        issuer="vps-srv1577611",
        audience="windows-laptop-chris",
        secret_key=secret,
        payload={"purpose": "sync", "items": 5},
    )
    ok, reason = verify_token(
        tok, secret_key=secret, expected_audience="windows-laptop-chris",
    )
    assert ok, reason
    passed(f"valid token: {reason}")


def test_token_replay_protection():
    print("\n[3] replay protection (gleicher token 2× → invalid)")
    secret = b"k"
    tok = issue_token(issuer="a", audience="b", secret_key=secret)
    ok1, _ = verify_token(tok, secret_key=secret, expected_audience="b")
    ok2, reason2 = verify_token(tok, secret_key=secret, expected_audience="b")
    assert ok1
    assert not ok2
    assert "replay" in reason2.lower()
    passed(f"replay erkannt: {reason2}")


def test_token_signature_mismatch():
    print("\n[4] verschiedene Keys → signature-mismatch")
    tok = issue_token(issuer="a", audience="b", secret_key=b"key1")
    ok, reason = verify_token(tok, secret_key=b"key2", expected_audience="b")
    assert not ok
    assert "signature" in reason.lower()
    passed(f"key-mismatch erkannt: {reason}")


def test_token_expiry():
    print("\n[5] expired token erkannt (ohne clock-skew)")
    tok = issue_token(issuer="a", audience="b", secret_key=b"k", ttl_seconds=1)
    time.sleep(1.5)
    # explicit allow_clock_skew_s=0 — sonst greift der 30s Default-Tolerance
    ok, reason = verify_token(tok, secret_key=b"k", expected_audience="b", allow_clock_skew_s=0)
    assert not ok
    assert "expired" in reason.lower()
    passed(f"expired: {reason}")


def test_audience_mismatch():
    print("\n[6] audience mismatch")
    tok = issue_token(issuer="a", audience="b", secret_key=b"k")
    ok, reason = verify_token(tok, secret_key=b"k", expected_audience="c")
    assert not ok
    assert "audience" in reason.lower()
    passed(f"audience-mismatch: {reason}")


def test_hash_file():
    print("\n[7] hash_file SHA256 + size")
    tmp = Path(tempfile.mkdtemp(prefix="cascade-fed-"))
    p = tmp / "test.txt"
    p.write_text("Hello World\n")
    digest, size = hash_file(p)
    assert size == 12
    assert len(digest) == 64
    print(f"     {digest[:12]}… size={size}")
    shutil.rmtree(tmp, ignore_errors=True)
    passed("hash + size correct")


def test_build_manifest_skips_pii():
    print("\n[8] manifest: skip-pii in textfile")
    tmp = Path(tempfile.mkdtemp(prefix="cascade-fed-"))
    safe = tmp / "safe.txt"
    safe.write_text("just plain text without secrets")
    pii = tmp / "secrets.env"
    pii.write_text("ANTHROPIC_API_KEY=sk-ant-AAAAAAAAAAAAAAAAAAAAAAAAAAAA1234\n")
    binary = tmp / "data.xlsx"
    binary.write_bytes(b"\x00\x01\x02\x03" * 100)

    m = build_manifest(
        source_machine="vps",
        target_machine="windows",
        files=[safe, pii, binary],
        base_dir=tmp,
        skip_pii=True,
    )
    print(f"     {len(m.files)} files, pii_skipped={m.pii_skipped}")
    by_intent = {sf.relative_path: sf.intent for sf in m.files}
    print(f"     intents: {by_intent}")
    assert by_intent["safe.txt"] == "sync"
    assert by_intent["secrets.env"] == "skip-pii"
    assert by_intent["data.xlsx"] == "sync"  # binär nicht gescannt
    assert m.pii_skipped == 1
    shutil.rmtree(tmp, ignore_errors=True)
    passed("PII-file marked skip-pii, plain + binary marked sync")


def main():
    print("=" * 60)
    print("  Plan v5 R7 — Federation Smoke")
    print("=" * 60)
    test_pii_scan_and_strip()
    test_token_roundtrip()
    test_token_replay_protection()
    test_token_signature_mismatch()
    test_token_expiry()
    test_audience_mismatch()
    test_hash_file()
    test_build_manifest_skips_pii()
    print("\n" + "=" * 60)
    print("  ✅ Alle 8 Tests grün")
    print("=" * 60)


if __name__ == "__main__":
    main()
