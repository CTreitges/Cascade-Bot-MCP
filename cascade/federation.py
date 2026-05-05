"""Plan v5 R7 — Federation: Zero-Trust-Sync zwischen Cascade-Instanzen.

Inspiration: Ruflo's federation-Plugin. User hat aktuell Multi-Maschine-
Setup (Windows+VPS) mit rsync+sshpass. Diese Module bietet sicherere +
auditierbare Sync-Primitives:

  - PII-Stripping: keyword-based-Filter für Secrets/Credentials/PII
    bevor Daten zwischen Maschinen wandern
  - Trust-Token: HMAC-SHA256-signiert, mit timestamp + nonce-replay-
    Protection (kein full-mTLS, aber vergleichbar leicht zu deployen)
  - Sync-Manifest: Liste der zu transferierenden Files mit hash + size
    + intent — auditable Logs

KEIN externer Daemon: das Modul exportiert Building-Blocks die ein
einfaches Sync-Skript (oder cascade-bot/cli) ergänzen kann. Existing
rsync+sshpass-Pipeline bleibt — federation strippt PII vor und
verifiziert das Manifest danach.

Bewusst minimal:
  - kein P2P-Routing (das wäre overkill)
  - kein Service-Discovery
  - kein Crypto-Geraffel jenseits HMAC (wer mTLS will, baut eigene)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


logger = logging.getLogger("cascade.federation")


# ──────────────────────────────────────────────────────────────────────
#  PII-Stripping
# ──────────────────────────────────────────────────────────────────────
# Patterns die zwischen Maschinen NICHT wandern sollen.
_PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # API-Keys
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("github_pat", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("aws_key", re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
    ("aws_secret", re.compile(r"\b[A-Za-z0-9/+=]{40}\b")),  # ggf. zu generisch — caller prüft
    ("telegram_bot", re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b")),
    # Bearer tokens generic
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9_\-.=]{20,}", re.IGNORECASE)),
    # E-Mail (vorsichtig — kein massiver Strip, nur logging-warning per default)
    # Passwort-Felder in env/.json
    ("password_kv", re.compile(r"(?i)(password|passwd|pwd|secret|token|api_key)\s*[:=]\s*['\"]?[^\s'\"<>]{6,}")),
]


@dataclass
class PIIScanResult:
    has_pii: bool
    findings: List[Tuple[str, str]] = field(default_factory=list)  # (kind, redacted-preview)


def scan_pii(text: str) -> PIIScanResult:
    """Sucht nach PII/Secrets-Patterns. Findet, redacted, returnt."""
    if not text:
        return PIIScanResult(has_pii=False)
    findings: List[Tuple[str, str]] = []
    for kind, rx in _PII_PATTERNS:
        for m in rx.finditer(text):
            redacted = m.group(0)[:8] + "***"
            findings.append((kind, redacted))
    return PIIScanResult(has_pii=bool(findings), findings=findings)


def strip_pii(text: str, replacement: str = "<REDACTED>") -> str:
    """Ersetzt alle gefundenen PII durch ein Replacement."""
    if not text:
        return text
    out = text
    for _kind, rx in _PII_PATTERNS:
        out = rx.sub(replacement, out)
    return out


# ──────────────────────────────────────────────────────────────────────
#  Trust-Token (HMAC-signiert)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TrustToken:
    """Eine kurzlebige Signatur die zwischen Maschinen ausgetauscht wird.
    Format-Beispiel: 'eyJ0c…' oder einfach JSON-Block."""
    issuer: str          # eindeutige Maschine-ID, z.B. "vps-srv1577611"
    audience: str        # ziel-maschine
    issued_at: float
    expires_at: float
    nonce: str           # 32-char hex
    payload: Dict        # additional claims
    signature: str       # hex-encoded HMAC


def issue_token(
    *,
    issuer: str,
    audience: str,
    secret_key: bytes,
    payload: Optional[Dict] = None,
    ttl_seconds: int = 300,
) -> TrustToken:
    """Erstellt einen signierten Token. secret_key wird auf BEIDEN Seiten
    geteilt (out-of-band, z.B. in .env als TRUSTED_FED_KEY)."""
    issued = time.time()
    expires = issued + ttl_seconds
    nonce = uuid.uuid4().hex
    body = {
        "issuer": issuer,
        "audience": audience,
        "issued_at": issued,
        "expires_at": expires,
        "nonce": nonce,
        "payload": payload or {},
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    sig = hmac.new(secret_key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return TrustToken(**body, signature=sig)


_NONCE_CACHE: Set[str] = set()
_NONCE_CACHE_MAX = 10_000


def verify_token(
    token: TrustToken,
    *,
    secret_key: bytes,
    expected_audience: str,
    allow_clock_skew_s: int = 30,
) -> Tuple[bool, str]:
    """Returns (valid, reason). Prüft signature, expiry, audience, replay."""
    body = {
        "issuer": token.issuer,
        "audience": token.audience,
        "issued_at": token.issued_at,
        "expires_at": token.expires_at,
        "nonce": token.nonce,
        "payload": token.payload,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    expected_sig = hmac.new(secret_key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, token.signature):
        return False, "signature mismatch"
    now = time.time()
    if token.expires_at + allow_clock_skew_s < now:
        return False, f"expired ({int(now - token.expires_at)}s ago)"
    if token.issued_at - allow_clock_skew_s > now:
        return False, f"issued in future ({int(token.issued_at - now)}s)"
    if token.audience != expected_audience:
        return False, f"audience mismatch (got {token.audience})"
    if token.nonce in _NONCE_CACHE:
        return False, "replay (nonce already seen)"
    # cache nonce
    if len(_NONCE_CACHE) >= _NONCE_CACHE_MAX:
        _NONCE_CACHE.clear()
    _NONCE_CACHE.add(token.nonce)
    return True, "ok"


# ──────────────────────────────────────────────────────────────────────
#  Sync-Manifest (Audit)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SyncFile:
    relative_path: str
    sha256: str
    size_bytes: int
    intent: str = "sync"   # "sync" / "delete" / "skip-pii"


@dataclass
class SyncManifest:
    sync_id: str
    created_at: float
    source_machine: str
    target_machine: str
    files: List[SyncFile] = field(default_factory=list)
    pii_skipped: int = 0
    total_bytes: int = 0

    def add_file(self, sf: SyncFile) -> None:
        self.files.append(sf)
        self.total_bytes += sf.size_bytes
        if sf.intent == "skip-pii":
            self.pii_skipped += 1


def hash_file(path: Path, *, chunk_size: int = 65536) -> Tuple[str, int]:
    """SHA256 + size. Returns (hex_digest, size_bytes)."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def build_manifest(
    *,
    source_machine: str,
    target_machine: str,
    files: Iterable[Path],
    base_dir: Optional[Path] = None,
    skip_pii: bool = True,
    pii_text_extensions: Tuple[str, ...] = (".txt", ".md", ".json", ".yaml", ".yml", ".env", ".conf", ".ini"),
) -> SyncManifest:
    """Erstellt ein Manifest für die genannten Files. Bei skip_pii=True
    werden Files mit textuellen Extensions zuerst gescannt — wenn PII
    gefunden, intent='skip-pii' (vom Caller dann nicht synct)."""
    sync_id = uuid.uuid4().hex[:12]
    manifest = SyncManifest(
        sync_id=sync_id,
        created_at=time.time(),
        source_machine=source_machine,
        target_machine=target_machine,
    )
    for p in files:
        if not p.exists() or not p.is_file():
            continue
        rel = str(p.relative_to(base_dir)) if base_dir else str(p)
        digest, size = hash_file(p)
        intent = "sync"
        if skip_pii and p.suffix.lower() in pii_text_extensions:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                scan = scan_pii(text)
                if scan.has_pii:
                    intent = "skip-pii"
                    logger.info(
                        f"manifest: {rel} → skip-pii ({len(scan.findings)} findings: "
                        f"{[k for k,_ in scan.findings[:3]]})"
                    )
            except Exception as e:
                logger.warning(f"manifest: PII-scan failed for {rel}: {e}")
        manifest.add_file(SyncFile(
            relative_path=rel,
            sha256=digest,
            size_bytes=size,
            intent=intent,
        ))
    return manifest


def manifest_to_dict(m: SyncManifest) -> Dict:
    return asdict(m)
