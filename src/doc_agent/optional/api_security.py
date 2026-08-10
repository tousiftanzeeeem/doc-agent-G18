"""OPTIONAL — API auth + rate limiting
Activate only if your data speciality or NFR requires it (profiles secure/high-stakes). Off by default; CI does not require impl."""
from __future__ import annotations

def require_auth(token: str) -> bool: raise NotImplementedError("optional: auth")
def rate_limit(client: str) -> bool: raise NotImplementedError("optional: rate limit")

