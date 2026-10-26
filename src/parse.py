```python
import json
import re

from .config import FINDINGS, STATES, SEVERITIES


def parse_output(raw):
    """Return (dict of 12 validated entries, n_invalid_fields, parse_ok)."""

    default = {
        "state": "not_mentioned",
        "severity": "none",
        "evidence": "",
    }

    text = re.sub(r"```(?:json)?", "", raw).strip()
    i, j = text.find("{"), text.rfind("}")

    try:
        obj = json.loads(text[i:j + 1])
    except Exception:
        return {
            f: dict(default) for f in FINDINGS
        }, len(FINDINGS), False

    clean, bad = {}, 0

    for f in FINDINGS:
        e = obj.get(f)

        if (
            isinstance(e, dict)
            and e.get("state") in STATES
            and e.get("severity") in SEVERITIES
        ):
            clean[f] = {
                "state": e["state"],
                "severity": e["severity"],
                "evidence": str(e.get("evidence", "")),
            }
        else:
            clean[f] = dict(default)
            bad += 1

    return clean, bad, True
```