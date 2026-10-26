```python
from .config import FINDINGS


# Decision thresholds
UNCERTAIN_IS_POSITIVE = False

# Minimum severity required when state == "present".
# "unspecified" severity always counts as positive.
MIN_SEVERITY = {f: "mild" for f in FINDINGS}
MIN_SEVERITY.update({
    "Medial OA": "severe",
    "Lateral OA": "severe",
    "PF OA": "severe",
})

RANK = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}


def to_binary(f, entry):
    s, sev = entry["state"], entry["severity"]

    if s == "uncertain":
        return int(UNCERTAIN_IS_POSITIVE)

    if s != "present":
        return 0

    if sev == "unspecified":
        return 1

    return int(RANK[sev] >= RANK[MIN_SEVERITY[f]])
```