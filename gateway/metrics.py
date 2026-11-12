"""In-process counters exposed in Prometheus text format."""

from __future__ import annotations

from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        self._counters[(name, tuple(sorted(labels.items())))] += value

    def get(self, name: str, **labels: str) -> float:
        return self._counters.get((name, tuple(sorted(labels.items()))), 0.0)

    def total(self, name: str) -> float:
        return sum(v for (n, _), v in self._counters.items() if n == name)

    def render(self) -> str:
        lines, seen = [], set()
        for (name, labels), value in sorted(self._counters.items()):
            if name not in seen:
                seen.add(name)
                lines.append(f"# TYPE {name} counter")
            rendered = ",".join(f'{k}="{v}"' for k, v in labels)
            lines.append(f"{name}{{{rendered}}} {value:g}" if rendered else f"{name} {value:g}")
        return "\n".join(lines) + "\n"
