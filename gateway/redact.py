"""Mask common PII before a prompt leaves the gateway."""

from __future__ import annotations

import copy
import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_PHONE = re.compile(r"(?<![\w.+])\+?\d[\d ()-]{8,}\d(?![\w.])")


def _luhn(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    checksum = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(reversed(digits)))
    return 13 <= len(digits) <= 19 and checksum % 10 == 0


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts = {"email": 0, "card": 0, "phone": 0}

    def card(match: re.Match) -> str:
        if not _luhn(match.group(0)):
            return match.group(0)  # a long number that is not a card (an order id, a timestamp)
        counts["card"] += 1
        return "[CARD]"

    def count(kind: str, token: str):
        def replace(_match: re.Match) -> str:
            counts[kind] += 1
            return token

        return replace

    def phone(match: re.Match) -> str:
        raw = match.group(0)
        digits = sum(c.isdigit() for c in raw)
        # Heuristic: international prefix, visible separators, or a bare 10-digit number. A long
        # unbroken digit run is more likely an order id or timestamp than a phone number.
        looks_like_phone = raw.startswith("+") or digits == 10 or any(c in raw for c in " ()-")
        if not (10 <= digits <= 15 and looks_like_phone):
            return raw
        counts["phone"] += 1
        return "[PHONE]"

    text = _EMAIL.sub(count("email", "[EMAIL]"), text)
    text = _CARD.sub(card, text)
    text = _PHONE.sub(phone, text)
    return text, counts


def redact_messages(messages: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Returns a redacted copy; the caller's messages are not modified."""
    total = {"email": 0, "card": 0, "phone": 0}
    out = copy.deepcopy(messages)
    for message in out:
        parts = message["content"] if isinstance(message.get("content"), list) else [message]
        for part in parts:
            key = "content" if part is message else "text"
            if isinstance(part, dict) and isinstance(part.get(key), str):
                part[key], counts = redact(part[key])
                for kind, n in counts.items():
                    total[kind] += n
    return out, total
