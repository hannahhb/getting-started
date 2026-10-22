"""Answer-preserving perturbations that code can make at test time.

Four types, each deterministic given a seed:

* ``typos``: swap, drop, or double a letter in a few words outside the maths.
* ``rename``: consistently rename single-letter variables inside ``$...$``.
* ``distract``: insert one irrelevant sentence with numbers from a fixed bank.
* ``rephrase``: reword the instruction (a lead-in sentence, verb synonyms).

None of them touches a number or a condition of the problem.
"""

from __future__ import annotations

import random
import re
import string

MATH = re.compile(r"(\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\])", re.DOTALL)
VARIABLE = re.compile(r"(?<![\\A-Za-z])([A-Za-z])(?![A-Za-z])")
DISTRACTORS = (
    "A local mathematics club, founded in {year}, has {n} members.",
    "The problem was first written on a whiteboard that is {n} centimetres wide.",
    "Last week the author solved {n} other puzzles before this one.",
    "The competition hall had {n} chairs, of which {k} were blue.",
    "A student copied this problem into a notebook with {n} pages.",
)
LEAD_INS = ("Solve the following problem.", "Consider the following question carefully.",
            "Here is a competition problem.", "Answer the following.")
VERBS = {"Find": "Determine", "Compute": "Calculate", "Determine": "Find", "Calculate": "Compute",
         "Evaluate": "Compute", "What is": "Determine"}


def _outside_math(text: str):
    """Split into (segment, is_math) pairs."""
    parts = MATH.split(text)
    return [(part, i % 2 == 1) for i, part in enumerate(parts)]


def typos(text: str, rng: random.Random, rate: float = 0.06) -> str:
    for _ in range(20):
        out, changed = [], 0
        for part, is_math in _outside_math(text):
            if is_math:
                out.append(part)
                continue
            words = re.split(r"(\s+)", part)
            for w, word in enumerate(words):
                if len(word) > 3 and word.isalpha() and rng.random() < rate:
                    i = rng.randrange(1, len(word) - 1)
                    edit = rng.choice(("swap", "drop", "double"))
                    if edit == "swap":
                        word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
                    elif edit == "drop":
                        word = word[:i] + word[i + 1:]
                    else:
                        word = word[:i] + word[i] + word[i:]
                    words[w], changed = word, changed + 1
            out.append("".join(words))
        if changed >= 2:
            return "".join(out)
    return "".join(out)


def rename(text: str, rng: random.Random) -> str | None:
    """Rename single-letter variables in maths consistently; None if there are none."""
    used = set()
    for part, is_math in _outside_math(text):
        if is_math:
            used.update(VARIABLE.findall(part))
    constants = {"e", "i"}  # Euler's number and the imaginary unit keep their names
    if not used - constants:
        return None
    mapping = {}
    for case_letters in (string.ascii_lowercase, string.ascii_uppercase):
        mine = sorted(v for v in used - constants if v in case_letters)
        free = [c for c in case_letters if c not in used and c not in "eiloIO"]
        rng.shuffle(free)
        mapping.update(zip(mine, free))
    if not mapping:
        return None
    out = []
    for part, is_math in _outside_math(text):
        out.append(VARIABLE.sub(lambda m: mapping.get(m.group(1), m.group(1)), part) if is_math else part)
    return "".join(out)


def distract(text: str, rng: random.Random) -> str:
    sentence = rng.choice(DISTRACTORS).format(year=rng.randint(1950, 2015), n=rng.randint(12, 480),
                                              k=rng.randint(2, 11))
    # Put it after the first sentence that ends outside maths, else in front.
    for match in re.finditer(r"[.!?](\s+)", text):
        before = text[: match.start()]
        if before.count("$") % 2 == 0:
            return text[: match.end()] + sentence + " " + text[match.end():]
    return sentence + " " + text


def rephrase(text: str, rng: random.Random) -> str:
    out = text
    for old, new in VERBS.items():
        pattern = re.compile(rf"(^|[.!?]\s+){re.escape(old)}\b")
        if pattern.search(out):
            out = pattern.sub(lambda m: m.group(1) + new, out, count=1)
            break
    return rng.choice(LEAD_INS) + " " + out


def make_perturbations(text: str, seed: int = 0) -> list[tuple[str, str]]:
    """The four (type, text) perturbations of ``text``; rename falls back to typos."""
    rng = random.Random(f"{seed}:{text}")
    renamed = rename(text, rng)
    return [
        ("typos", typos(text, rng)),
        ("rename", renamed) if renamed and renamed != text else ("typos2", typos(text, rng)),
        ("distract", distract(text, rng)),
        ("rephrase", rephrase(text, rng)),
    ]
