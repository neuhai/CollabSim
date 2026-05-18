#!/usr/bin/env python3
"""Generate a stratified persona pool for LLM agent experiments.

Each persona has demographic attributes (age, gender, education, occupation, income)
plus a full Big Five personality profile (high/low per dimension), and a natural-
language description suitable for direct injection into an agent system prompt.

Usage:
    python scripts/generate_personas.py
    python scripts/generate_personas.py --count 60 --seed 7 --out prompts/persona_profiles.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ── Demographic attribute spaces ──────────────────────────────────────────────

AGE_RANGES = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

GENDERS = ["male", "female", "non-binary"]
# Approximate US adult distribution (~49.5% / ~49.5% / ~1% non-binary; Pew 2022).
GENDER_WEIGHTS: dict[str, float] = {
    "male": 49.5,
    "female": 49.5,
    "non-binary": 1.0,
}

EDUCATION_LEVELS = [
    "less than high school",
    "high school diploma",
    "some college",
    "bachelor's degree",
    "graduate degree",
]

OCCUPATIONS = [
    "student",
    "professional/technical",
    "management/executive",
    "service/sales",
    "skilled labor",
    "retired",
]

INCOME_LEVELS = ["low", "middle", "upper-middle", "high"]

# ── Big Five dimensions and levels ────────────────────────────────────────────

BIG_FIVE_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
BIG_FIVE_LEVELS = ["high", "low"]

# ── Demographic conditional constraints ───────────────────────────────────────

# Valid occupations per age range
OCCUPATION_BY_AGE: dict[str, list[str]] = {
    "18-24": ["student", "service/sales", "professional/technical"],
    "25-34": ["professional/technical", "service/sales", "management/executive", "student"],
    "35-44": ["professional/technical", "management/executive", "service/sales", "skilled labor"],
    "45-54": ["professional/technical", "management/executive", "service/sales", "skilled labor"],
    "55-64": ["professional/technical", "management/executive", "service/sales", "skilled labor", "retired"],
    "65+":   ["retired", "service/sales"],
}

# Valid education levels per (age_range, occupation) pair
EDUCATION_BY_AGE_OCC: dict[tuple[str, str], list[str]] = {
    ("18-24", "student"):                ["high school diploma", "some college"],
    ("18-24", "service/sales"):          ["less than high school", "high school diploma", "some college"],
    ("18-24", "professional/technical"): ["some college", "bachelor's degree"],

    ("25-34", "student"):                ["bachelor's degree", "graduate degree"],
    ("25-34", "professional/technical"): ["some college", "bachelor's degree", "graduate degree"],
    ("25-34", "management/executive"):   ["bachelor's degree", "graduate degree"],
    ("25-34", "service/sales"):          ["high school diploma", "some college", "bachelor's degree"],

    ("35-44", "professional/technical"): ["some college", "bachelor's degree", "graduate degree"],
    ("35-44", "management/executive"):   ["bachelor's degree", "graduate degree"],
    ("35-44", "service/sales"):          ["less than high school", "high school diploma", "some college", "bachelor's degree"],
    ("35-44", "skilled labor"):          ["less than high school", "high school diploma", "some college"],

    ("45-54", "professional/technical"): ["some college", "bachelor's degree", "graduate degree"],
    ("45-54", "management/executive"):   ["bachelor's degree", "graduate degree"],
    ("45-54", "service/sales"):          ["less than high school", "high school diploma", "some college", "bachelor's degree"],
    ("45-54", "skilled labor"):          ["less than high school", "high school diploma", "some college"],

    ("55-64", "professional/technical"): ["some college", "bachelor's degree", "graduate degree"],
    ("55-64", "management/executive"):   ["bachelor's degree", "graduate degree"],
    ("55-64", "service/sales"):          ["less than high school", "high school diploma", "some college", "bachelor's degree"],
    ("55-64", "skilled labor"):          ["less than high school", "high school diploma", "some college"],
    ("55-64", "retired"):                ["less than high school", "high school diploma", "some college", "bachelor's degree", "graduate degree"],

    ("65+",   "retired"):                ["less than high school", "high school diploma", "some college", "bachelor's degree", "graduate degree"],
    ("65+",   "service/sales"):          ["less than high school", "high school diploma", "some college"],
}

# Income pool per occupation (duplicates = higher probability)
INCOME_BY_OCC: dict[str, list[str]] = {
    "student":               ["low", "low", "low", "middle"],
    "professional/technical":["middle", "middle", "upper-middle", "upper-middle", "high"],
    "management/executive":  ["upper-middle", "upper-middle", "high", "high", "high"],
    "service/sales":         ["low", "low", "middle", "middle"],
    "skilled labor":         ["low", "middle", "middle"],
    "retired":               ["low", "middle", "middle", "upper-middle"],
}

# ── Description templates ─────────────────────────────────────────────────────

_AGE_PHRASE_GENDERED: dict[tuple[str, str], str] = {
    **{(a, g): f"in {p} {suf}"
       for a, suf in [
           ("18-24", "early twenties"),
           ("25-34", "late twenties or early thirties"),
           ("35-44", "mid-thirties to early forties"),
           ("45-54", "late forties to early fifties"),
           ("55-64", "late fifties to early sixties"),
           ("65+",   "mid-sixties or older"),
       ]
       for g, p in [("male", "his"), ("female", "her"), ("non-binary", "their")]
    }
}

_GENDER_NOUN:    dict[str, str] = {"male": "man",  "female": "woman",  "non-binary": "person"}
_GENDER_PRONOUN: dict[str, str] = {"male": "He",   "female": "She",    "non-binary": "They"}
_GENDER_POSS:    dict[str, str] = {"male": "his",  "female": "her",    "non-binary": "their"}

_OCC_PHRASE: dict[str, str] = {
    "student":               "currently a student",
    "professional/technical":"working in a professional or technical role",
    "management/executive":  "working in a management or executive position",
    "service/sales":         "working in a service or sales role",
    "skilled labor":         "working in a skilled trade or production role",
    "retired":               "retired",
}

_EDU_PHRASE: dict[str, str] = {
    "less than high school": "without a high school diploma",
    "high school diploma":   "holding a high school diploma",
    "some college":          "with some college education",
    "bachelor's degree":     "with a bachelor's degree",
    "graduate degree":       "holding a graduate degree",
}

_INCOME_PHRASE: dict[str, str] = {
    "low":          "on a modest income",
    "middle":       "with a comfortable middle-class income",
    "upper-middle": "with an upper-middle-class income",
    "high":         "with a high income",
}

# Compact behavioral phrases for each Big Five dimension × level.
# These are joined into a single sentence; no trait labels are exposed to the LLM.
_BIG_FIVE_PHRASE: dict[str, dict[str, str]] = {
    "openness": {
        "high": "curious and open to new ideas, drawn to unconventional approaches",
        "low":  "practical and conventional, preferring proven methods over novelty",
    },
    "conscientiousness": {
        "high": "organized and disciplined, planning carefully before taking action",
        "low":  "flexible and spontaneous, comfortable improvising as situations evolve",
    },
    "extraversion": {
        "high": "outgoing and vocal, energized by group discussion and social exchange",
        "low":  "reserved and reflective, preferring to listen and speak selectively",
    },
    "agreeableness": {
        "high": "cooperative and consensus-oriented, prioritizing group harmony",
        "low":  "assertive and direct, prioritizing outcomes over social harmony",
    },
    "neuroticism": {
        "high": "prone to worry under uncertainty, cautious and risk-averse",
        "low":  "calm and emotionally stable, comfortable with ambiguity and pressure",
    },
}


def _build_description(attrs: dict) -> str:
    noun       = _GENDER_NOUN[attrs["gender"]]
    age_phrase = _AGE_PHRASE_GENDERED[(attrs["age_range"], attrs["gender"])]
    occ_phrase = _OCC_PHRASE[attrs["occupation"]]
    edu_phrase = _EDU_PHRASE[attrs["education"]]
    inc_phrase = _INCOME_PHRASE[attrs["income"]]
    pronoun    = _GENDER_PRONOUN[attrs["gender"]]
    be_verb    = "are" if attrs["gender"] == "non-binary" else "is"

    # Build Big Five sentence: "{Pronoun} is/are <trait1>, <trait2>, ..., and <trait5>."
    bf = attrs["big_five"]
    trait_phrases = [_BIG_FIVE_PHRASE[dim][bf[dim]] for dim in BIG_FIVE_DIMS]
    traits_text = ", ".join(trait_phrases[:-1]) + ", and " + trait_phrases[-1]

    return (
        f"A {noun} {age_phrase}, {occ_phrase} {edu_phrase}, {inc_phrase}. "
        f"{pronoun} {be_verb} {traits_text}."
    )


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_gender(rng: random.Random) -> str:
    """Sample gender with population-approximate weights (not uniform 1:1:1)."""
    return rng.choices(
        GENDERS,
        weights=[GENDER_WEIGHTS[g] for g in GENDERS],
        k=1,
    )[0]


def _default_min_non_binary(pool_size: int) -> int:
    """Floor for non-binary count: at least 1 when the pool is not tiny."""
    if pool_size < 5:
        return 0
    return 1


def _target_non_binary_count(pool_size: int, min_non_binary: int) -> int:
    """Target non-binary slots: population rate (~1%), but never below ``min_non_binary``."""
    weight_sum = sum(GENDER_WEIGHTS[g] for g in GENDERS)
    rate = GENDER_WEIGHTS["non-binary"] / weight_sum
    expected = round(pool_size * rate)
    return max(min_non_binary, expected)


def assign_genders_for_pool(
    rng: random.Random,
    pool_size: int,
    *,
    min_non_binary: int | None = None,
) -> list[str]:
    """Assign genders for a pool using population weights, with a non-binary floor."""
    if pool_size <= 0:
        return []
    floor = _default_min_non_binary(pool_size) if min_non_binary is None else min_non_binary
    target_nb = _target_non_binary_count(pool_size, floor)

    genders = [sample_gender(rng) for _ in range(pool_size)]
    nb_count = genders.count("non-binary")
    if nb_count < target_nb:
        candidates = [i for i, g in enumerate(genders) if g != "non-binary"]
        rng.shuffle(candidates)
        for i in candidates[: target_nb - nb_count]:
            genders[i] = "non-binary"
    return genders


def resample_pool_genders(
    pool: list[dict],
    seed: int,
    *,
    min_non_binary: int | None = None,
) -> None:
    """Reassign genders in an existing persona pool in place."""
    rng = random.Random(seed)
    genders = assign_genders_for_pool(rng, len(pool), min_non_binary=min_non_binary)
    for persona, gender in zip(pool, genders):
        persona["gender"] = gender


def _sample_big_five(rng: random.Random) -> dict[str, str]:
    """Sample each Big Five dimension independently at high or low."""
    return {dim: rng.choice(BIG_FIVE_LEVELS) for dim in BIG_FIVE_DIMS}


def _sample_one(
    rng: random.Random,
    age: str | None = None,
    *,
    gender: str | None = None,
) -> dict:
    if age is None:
        age = rng.choice(AGE_RANGES)
    occ      = rng.choice(OCCUPATION_BY_AGE[age])
    edu_pool = EDUCATION_BY_AGE_OCC.get((age, occ), ["high school diploma", "some college", "bachelor's degree"])
    edu      = rng.choice(edu_pool)
    income   = rng.choice(INCOME_BY_OCC.get(occ, INCOME_LEVELS))
    gender   = gender if gender is not None else sample_gender(rng)
    big_five = _sample_big_five(rng)

    attrs: dict = {
        "age_range":  age,
        "gender":     gender,
        "education":  edu,
        "occupation": occ,
        "income":     income,
        "big_five":   big_five,
    }
    attrs["description"] = _build_description(attrs)
    return attrs


def generate_pool(
    count: int,
    seed: int,
    *,
    min_non_binary: int | None = None,
) -> list[dict]:
    """Generate `count` personas stratified over age ranges.

    Age ranges are covered in round-robin order for the first min(count, 6*k)
    personas so each age band gets equal representation.
    """
    rng = random.Random(seed)
    # Build age strata list: cycle through AGE_RANGES until we have `count` slots
    age_slots = (AGE_RANGES * ((count // len(AGE_RANGES)) + 1))[:count]
    rng.shuffle(age_slots)
    gender_slots = assign_genders_for_pool(rng, count, min_non_binary=min_non_binary)

    pool: list[dict] = []
    for age, gender in zip(age_slots, gender_slots):
        pool.append(_sample_one(rng, age=age, gender=gender))
    return pool


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=30,  help="Number of personas (default: 30)")
    parser.add_argument("--seed",  type=int, default=42,  help="Random seed (default: 42)")
    parser.add_argument(
        "--min-non-binary",
        type=int,
        default=None,
        help="Minimum non-binary personas (default: 1 when count>=5, else 0)",
    )
    parser.add_argument("--out",   type=str, default="prompts/persona_profiles.json", help="Output path")
    args = parser.parse_args()

    pool = generate_pool(args.count, args.seed, min_non_binary=args.min_non_binary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)

    print(f"✓ Generated {len(pool)} personas → {out_path}\n")

    # ── Summary stats ──
    age_counts: dict[str, int] = {}
    gender_counts: dict[str, int] = {}
    bf_counts: dict[str, dict[str, int]] = {dim: {"high": 0, "low": 0} for dim in BIG_FIVE_DIMS}
    for p in pool:
        age_counts[p["age_range"]] = age_counts.get(p["age_range"], 0) + 1
        gender_counts[p["gender"]] = gender_counts.get(p["gender"], 0) + 1
        for dim in BIG_FIVE_DIMS:
            bf_counts[dim][p["big_five"][dim]] += 1

    print("  Age range breakdown:")
    for a in AGE_RANGES:
        print(f"    {a:<8} {age_counts.get(a, 0)}")

    print("\n  Gender breakdown:")
    for g in GENDERS:
        print(f"    {g:<12} {gender_counts.get(g, 0)}")

    print("\n  Big Five high/low breakdown:")
    for dim in BIG_FIVE_DIMS:
        h = bf_counts[dim]["high"]
        l = bf_counts[dim]["low"]
        print(f"    {dim:<20} high={h}  low={l}")

    print("\n  Sample personas:")
    for i in [0, 1, 2]:
        p = pool[i]
        bf = p["big_five"]
        bf_str = "  ".join(f"{d[:1].upper()}:{v[0]}" for d, v in bf.items())
        print(f"\n  [{i}] age={p['age_range']}  gender={p['gender']}  occ={p['occupation']}")
        print(f"       Big5: {bf_str}")
        print(f"       {p['description']}")


if __name__ == "__main__":
    main()
