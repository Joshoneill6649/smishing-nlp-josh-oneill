#!/usr/bin/env python3

#  rule based user profile risk estimator for smishing messages.
# 
# used only when predict_message.py is ran
# 
# Inputs from the caller:
#   - smish_prob: float in [0, 1], 
#   - emo_probs: dict[label -> prob], 
# 
# Output:
#   - assess each of the profiles 
#      
# Used as an interpretable user centric layer on top of
# the smishing classifier and emotion model.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

# profile definitions
# These are hand crafted, interpretable personas.

@dataclass
class UserProfile:
    # Short identifier for the profile
    id: str
    # Human readable name for the profile
    name: str
    # Description of the profile behaviour and context
    description: str
    # Emotions that, if strong in a message, increase this profile risk
    sensitive_to: List[str]
    # Emotions that, if strong, decrease this profile risk
    protective: List[str]
    # Baseline tendency to click risky content (between 0 and 1), purely heuristic
    base_click_risk: float


# List of all predefined profiles used by the risk scoring logic (3)
PROFILES: List[UserProfile] = [
    UserProfile(
        id="busy_pro",
        name="Busy Professional",
        description=(
            "High message volume and time pressure; more likely to react "
            "quickly to urgent or threatening messages without verifying."
        ),
        sensitive_to=["fear", "anxiety", "urgency", "embarrassment", "anger"],
        protective=["calm", "neutral"],
        base_click_risk=0.35,
    ),
    UserProfile(
        id="reward_seeker",
        name="Reward-Seeking User",
        description=(
            "Likes rewards, refunds, prizes, discounts; more responsive "
            "to positive and gain-framed emotional cues."
        ),
        sensitive_to=["joy", "excitement", "desire", "admiration"],
        protective=["fear", "disgust"],
        base_click_risk=0.30,
    ),
    UserProfile(
        id="cautious_user",
        name="Cautious User",
        description=(
            "Security-aware and skeptical; less influenced by emotional tone "
            "and more likely to double-check suspicious content."
        ),
        sensitive_to=[],
        protective=["fear", "neutral", "annoyance"],
        base_click_risk=0.10,
    ),
]

# helper

# Helper to sum probabilities for emotion labels that contain any of the given keys
def _get_prob(emo_probs: Dict[str, float], keys: List[str]) -> float:
    # If there are no emotion probabilities or no keys, return zero
    if not emo_probs or not keys:
        return 0.0

    total = 0.0
    # Normalise keys to lower case and cast to float for safety
    lower = {k.lower(): float(v) for k, v in emo_probs.items()}
    for key in keys:
        k = key.lower()
        for lbl, p in lower.items():
            # Substring based match, for example "fear" matches "fear" and "fearful"
            if k in lbl:
                total += p
    return total


# Compute a simple risk score for one profile based on smish probability and emotions
def score_profile_risk(smish_prob: float, emo_probs: Dict[str, float], profile: UserProfile) -> Tuple[float, str]:
    # Clamp model probability into [0, 1]
    sp = max(0.0, min(1.0, float(smish_prob)))
    # Start from the profile baseline risk
    score = profile.base_click_risk
    # Collect text reasons that explain the score
    reasons = []

    # 1) Effect of classifier smishing probability
    if sp >= 0.9:
        score += 0.40
        reasons.append("very high smishing probability")
    elif sp >= 0.7:
        score += 0.25
        reasons.append("high smishing probability")
    elif sp <= 0.3:
        score -= 0.20
        reasons.append("low smishing probability")

    # 2) Emotional triggers for this profile
    # Sum probabilities for emotions this profile is sensitive to
    sens = _get_prob(emo_probs, profile.sensitive_to)
    # Sum probabilities for emotions that are protective or skeptical
    prot = _get_prob(emo_probs, profile.protective)

    # If there is a lot of triggering emotion, increase risk
    if sens > 0.30:
        score += 0.20
        reasons.append("strong presence of emotions this profile is sensitive to")
    elif sens > 0.15:
        score += 0.10
        reasons.append("some triggering emotional cues")

    # If protective emotions are strong, reduce risk
    if prot > 0.30:
        score -= 0.15
        reasons.append("protective or skeptical emotional tone")

    # Clamp final score into [0, 1]
    score = max(0.0, min(1.0, score))

    # If no specific reason was added, give a default explanation
    if not reasons:
        reasons.append("no strong emotional triggers for this profile")

    # Join reasons into a single explanation string
    explanation = "; ".join(reasons)
    return score, explanation


# Map numeric score into LOW, MEDIUM or HIGH for readability
def label_risk(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


# Compute the risk view for all profiles and return them as a sorted list of dicts
def assess_all_profiles(smish_prob: float, emo_probs: Dict[str, float]):
    # all profile results here
    results = []
    for prof in PROFILES:
        # Compute score and explanation for this profile
        score, why = score_profile_risk(smish_prob, emo_probs, prof)
        results.append({
            "id": prof.id,
            "name": prof.name,
            "description": prof.description,
            "risk_score": round(score, 3),
            "risk_label": label_risk(score),
            "why": why,
        })

    # Sort profiles by descending risk score so highest risk appears first
    results.sort(key=lambda d: d["risk_score"], reverse=True)
    return results
