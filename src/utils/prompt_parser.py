"""
prompt_parser.py
Extract expected objects and attributes from text prompts.

Used for Signal 3: Object Gap Detection
"""

import re
from dataclasses import dataclass


@dataclass
class ExpectedObject:
    """An object expected to appear in the image based on prompt."""
    noun: str           # e.g., "car", "umbrella"
    attributes: list    # e.g., ["red", "sports"]
    full_phrase: str    # e.g., "red sports car"


# Common colors for attribute extraction
COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "brown", "gray", "grey", "silver", "gold", "golden", "bronze",
    "beige", "tan", "navy", "teal", "cyan", "magenta", "maroon", "olive"
}

# Size/material attributes
SIZE_ATTRS = {"large", "small", "tiny", "huge", "big", "little", "tall", "short"}
MATERIAL_ATTRS = {"wooden", "metal", "glass", "plastic", "leather", "fabric", "stone", "brick"}
STYLE_ATTRS = {"modern", "vintage", "old", "new", "shiny", "matte", "glossy", "rustic"}

ALL_ATTRIBUTES = COLORS | SIZE_ATTRS | MATERIAL_ATTRS | STYLE_ATTRS

# Background/location words to ignore (not objects)
BACKGROUND_WORDS = {
    "sky", "ground", "floor", "wall", "ceiling", "background", "grass", "sand",
    "water", "air", "space", "room", "area", "scene", "land", "surface", "sunset",
    "beach", "park", "street", "garden", "kitchen", "bathroom", "bedroom", "gym",
    "cafe", "library", "farm", "studio", "forest", "mountain", "ocean", "sea",
    "sunrise", "dawn", "dusk", "night", "day", "morning", "evening", "afternoon"
}

# Common articles and prepositions
ARTICLES = {"a", "an", "the"}
PREPOSITIONS = {"on", "in", "at", "by", "with", "near", "beside", "behind", "under", "over", "above", "below"}


def parse_prompt(prompt: str) -> list[ExpectedObject]:
    """
    Parse a text prompt to extract expected objects with their attributes.

    Example:
        Input: "a red sports car, blue umbrella, and white surfboard on the beach"
        Output: [
            ExpectedObject(noun="car", attributes=["red", "sports"], full_phrase="red sports car"),
            ExpectedObject(noun="umbrella", attributes=["blue"], full_phrase="blue umbrella"),
            ExpectedObject(noun="surfboard", attributes=["white"], full_phrase="white surfboard")
        ]
    """
    # Normalize prompt
    text = prompt.lower().strip()

    # Split by common separators
    # "a red car, blue umbrella, and white surfboard" → ["a red car", "blue umbrella", "white surfboard"]
    segments = re.split(r',\s*(?:and\s+)?|\s+and\s+', text)

    objects = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Tokenize
        words = segment.split()

        # First, strip location phrases (everything after preposition)
        # "green parrot in a sunny garden" -> "green parrot"
        clean_words = []
        for word in words:
            if word in PREPOSITIONS:
                break  # Stop at first preposition
            clean_words.append(word)

        if not clean_words:
            continue

        words = clean_words

        # Find the main noun (last non-background, non-attribute word)
        noun = None
        noun_idx = -1

        for i in range(len(words) - 1, -1, -1):
            word = words[i]
            # Skip background words
            if word in BACKGROUND_WORDS:
                continue
            # Skip articles
            if word in ARTICLES:
                continue
            # Skip attributes (they modify nouns, not nouns themselves)
            if word in ALL_ATTRIBUTES:
                continue
            # Found a potential noun
            if len(word) > 2:  # Skip very short words
                noun = word
                noun_idx = i
                break

        if noun is None:
            continue

        # Extract attributes (words before the noun that are in our attribute sets)
        attributes = []
        full_phrase_words = []

        for i in range(noun_idx):
            word = words[i]
            if word in ARTICLES:
                continue
            if word in ALL_ATTRIBUTES:
                attributes.append(word)
                full_phrase_words.append(word)
            elif word not in PREPOSITIONS and len(word) > 2:
                # Could be a compound noun modifier (e.g., "sports" in "sports car")
                full_phrase_words.append(word)

        full_phrase_words.append(noun)
        full_phrase = " ".join(full_phrase_words)

        objects.append(ExpectedObject(
            noun=noun,
            attributes=attributes,
            full_phrase=full_phrase
        ))

    return objects


def get_object_nouns(prompt: str) -> list[str]:
    """Simple extraction of just noun words (for CLIP filtering)."""
    objects = parse_prompt(prompt)
    return [obj.noun for obj in objects]


def get_color_object_pairs(prompt: str) -> list[tuple[str, str]]:
    """Extract (color, object) pairs from prompt."""
    objects = parse_prompt(prompt)
    pairs = []
    for obj in objects:
        colors = [attr for attr in obj.attributes if attr in COLORS]
        if colors:
            pairs.append((colors[0], obj.noun))  # Take first color if multiple
    return pairs


def compare_objects(expected: list[ExpectedObject], detected: list[dict]) -> list[dict]:
    """
    Compare expected objects (from prompt) with detected objects (from image).

    Args:
        expected: List of ExpectedObject from parse_prompt()
        detected: List of dicts with {"label": str, "attributes": list, "confidence": float}

    Returns:
        List of gaps/mismatches:
        [
            {"object": "car", "issue": "wrong_color", "expected": "red", "found": "blue"},
            {"object": "dog", "issue": "missing", "expected": "golden retriever", "found": None},
        ]
    """
    gaps = []
    detected_labels = {d["label"].lower() for d in detected}

    for exp_obj in expected:
        # Check if object exists
        matching_detected = [d for d in detected if d["label"].lower() == exp_obj.noun]

        if not matching_detected:
            # Object missing entirely
            gaps.append({
                "object": exp_obj.noun,
                "issue": "missing",
                "expected": exp_obj.full_phrase,
                "found": None
            })
            continue

        # Object exists - check attributes
        det = matching_detected[0]
        det_attrs = set(det.get("attributes", []))

        for attr in exp_obj.attributes:
            if attr in COLORS:
                # Check color mismatch
                det_colors = det_attrs & COLORS
                if det_colors and attr not in det_colors:
                    gaps.append({
                        "object": exp_obj.noun,
                        "issue": "wrong_color",
                        "expected": attr,
                        "found": list(det_colors)[0] if det_colors else "unknown"
                    })

    return gaps


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    test_prompts = [
        "a red sports car, blue umbrella, white surfboard, yellow beach ball, and green palm tree on a sandy beach at sunset",
        "a golden retriever, black cat, and white rabbit in a sunny garden",
        "two red apples and three green pears on a wooden table",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        objects = parse_prompt(prompt)
        for obj in objects:
            print(f"  {obj.noun}: {obj.attributes} → '{obj.full_phrase}'")

        colors = get_color_object_pairs(prompt)
        print(f"  Color pairs: {colors}")
