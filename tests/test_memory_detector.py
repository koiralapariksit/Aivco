import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.memory_detector import MemoryDetector
from memory.memory_controller import MemoryController


def test_positive_detection_simple():
    d = MemoryDetector()
    cases = [
        ("My name is Alex.", "identity"),
        ("I am learning Python.", "education"),
        ("I am building AIVCO.", "project"),
        ("My goal is to become a data scientist.", "goal"),
        ("I prefer dark mode.", "preference"),
        ("I am studying physics.", "education"),
    ]
    for text, cat in cases:
        cands = d.detect(text)
        assert len(cands) >= 1
        assert cands[0].category == cat
        assert cands[0].key != ""
        assert len(cands[0].value) > 0


def test_negative_detection():
    d = MemoryDetector()
    negatives = [
        "What is Python?",
        "How do I learn Python?",
        "Python is popular.",
        "Tell me about AIVCO.",
        "This code is broken.",
    ]
    for text in negatives:
        assert d.detect(text) == []


def test_duplicate_protection(tmp_path):
    memfile = tmp_path / "profile.json"
    memory = MemoryController(memfile)
    # ensure clean start
    memory.forget_all()

    # simulate existing memory
    memory.remember("learning_python", "User is learning Python", category="education", importance="medium")

    d = MemoryDetector()
    cands = d.detect("I am learning Python.")
    # detector will still return a candidate, but higher-level logic should
    # avoid prompting if duplicate exists. Here assert candidate key matches.
    assert any(c.key.startswith("learning_") for c in cands)


def test_key_validation():
    d = MemoryDetector()
    cands = d.detect("I am learning C++ and Rust.")
    for c in cands:
        # keys should be alphanumeric with underscores/hyphens only
        assert c.key.replace("_", "").replace("-", "").isalnum()


def test_goal_detection_one_candidate():
    d = MemoryDetector()
    cands = d.detect("I want to become an AI engineer")
    assert len(cands) == 1
    cand = cands[0]
    assert cand.category == "goal"
    assert cand.importance == "high"
    assert cand.value == "Ai engineer" or cand.value == "AI engineer"
    assert cand.key == "goal_ai_engineer"


def test_goal_detection_variants_one_candidate():
    d = MemoryDetector()
    variants = [
        "I want to become an AI engineer",
        "I want to be an AI engineer",
        "My goal is to become an AI engineer",
        "I plan to become an AI engineer",
    ]
    for text in variants:
        cands = d.detect(text)
        assert len(cands) == 1, f"Unexpected count for {text}: {len(cands)}"
        cand = cands[0]
        assert cand.category == "goal"
        assert cand.importance == "high"
        assert cand.key == "goal_ai_engineer"


def test_education_detection_one_candidate():
    d = MemoryDetector()
    cands = d.detect("I am learning machine learning")
    assert len(cands) == 1
    cand = cands[0]
    assert cand.category == "education"
    assert cand.importance == "medium"
    assert cand.value == "Machine learning"
    assert cand.key == "learning_machine_learning"


def test_preference_detection_one_candidate():
    d = MemoryDetector()
    cands = d.detect("I prefer practical projects")
    assert len(cands) == 1
    cand = cands[0]
    assert cand.category == "preference"
    assert cand.importance == "medium"
    assert cand.value == "Practical projects"
    assert cand.key == "preference_practical_projects"

    cands = d.detect("I like Python")
    assert len(cands) == 1
    cands = d.detect("I don't like unnecessary complexity")
    assert len(cands) == 1


def test_project_detection_one_candidate():
    d = MemoryDetector()
    cands = d.detect("I am building AIVCO")
    assert len(cands) == 1
    cand = cands[0]
    assert cand.category == "project"
    assert cand.importance == "high"


def test_conservative_negative_sentences():
    d = MemoryDetector()
    negatives = [
        "The weather is nice today.",
        "I went to the store.",
        "This movie was good.",
        "I am tired today.",
        "I ate rice.",
        "The computer is slow.",
    ]
    for text in negatives:
        assert d.detect(text) == []
