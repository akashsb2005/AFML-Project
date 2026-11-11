from src.select_eval_prompts import fingerprint, select, story_openings


def _rows(n, cat):
    return [{"id": f"{cat}{i}", "instruction": f"{cat} question {i}", "context": "", "category": cat} for i in range(n)]


def test_balanced_reproducible_and_leak_check():
    test = _rows(10, "qa") + _rows(3, "summarization") + _rows(10, "brainstorming")
    a, leaked = select(test, 9, seed=1)
    b, _ = select(test, 9, seed=1)
    assert [r["id"] for r in a] == [r["id"] for r in b] and not leaked
    cats = [r["category"] for r in a]
    assert cats.count("qa") == cats.count("summarization") == cats.count("brainstorming") == 3
    _, leaked = select(test, 9, seed=1, exclude_rows=[dict(test[0], instruction=" QA  question 0 ")])
    assert len(leaked) == 1                                    # whitespace/case-insensitive match
    assert fingerprint({"instruction": "A  b"}) == fingerprint({"instruction": "a b"})


def test_story_openings_are_distinct_and_cut_short():
    rows = [{"text": "Once upon a time, there was a little girl."},
            {"text": "Once upon a time, there was a big dog."},
            {"text": "Too short."},
            {"text": "Lily and Ben are friends. They like to play."}]
    out = story_openings(rows, n=5)
    assert [r["id"] for r in out] == ["val_story_00000", "val_story_00003"]
    assert out[1]["prompt"] == "Lily and Ben are friends. They like to"
