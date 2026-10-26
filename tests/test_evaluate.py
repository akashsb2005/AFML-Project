"""Tests for Member 3's metric code: known-answer checks, not just "it runs"."""
import json
import math

import numpy as np
import pytest
import torch

from src import evaluate as ev
from src import model_api
from tests.fixtures.tiny_model import ByteTokenizer, ConstantLogits, build_model


@pytest.mark.parametrize("stride", [None, 8, 5, 1])
def test_uniform_model_perplexity_equals_vocab(stride):
    V = 50
    tokens = np.random.default_rng(0).integers(0, V, 301).astype(np.uint16)
    res = ev.stream_nll(ConstantLogits(V), tokens, context_length=16, stride=stride, batch_size=4)
    assert res["tokens_scored"] == 300                       # every target scored exactly once
    assert res["perplexity"] == pytest.approx(V, rel=1e-6)


def test_windows_cover_each_target_once_with_enough_history():
    T, S, n = 16, 5, 103
    seen = []
    for b, e, first in ev._windows(n, T, S):
        assert e - b <= T and first >= b
        seen += list(range(first, e))
        if b > 0:
            assert first - b >= T - S                        # minimum history for scored tokens
    assert seen == list(range(n))


def test_stream_nll_matches_direct_computation():
    torch.manual_seed(0)
    model = build_model(dict(vocab_size=40, n_layer=1, n_head=2, d_model=16, context_length=32)).eval()
    tokens = np.random.default_rng(1).integers(0, 40, 65).astype(np.int64)
    res = ev.stream_nll(model, tokens, context_length=32, stride=32, batch_size=8)
    with torch.no_grad():
        x = torch.tensor(tokens)
        direct = sum(float(torch.nn.functional.cross_entropy(model(x[i:i + 32][None])[0], x[i + 1:i + 33], reduction="sum"))
                     for i in (0, 32))
    assert res["mean_nll"] == pytest.approx(direct / 64, rel=1e-5)


def test_response_nll_counts_only_response_tokens():
    tok = ByteTokenizer()
    ex = {"instruction": "Say hi.", "context": "", "response": "hi"}
    ids, labels, _ = ev.build_sft_example(ex, tok)
    n_resp = len(tok.encode("hi")) + 1                        # response (no leading space, as Member 2) + EOS
    assert sum(l != -100 for l in labels) == n_resp
    res = ev.response_nll(ConstantLogits(tok.vocab_size), [ex], tok, context_length=64)
    assert res["response_tokens"] == n_resp
    assert res["response_perplexity"] == pytest.approx(tok.vocab_size, rel=1e-6)


def test_text_stats():
    assert ev.repetition_rate("a b c d a b c d a b c d".split(), 4) == pytest.approx(5 / 9)
    assert ev.repetition_rate("one two three four five".split(), 4) == 0.0
    assert ev.prompt_copy_rate(ev.words("explain gravity to me"), ev.words("Explain gravity to me please")) == 1.0
    assert ev.rouge_l_f1(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert ev.rouge_l_f1(["x"], ["a", "b"]) == 0.0
    assert ev.text_stats("", "q")["empty"] == 1


def test_generation_and_rubric_roundtrip(tmp_path):
    tok = ByteTokenizer()
    model = build_model(dict(vocab_size=tok.vocab_size, n_layer=1, n_head=2, d_model=16, context_length=64)).eval()
    prompts = [{"id": i, "instruction": f"Name a colour {i}.", "context": "", "response": "Blue.", "category": "brainstorming"}
               for i in range(3)]
    rows = ev.generate_for_prompts(model, tok, prompts, 64, max_new_tokens=8)
    assert len(rows) == 3 and all(r["new_tokens"] <= 8 for r in rows)
    again = ev.generate_for_prompts(model, tok, prompts, 64, max_new_tokens=8)
    assert [r["generation"] for r in rows] == [r["generation"] for r in again]   # greedy is deterministic
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    ev.write_jsonl(a, rows)
    ev.write_jsonl(b, rows)
    sheet, key = tmp_path / "sheet.csv", tmp_path / "key.json"
    assert ev.make_rubric_sheet({"pre": str(a), "sft": str(b)}, str(sheet), str(key)) == 6
    import csv
    header = next(csv.reader(open(sheet)))
    assert "model" not in header                                # blind: model identity only in the key file
    filled = list(csv.DictReader(open(sheet)))
    for r in filled:
        for c in ev.RUBRIC_CRITERIA:
            r[c] = "2" if json.loads(key.read_text())[r["item"]]["model"] == "sft" else "1"
    with open(sheet, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filled[0].keys()))
        w.writeheader()
        w.writerows(filled)
    s = ev.score_rubric(str(sheet), str(key))["per_model"]
    assert s["sft"]["mean_total_out_of_8"] == 8 and s["pre"]["mean_total_out_of_8"] == 4


def test_checkpoint_roundtrip_and_cli(tmp_path, monkeypatch):
    tok = ByteTokenizer()
    cfg = dict(vocab_size=tok.vocab_size, n_layer=1, n_head=2, d_model=16, context_length=32, attention="standard")
    model = build_model(cfg)
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg, "tokenizer_path": "unused", "step": 0}, ckpt)
    loaded, meta = model_api.load_checkpoint(str(ckpt), torch.device("cpu"), "tests.fixtures.tiny_model:build_model")
    x = torch.randint(0, tok.vocab_size, (1, 10))
    with torch.no_grad():
        assert torch.allclose(loaded(x), model.eval()(x))
    toks = tmp_path / "val.npy"
    np.save(toks, np.random.default_rng(0).integers(0, 256, 200).astype(np.uint16))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_api, "load_tokenizer", lambda path: model_api.wrap_tokenizer(tok))
    ev.main(["perplexity", "--ckpt", str(ckpt), "--builder", "tests.fixtures.tiny_model:build_model",
             "--device", "cpu", "--label", "t", "--tokens", str(toks)])
    out = json.loads((tmp_path / "results/metrics/t_perplexity.json").read_text())
    assert out["tokens_scored"] == 199 and math.isfinite(out["perplexity"])
