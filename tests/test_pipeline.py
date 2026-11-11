import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dataset
import preprocess
import sft
import tokenizer
from src import evaluate, model_api, training
from src.model import SmallLanguageModel
from src.efficient_model import EfficientSmallLanguageModel


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PreprocessTests(unittest.TestCase):
    def test_normalization_dedup_conflicts_reproducibility_and_jsonl(self):
        self.assertEqual(preprocess.clean_text("  A\n   B "), "A B")
        rows = [
            {"instruction": "  What   IS this? ", "response": "zebra", "category": "z"},
            {"instruction": "what is THIS?", "response": "apple", "category": "a"},
            {"instruction": "Another prompt", "response": "ok", "category": "x"},
            {"instruction": " ", "response": "invalid"},
        ]
        first, stats = preprocess.prepare_dolly_rows(rows, seed=9)
        again, _ = preprocess.prepare_dolly_rows(rows, seed=9)
        self.assertEqual(first, again)
        self.assertEqual(stats["original_records"], 4)
        self.assertEqual(stats["invalid_records_removed"], 1)
        self.assertEqual(stats["duplicate_prompts_removed"], 1)
        self.assertEqual(stats["records_retained"], 2)
        retained = [r for split in first.values() for r in split]
        self.assertEqual(len(retained), 2)
        self.assertIn("apple", [r["response"] for r in retained])
        self.assertTrue(preprocess.validate_dolly_splits(first))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.jsonl"
            preprocess.save_jsonl(retained, path)
            loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(loaded, retained)


class TokenizerTests(unittest.TestCase):
    def test_existing_tokenizer_artifact(self):
        tok = tokenizer.load_and_validate_tokenizer()
        self.assertEqual(tok.get_vocab_size(), 8000)
        self.assertEqual([tok.token_to_id(t) for t in tokenizer.SPECIAL_TOKENS], [0, 1, 2, 3, 4])
        enc = tok.encode("hello world")
        self.assertEqual((enc.ids[0], enc.ids[-1]), (2, 3))
        self.assertEqual(tok.token_to_id("[PAD]"), 0)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tiny_outputs_and_padding(self):
        write_rows(self.data_dir / "tiny_stories_train.jsonl", [{"text": "A short story."}])
        with patch.object(dataset, "DATA_DIR", self.data_dir):
            ds = dataset.TinyStoriesDataset("train", block_size=16)
        item = ds[0]
        self.assertEqual(set(item), {"input_ids", "labels", "attention_mask"})
        self.assertTrue(all(v.dtype == torch.long and v.shape == (16,) for v in item.values()))
        self.assertEqual(item["input_ids"][0].item(), 2)
        self.assertEqual(item["labels"][0].item(), int(item["input_ids"][1]))
        self.assertIn(-100, item["labels"].tolist())
        valid = item["attention_mask"].bool()
        self.assertTrue(torch.all(item["labels"][~valid] == -100))

    def test_tiny_stories_loads_all_splits(self):
        for split in ("train", "validation", "test"):
            write_rows(self.data_dir / f"tiny_stories_{split}.jsonl", [{"text": f"A {split} story."}])
        with patch.object(dataset, "DATA_DIR", self.data_dir):
            for split in ("train", "validation", "test"):
                ds = dataset.TinyStoriesDataset(split, block_size=12)
                self.assertGreater(len(ds), 0)
                self.assertEqual(ds[0]["input_ids"].shape, (12,))

    def test_dolly_response_mask_eos_truncation_skip_and_stats(self):
        prompt = "Instruction: What does this detailed test ask?\nResponse:"
        long_instruction = " ".join(["preserve" + str(i) for i in range(30)])
        rows = [
            {"prompt": prompt, "response": "A useful answer."},
            {"prompt": "Instruction: " + long_instruction + "\nResponse:", "response": "answer"},
            {"prompt": prompt, "response": "long " * 400},
            {"prompt": "Instruction: " + "x " * 400 + "\nResponse:", "response": "answer"},
            {"prompt": "Instruction: empty\nResponse:", "response": ""},
        ]
        write_rows(self.data_dir / "dolly_train.jsonl", rows)
        with patch.object(dataset, "DATA_DIR", self.data_dir):
            ds = dataset.DollySFTDataset("train")
        self.assertEqual(ds.stats["original_examples"], 5)
        self.assertEqual(ds.stats["prompt_too_long_skipped"], 1)
        self.assertEqual(ds.stats["empty_examples_skipped"], 1)
        self.assertEqual(ds.stats["responses_truncated"], 1)
        self.assertEqual(len(ds), 3)
        for item in ds:
            self.assertEqual(set(item), {"input_ids", "labels", "attention_mask"})
            self.assertEqual(item["input_ids"].shape, (256,))
            self.assertEqual(item["input_ids"].dtype, torch.long)
            self.assertTrue(torch.all(item["attention_mask"][:-1] >= item["attention_mask"][1:]))
            self.assertTrue(torch.all(item["labels"][item["attention_mask"] == 0] == -100))
            supervised = item["labels"][item["labels"] != -100]
            self.assertGreaterEqual(supervised.numel(), 2)
            self.assertEqual(supervised[-1].item(), 3)
            pad_start = item["attention_mask"].sum().item()
            self.assertTrue(torch.all(item["labels"][pad_start:] == -100))
        # Long instruction is wholly kept: locate tokenized leading word.
        full_prompt = "Instruction: " + long_instruction + "\nResponse:"
        long_ids = tokenizer.load_and_validate_tokenizer().encode(full_prompt, add_special_tokens=False).ids
        retained_input = ds.examples[1]["input_ids"]
        self.assertEqual(retained_input[:len(long_ids)], long_ids)

    def test_dolly_loads_all_splits_and_dataloaders_initialize(self):
        rows = [{"prompt": "Instruction: Say hello.\nResponse:", "response": "Hello!"}]
        for split in ("train", "validation", "test"):
            write_rows(self.data_dir / f"dolly_{split}.jsonl", rows)
        with patch.object(dataset, "DATA_DIR", self.data_dir):
            for split in ("train", "validation", "test"):
                ds = dataset.DollySFTDataset(split, max_length=32)
                self.assertEqual(len(ds), 1)
            train, validation = sft.create_dataloaders(sft.SFTConfig(max_length=32, batch_size=1))
            self.assertEqual(next(iter(train))["input_ids"].shape, (1, 32))
            self.assertEqual(next(iter(validation))["labels"].shape, (1, 32))


class SFTTests(unittest.TestCase):
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 8000
            self.block_size = 512
            self.embed = torch.nn.Embedding(8000, 8)
            self.head = torch.nn.Linear(8, 8000)

        def forward(self, input_ids, attention_mask=None):
            return self.head(self.embed(input_ids))

    def test_output_loss_epoch_and_checkpoint_roundtrip_cpu(self):
        logits = torch.randn(2, 5, 10, requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 3, -100], [-100, 5, 2, 1, 3]])
        loss = sft.causal_lm_loss(logits, labels)
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 10), labels[:, 1:].reshape(-1), ignore_index=-100
        )
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.equal(sft.get_logits((logits, None)), logits))
        model = self.Dummy()
        self.assertTrue(sft.validate_model_compatibility(model, 256))
        with self.assertRaises(ValueError):
            sft.validate_model_compatibility(model, 1024)
        batch = {"input_ids": torch.tensor([[2, 5, 3, 0]]), "labels": torch.tensor([[-100, 5, 3, -100]]), "attention_mask": torch.tensor([[1, 1, 1, 0]])}
        loader = [batch]
        optimizer = torch.optim.AdamW(model.parameters())
        train_loss = sft.run_epoch(model, loader, torch.device("cpu"), optimizer)
        val_loss = sft.run_epoch(model, loader, torch.device("cpu"))
        self.assertGreater(train_loss, 0)
        self.assertGreater(val_loss, 0)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pt"
            sft.save_checkpoint(model, optimizer, 1, val_loss, sft.SFTConfig(), path)
            clone = self.Dummy()
            sft.load_pretrained_checkpoint(clone, path, torch.device("cpu"))
            for a, b in zip(model.parameters(), clone.parameters()):
                self.assertTrue(torch.equal(a, b))
            with self.assertRaises(FileNotFoundError):
                sft.load_pretrained_checkpoint(clone, Path(td) / "missing.pt", torch.device("cpu"))

    def test_project_model_logits_attention_and_evaluation_checkpoint_format(self):
        model = SmallLanguageModel(vocab_size=8000, embedding_dim=16, num_heads=4,
                                   num_layers=1, max_seq_len=32, dropout=0.0)
        ids = torch.randint(0, 8000, (2, 8))
        mask = torch.ones_like(ids)
        logits = model(ids, attention_mask=mask)
        self.assertEqual(logits.shape, (2, 8, 8000))
        result = model(ids, targets=ids, attention_mask=mask)
        self.assertEqual(result[0].shape, logits.shape)
        self.assertTrue(torch.isfinite(result[1]))
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "member2_sft.pt"
            sft.save_checkpoint(model, optimizer, 1, 1.0, sft.SFTConfig(max_length=32), path)
            loaded, payload = model_api.load_checkpoint(str(path), torch.device("cpu"))
            self.assertEqual(payload["stage"], "sft")
            self.assertTrue(torch.equal(loaded(ids), logits))

    def test_pretraining_checkpoint_loads_through_sft_and_evaluation_adapter(self):
        ids = torch.randint(0, 8000, (2, 8))
        batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
        for cls, attention in ((SmallLanguageModel, "standard"), (EfficientSmallLanguageModel, "linear")):
            model = cls(vocab_size=8000, embedding_dim=16, num_heads=4,
                        num_layers=1, max_seq_len=32, dropout=0.0)
            val_loss, _ = training.evaluate(model, [batch], torch.device("cpu"))
            self.assertGreater(val_loss, 0)
            optimizer = torch.optim.AdamW(model.parameters())
            config = {"model": {"vocab_size": 8000, "n_layer": 1, "n_head": 4, "d_model": 16, "context_length": 32,
                                "dropout": 0.0, "attention": attention},
                      "train": {}, "data": {"tokenizer": "artifacts/tokenizer/tokenizer.json"}}
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "pretrain.pt"
                training._save_checkpoint(path, step=1, tokens_seen=16, model=model, optimizer=optimizer,
                                          config=config, train_loss=1.0, validation_loss=val_loss,
                                          best_validation_loss=val_loss, elapsed_s=0.0,
                                          data_rng_state=torch.get_rng_state())
                payload = sft.read_checkpoint(path, torch.device("cpu"))
                self.assertEqual(payload["config"]["attention"], attention)
                loaded, _ = model_api.load_checkpoint(str(path), torch.device("cpu"))
                self.assertTrue(torch.allclose(loaded(ids), model(ids), atol=1e-6))

    def test_fine_tune_logs_validation_and_keeps_best_checkpoint(self):
        ids = torch.randint(5, 8000, (2, 8))
        labels = ids.clone()
        labels[:, :3] = -100
        batch = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
        model = SmallLanguageModel(vocab_size=8000, embedding_dim=16, num_heads=4,
                                   num_layers=1, max_seq_len=32, dropout=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        config = sft.SFTConfig(max_length=8, epochs=2, log_every=0, eval_interval=2, learning_rate=1e-2)
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "log.csv"
            sft.fine_tune(model, [batch] * 3, [batch], optimizer, torch.device("cpu"), config, Path(td), log_path)
            rows = list(csv.DictReader(log_path.open()))
            self.assertEqual([int(r["step"]) for r in rows], [2, 3, 4, 6])
            self.assertEqual(int(rows[-1]["tokens_seen"]), 6 * 16)
            self.assertTrue((Path(td) / "best.pt").exists() and (Path(td) / "latest.pt").exists())
            self.assertTrue(model.training)
            config = sft.SFTConfig(max_length=8, epochs=5, log_every=0, eval_interval=0, max_steps=4)
            sft.fine_tune(model, [batch] * 3, [batch], optimizer, torch.device("cpu"), config, Path(td), log_path)
            self.assertEqual([int(r["step"]) for r in csv.DictReader(log_path.open())], [3, 4])

    def test_evaluation_builder_preserves_prompt_and_matches_response_truncation(self):
        tok = model_api.load_tokenizer(str(ROOT / "artifacts" / "tokenizer" / "tokenizer.json"))
        row = {"prompt": "Instruction: Keep the complete instruction intact.\nResponse:",
               "response": "An answer with enough words to exceed the very small context window."}
        ids, labels, _ = evaluate.build_sft_example(row, tok, max_length=30)
        prompt_ids = tok.encode(row["prompt"])
        budget = 30 - len(prompt_ids) - 1
        self.assertEqual(ids[:len(prompt_ids)], prompt_ids)
        self.assertEqual(len(ids), 30)
        self.assertEqual(labels[:len(prompt_ids)], [-100] * len(prompt_ids))
        self.assertEqual(labels[-1], tok.eos_id)
        self.assertEqual(len([v for v in labels if v != -100]), budget + 1)


if __name__ == "__main__":
    unittest.main()
