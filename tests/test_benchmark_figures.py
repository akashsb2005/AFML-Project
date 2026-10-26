import json

from src import benchmark, make_figures


def test_benchmark_and_figures_smoke(tmp_path):
    cfgs = []
    for att in ("standard", "linear"):
        p = tmp_path / f"{att}.yaml"
        p.write_text(f"model:\n  vocab_size: 97\n  n_layer: 1\n  n_head: 2\n  d_model: 16\n  context_length: 64\n  attention: {att}\n")
        cfgs.append(str(p))
    res = benchmark.run(cfgs, "smoke", "cpu", batch_size=2, seq_lens=[16, 32, 64], warmup=1, steps=2, repeats=2,
                        gen_prompt_len=4, gen_new_tokens=4, builder="tests.fixtures.tiny_model:build_model",
                        out_dir=str(tmp_path), overrides={"n_layer": 1})
    for v in res["variants"].values():
        assert v["params"] > 0 and v["train_tokens_per_s"] > 0 and v["gen_ms_per_token_median"] > 0
        assert set(v["forward"]) == {16, 32, 64}
    make_figures.plot_benchmark(tmp_path / "smoke_benchmark.json", str(tmp_path / "fig"))
    assert (tmp_path / "fig_forward_latency.png").exists() and (tmp_path / "fig_train_throughput.png").exists()

    log = tmp_path / "log.csv"
    log.write_text("step,tokens_seen,train_loss,val_loss,lr,elapsed_s\n1,100,5.0,,0.001,1\n2,200,4.0,4.2,0.001,2\n")
    make_figures.plot_loss([str(log)], ["run"], str(tmp_path / "loss.png"))
    assert (tmp_path / "loss.png").exists()
