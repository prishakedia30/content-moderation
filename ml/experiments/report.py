"""Collect every stage's JSON into REPORT.md (tables + figures) and a small results_bundle.zip
(no model weights) that is easy to download from Kaggle."""
import json
import zipfile
from pathlib import Path


def _load(out, name):
    p = out / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _f(x, d=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def _pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write(out):
    out = Path(out)
    L = ["# ML evaluation results", ""]
    setup = _load(out, "setup")
    if setup:
        r = setup["rows"]
        L += [f"Data: Jigsaw Toxic Comments. Train {r['train']:,}, validation {r['val']:,}, test {r['test']:,} "
              f"(stratified 80/10/10, same split as model v1). Official competition test set: {r['official_test']:,} rows.", "",
              _table(["Category", "Training comments", "Share"],
                     [[k.replace("_", " "), f"{setup['category_count_train'][k]:,}", _pct(v, 2)]
                      for k, v in setup["category_rate_train"].items()]), ""]

    comp = _load(out, "comparison")
    if comp:
        L += ["## 1. Model comparison", "",
              "Binary any-toxic metrics on the test set (precision/recall/F1 at 0.5). Latency: " + comp["latency_note"] + ".", "",
              _table(["Model", "Backend", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "Size (MB)", "p50 (ms)", "p95 (ms)"],
                     [[r["model"], r["backend"], _f(r["precision"]), _f(r["recall"]), _f(r["f1"]), _f(r["pr_auc"]),
                       _f(r["roc_auc"]), r["size_mb"], r["p50_ms"], r["p95_ms"]] for r in comp["rows"]]), ""]
        d = next((r for r in comp["rows"] if r["model"].startswith("DistilBERT multi-label") and "debias" not in r["model"]
                  and r["backend"] == "ONNX INT8"), None)
        rb = next((r for r in comp["rows"] if r["model"].startswith("RoBERTa") and r["backend"] == "PyTorch FP32"), None)
        if d and rb:
            L += [f"**Finding:** DistilBERT INT8 reaches **{100 * d['pr_auc'] / rb['pr_auc']:.1f}%** of RoBERTa-base's PR-AUC "
                  f"at **1/{rb['p50_ms'] / d['p50_ms']:.0f}** of its CPU latency ({d['p50_ms']} ms vs {rb['p50_ms']} ms) "
                  f"and {rb['size_mb'] / d['size_mb']:.0f}x less storage.", ""]
        L += ["![comparison](fig/comparison.png)", ""]

    seeds = _load(out, "seeds")
    if seeds:
        a = seeds["aggregate"]
        L += ["## 2. Multi-label model and training stability (3 seeds)", "",
              _table(["Metric", "Mean ± std (seeds 42, 43, 44)"],
                     [[k.replace("_", " "), f"{v['mean']:.4f} ± {v['std']:.4f}"] for k, v in a.items()]), "",
              "Per-category PR-AUC:", "",
              _table(["Category", "Share of training data", "PR-AUC mean ± std"],
                     [[k.replace("_", " "), _pct(setup["category_rate_train"][k], 2) if setup else "", f"{v['mean']:.3f} ± {v['std']:.3f}"]
                      for k, v in seeds["per_class_pr_auc"].items()]), "",
              "![per class](fig/per_class.png)", ""]

    sweep = _load(out, "sweep")
    if sweep:
        L += ["## 3. Hyper-parameter sweep", "", f"{sweep['subset_rows']:,} training rows; {sweep['note']}.", "",
              _table(["lr", "epoch", "val PR-AUC"], [[r["lr"], r["epochs"], _f(r["val_pr_auc"], 4)] for r in sweep["rows"]]), "",
              f"Chosen: lr {sweep['best']['lr']}, {sweep['best']['epochs']} epoch(s) (full-data runs capped at 2 epochs).", "",
              "![sweep](fig/sweep.png)", ""]

    ab = _load(out, "ablations")
    if ab:
        rows = []
        for tag, label in [("weighted_len128", "class-weighted loss, max_len 128 (reference)"),
                           ("plain_len128", "no class weights"), ("weighted_len64", "max_len 64"),
                           ("weighted_len256", "max_len 256")]:
            if tag in ab:
                r = ab[tag]
                rows.append([label, _f(r["test"]["pr_auc"]), _f(r["test"]["f1"]), _f(r["test"]["recall"]), _f(r.get("macro_pr_auc")),
                             _f(r.get("long_test_pr_auc")), _f(r.get("train_minutes"), 1)])
        L += ["## 4. Ablations", "", f"Trained on {ab['subset_rows']:,} rows. Long comments = test comments over 128 tokens "
              f"({_pct(ab.get('long_comments', {}).get('share_over_128_tokens'))} of the test set).", "",
              _table(["Variant", "PR-AUC", "F1", "Recall", "Macro PR-AUC (6 categories)", "PR-AUC on long comments", "Train min"], rows), ""]
        if "calibration" in ab:
            c = ab["calibration"]
            L += ["Calibration (main model, full data):", "",
                  _table(["", "Test ECE", "Test log-loss", "Thresholds (95/95 rule)", "Review rate", "Toxic auto-allowed"],
                         [[k, _f(v["test_ece"], 4), _f(v["test_log_loss"], 4), f"{v['thresholds'][0]:.2f} / {v['thresholds'][1]:.2f}",
                           _pct(v["test_policy"]["review_rate"]), _pct(v["test_policy"]["toxic_auto_allowed"])] for k, v in c.items()]), "",
                  "![reliability](fig/reliability.png)", ""]

    sp = _load(out, "splits")
    if sp and "official_rows" in sp:
        L += ["## 5. Split comparison", "", sp["note"], "",
              _table(["Model", "Random split PR-AUC", "Official test PR-AUC", "Random F1", "Official F1"],
                     [[k, _f(v["random_split_test"]["pr_auc"]), _f(v["official_test"]["pr_auc"]), _f(v["random_split_test"]["f1"]),
                       _f(v["official_test"]["f1"])] for k, v in sp.items() if isinstance(v, dict) and "official_test" in v]), ""]

    rb = _load(out, "robustness")
    if rb:
        L += ["## 6. Robustness to disguised abuse", "", f"{rb['definition']}; thresholds {rb['thresholds'][0]:.2f} / {rb['thresholds'][1]:.2f}. "
              f"Toxic test comments containing a known abusive word ({rb['toxic_rows'][0]['rows']:,}); every abusive word is disguised.", "",
              _table(["Attack", "Example", "Caught: model only", "Caught: + normalizer", "Blocked: model only", "Blocked: + normalizer", "Char TF-IDF recall"],
                     [[r["attack"].replace("_", " "), next((e["text"] for e in rb["examples"] if e["attack"] == r["attack"]), "").replace("|", "\\|").replace("\u200b", "&#8203;"),
                       _pct(r["caught_before"]), _pct(r["caught_after"]), _pct(r["blocked_before"]), _pct(r["blocked_after"]),
                       _pct(r.get("char_tfidf_recall@0.5"))] for r in rb["toxic_rows"]]), "",
              f"Side effect on {rb['clean']['rows']:,} clean comments: normalizer changed {_pct(rb['clean']['normalizer_changed_share'])} of them; "
              f"flag rate {_pct(rb['clean']['flag_rate_before'], 2)} → {_pct(rb['clean']['flag_rate_after'], 2)}, "
              f"block rate {_pct(rb['clean']['block_rate_before'], 2)} → {_pct(rb['clean']['block_rate_after'], 2)}.", "",
              "![robustness](fig/robustness.png)", ""]

    bias = _load(out, "bias")
    if bias:
        sb, sa = bias["summary_before"], bias["summary_after"]
        L += ["## 7. Identity-term bias", "", bias["definition"] + ".", "",
              _table(["", "Mean FPR", "Worst group FPR", "FPR gap (max − min)", "Control word FPR", "Toxic templates caught", "Test PR-AUC"],
                     [["main model", _pct(sb["mean_fpr"]), _pct(sb["max_fpr"]), _pct(sb["fpr_gap"]), _pct(sb["control_fpr"]), _pct(sb["mean_tpr"]), _f(bias["test_before"]["pr_auc"])],
                      ["after counterfactual augmentation", _pct(sa["mean_fpr"]), _pct(sa["max_fpr"]), _pct(sa["fpr_gap"]), _pct(sa["control_fpr"]), _pct(sa["mean_tpr"]), _f(bias["test_after"]["pr_auc"])]]), "",
              _table(["Group", "FPR before", "FPR after", "Mean score before", "Mean score after"],
                     [[b["group"], _pct(b["fpr"]), _pct(a_["fpr"]), _f(b["mean_score_nontoxic"]), _f(a_["mean_score_nontoxic"])]
                      for b, a_ in zip(bias["before"], bias["after"])]), "",
              f"Augmentation: {bias['augmentation']}.", "", "![bias](fig/bias.png)", ""]

    ood = _load(out, "ood")
    if ood:
        L += ["## 8. Out-of-domain", "", _table(["Model", "Dataset", "PR-AUC", "F1", "Precision", "Recall"],
                                                [[r["model"], r["dataset"], _f(r["pr_auc"]), _f(r["f1"]), _f(r["precision"]), _f(r["recall"])] for r in ood["rows"]]), "",
              "![ood](fig/ood.png)", ""]

    pol = _load(out, "policy")
    if pol:
        fx, cp = pol["fixed_95_95"], pol["capacity_policy"]
        L += ["## 9. Threshold policy", "", f"Model: {pol['model']}. Rates on the test set.", "",
              _table(["Policy", "t_low / t_high", "Allowed", "Review", "Blocked", "Block precision", "Toxic auto-allowed"],
                     [["95/95 rule (v1)", f"{fx['t_low']:.2f} / {fx['t_high']:.2f}", _pct(fx["test"]["allow_rate"]), _pct(fx["test"]["review_rate"]),
                       _pct(fx["test"]["block_rate"]), _pct(fx["test"]["block_precision"]), _pct(fx["test"]["toxic_auto_allowed"])],
                      [f"review capacity {cp['capacity']:.0%}", f"{cp['t_low']:.2f} / {cp['t_high']:.2f}", _pct(cp["test"]["allow_rate"]),
                       _pct(cp["test"]["review_rate"]), _pct(cp["test"]["block_rate"]), _pct(cp["test"]["block_precision"]),
                       _pct(cp["test"]["toxic_auto_allowed"])]]), "",
              "Cost model (cost per post = C_fn·P(false allow) + C_fp·P(false block) + C_r·P(review), C_r = 1):", "",
              _table(["C_fn", "C_fp", "t_low / t_high", "Review", "Toxic auto-allowed", "Innocent blocked", "Cost / 1000 posts"],
                     [[s["C_fn"], s["C_fp"], f"{s['t_low']:.2f} / {s['t_high']:.2f}", _pct(s["review_rate"]), _pct(s["harmful_allowed"]),
                       _pct(s["innocent_blocked"], 2), _f(s["cost_per_1000_posts"], 1)] for s in pol["cost_sweep"]]), "",
              "![policy](fig/policy.png)", ""]

    ex = _load(out, "export")
    if ex:
        L += ["## 10. Deployed model (v2)", "", f"Model: {ex['model']}. INT8 ONNX {ex['int8_mb']} MB. Decision agreement INT8 vs PyTorch "
              f"{_pct(ex['decision_agreement_int8_vs_pytorch'])} (max score difference {_f(ex['max_score_diff'], 4)}). "
              f"Production code latency p50 {ex['production_code_p50_ms']} ms, p95 {ex['production_code_p95_ms']} ms. Uploaded: {ex['uploaded']}.", "",
              "```yaml", json.dumps(ex["thresholds"], indent=2), "```", ""]
    video = out / "video" / "VIDEO.md"
    if video.exists():
        body = video.read_text().replace("\n## ", "\n### ").replace("# Video censoring benchmark", "## 11. Video censoring benchmark")
        L += [body, ""]
    (out / "REPORT.md").write_text("\n".join(L))

    with zipfile.ZipFile(out / "results_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.suffix in (".json", ".md", ".csv", ".txt"):
                z.write(f, f.name)
        for f in (out / "fig").glob("*.png"):
            z.write(f, f"fig/{f.name}")
        for f in (out / "video").glob("*"):
            z.write(f, f"video/{f.name}")
        for f in (out / "deploy").glob("thresholds.yaml"):
            z.write(f, "deploy/thresholds.yaml")
    return out / "REPORT.md"


if __name__ == "__main__":
    import sys
    print(write(sys.argv[1] if len(sys.argv) > 1 else "results"))
