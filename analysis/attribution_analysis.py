import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon


def holm_bonferroni(pvals, alpha=0.05):
    """Return Holm-adjusted p-values and reject decisions."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)

    reject = adjusted < alpha
    return adjusted, reject


rows = []

for fp in sorted(Path("results/interpretability").glob("*_s*_ig.json")):
    if fp.name == "smoke.json":
        continue

    with open(fp) as f:
        data = json.load(f)

    for ex in data["examples"]:
        if "id" not in ex:
            raise KeyError(
                f"{fp} has an example without field 'id'. "
                "Paired Wilcoxon needs the same example_id across seeds/models."
            )

        rows.append({
            "example_id": ex["id"],
            "tag": data["tag"],
            "seed": data["seed"],
            "fraction_on_identity": ex["fraction_on_identity"],
            "true_label": ex["true_label"],
            "pred_label": ex["pred_label"],
        })

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No *_s*_ig.json files found under results/interpretability/")

print(f"Total: {len(df)} rows across {df['tag'].nunique()} models")
print("\nExamples per tag/seed:")
print(df.groupby(["tag", "seed"]).size())

required_tags = {"modelb", "modelc", "cciv2"}
missing_tags = required_tags - set(df["tag"].unique())
if missing_tags:
    raise RuntimeError(f"Missing expected tags: {sorted(missing_tags)}")

# Distribution summary. We keep mean/std for internal reference, but report
# median [IQR] in the paper because the distribution is right-skewed.
def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


summary = (
    df.groupby("tag")["fraction_on_identity"]
    .agg(median="median", q1=q1, q3=q3, mean="mean", std="std", count="count")
)

summary["iqr"] = summary["q3"] - summary["q1"]
summary = summary.round(6)

print("\n=== Fraction on identity tokens: median [IQR] ===")
print(summary[["median", "q1", "q3", "iqr", "mean", "std", "count"]])

# Correct paired test:
# average repeated seed measurements per (example_id, model),
# then compare the same examples across models.
pivot = (
    df.groupby(["example_id", "tag"])["fraction_on_identity"]
    .mean()
    .unstack("tag")
)

print("\nPaired pivot shape:")
print(pivot.shape)
print("\nMissing values per tag in pivot:")
print(pivot.isna().sum())

pairs = [
    ("modelb", "modelc"),
    ("modelb", "cciv2"),
    ("modelc", "cciv2"),
]

test_results = []

for a, b in pairs:
    diff = (pivot[a] - pivot[b]).dropna()
    n_zero = int((diff == 0).sum())
    n_eff = len(diff) - n_zero

    w, p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    d_z = diff.mean() / diff.std(ddof=1)

    test_results.append({
        "pair": f"{a} vs {b}",
        "n": len(diff),
        "n_effective": n_eff,
        "n_zero": n_zero,
        "W": w,
        "p_raw": p,
        "p_raw_scientific": f"{p:.6e}",
        "cohens_d_z": d_z,
    })

p_holm, reject = holm_bonferroni([r["p_raw"] for r in test_results])

for r, ph, rej in zip(test_results, p_holm, reject):
    r["p_holm"] = ph
    r["p_holm_scientific"] = f"{ph:.6e}"
    r["reject_holm"] = bool(rej)

tests_df = pd.DataFrame(test_results)

print("\n=== Paired Wilcoxon tests, Holm-corrected ===")
print(tests_df)

# Figure: violin for distribution shape + CDF for stochastic ordering.
Path("paper/figures").mkdir(parents=True, exist_ok=True)
Path("results/interpretability").mkdir(parents=True, exist_ok=True)

order = ["modelb", "modelc", "cciv2"]
labels = {
    "modelb": "Model B: vanilla DeBERTa-v3",
    "modelc": "Model C: focal + identity mask",
    "cciv2": "CCI v2: fixed T, JS",
}

plot_df = df.copy()
plot_df["model"] = plot_df["tag"].map(labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

sns.violinplot(
    data=plot_df,
    x="tag",
    y="fraction_on_identity",
    order=order,
    inner="quartile",
    ax=ax1,
)

ax1.set_xlabel("")
ax1.set_ylabel("Fraction on identity tokens")
ax1.set_xticklabels([labels[t] for t in order], rotation=20, ha="right")

for tag in order:
    x = np.sort(df[df.tag == tag]["fraction_on_identity"].values)
    y = np.linspace(0, 1, len(x))
    ax2.plot(x, y, label=labels[tag])

ax2.set_xlabel("Fraction on identity tokens")
ax2.set_ylabel("CDF")
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig("paper/figures/attribution_shift.pdf", bbox_inches="tight")
fig.savefig("paper/figures/attribution_shift.png", dpi=200, bbox_inches="tight")

summary.to_csv("results/interpretability/summary_stats.csv")
tests_df.to_csv("results/interpretability/wilcoxon_tests.csv", index=False)

with open("results/interpretability/pvals.txt", "w") as f:
    for _, row in tests_df.iterrows():
        f.write(
            f"{row['pair']}: "
            f"W={row['W']:.0f}, "
            f"n={int(row['n'])}, "
            f"n_effective={int(row['n_effective'])}, "
            f"n_zero={int(row['n_zero'])}, "
            f"p_raw={row['p_raw']:.6e}, "
            f"p_holm={row['p_holm']:.6e}, "
            f"d_z={row['cohens_d_z']:.4f}, "
            f"reject_holm={row['reject_holm']}\n"
        )

print("\nSaved:")
print("- paper/figures/attribution_shift.pdf")
print("- paper/figures/attribution_shift.png")
print("- results/interpretability/summary_stats.csv")
print("- results/interpretability/wilcoxon_tests.csv")
print("- results/interpretability/pvals.txt")
