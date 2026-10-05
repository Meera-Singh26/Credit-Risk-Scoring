"""Quick EDA report -> reports/figures/eda_*.png and reports/eda_summary.txt"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from .data import load_raw


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    df["default"] = 1 - df[C.SOURCE_TARGET]
    lines = [f"rows={len(df)} cols={df.shape[1]} default_rate={df['default'].mean():.3f}",
             f"nulls={int(df.isna().sum().sum())}", ""]
    for c in ["status", "savings", "credit_history", "purpose"]:
        rate = df.groupby(c)["default"].agg(["mean", "size"]).sort_values("mean", ascending=False)
        lines += [f"Default rate by {c}:", rate.round(3).to_string(), ""]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.barh(rate.index, rate["mean"], color="#e11d48")
        ax.axvline(df["default"].mean(), color="k", ls="--"); ax.set_title(f"Default rate by {c}")
        fig.tight_layout(); fig.savefig(C.FIG_DIR / f"eda_{c}.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, c in zip(axes, ["duration", "amount", "age"], strict=True):
        df.boxplot(column=c, by="default", ax=ax); ax.set_title(c); ax.set_xlabel("default")
    fig.suptitle(""); fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_numeric.png", dpi=130); plt.close(fig)
    (C.REPORT_DIR / "eda_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines[:3]))


if __name__ == "__main__":
    main()
