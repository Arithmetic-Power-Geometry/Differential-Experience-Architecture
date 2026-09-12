from pathlib import Path
import csv
import json
import sys
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dea.experiments import coupled_architecture, separable_architecture, ablated_architecture, evaluate_architecture


def jsonable(result):
    out = {}
    for k, v in result.items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def main():
    results_dir = ROOT / "results"; figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True); figures_dir.mkdir(exist_ok=True)
    c = coupled_architecture()
    results = [evaluate_architecture(a) for a in [separable_architecture(), c, ablated_architecture(c)]]
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump([jsonable(r) for r in results], f, indent=2)
    fields = ["architecture","hessian_fro_error","symmetry_error","min_eigenvalue","rank","interaction_fraction","determinant_ratio","rmse_full","rmse_diag","full_vs_diag_improvement","noise_true_kl","noise_mae"]
    with open(results_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in results: w.writerow({k:r[k] for k in fields})
    for r in results:
        np.savetxt(results_dir / f"hessian_{r['architecture']}.csv", r["hessian"], delimiter=",", fmt="%.12g")
    names = [r["architecture"] for r in results]
    plt.figure(figsize=(6,4)); plt.bar(names,[r["interaction_fraction"] for r in results]); plt.ylabel("Interaction fraction"); plt.tight_layout(); plt.savefig(figures_dir / "interaction_fraction.png", dpi=180); plt.close()
    x=np.arange(len(results)); width=.35
    plt.figure(figsize=(7,4)); plt.bar(x-width/2,[r["rmse_diag"] for r in results],width,label="Diagonal"); plt.bar(x+width/2,[r["rmse_full"] for r in results],width,label="Full"); plt.xticks(x,names); plt.ylabel("RMSE"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / "prediction_rmse.png",dpi=180); plt.close()
    rc = next(r for r in results if r["architecture"]=="coupled")
    plt.figure(figsize=(6,4)); plt.loglog(rc["step_sizes"],rc["step_errors"],marker="o"); plt.xlabel("Finite-difference step h"); plt.ylabel("Frobenius error"); plt.tight_layout(); plt.savefig(figures_dir / "step_convergence.png",dpi=180); plt.close()
    with open(results_dir / "REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Differential Experience Architecture computational validation\n\n")
        for r in results:
            f.write(f"## {r['architecture'].title()} architecture\n\n")
            f.write(f"- Hessian/Fisher Frobenius error: `{r['hessian_fro_error']:.3e}`\n- Symmetry error: `{r['symmetry_error']:.3e}`\n- Minimum eigenvalue: `{r['min_eigenvalue']:.3e}`\n- Rank: `{r['rank']}`\n- Interaction fraction: `{r['interaction_fraction']:.6f}`\n- Determinant ratio: `{r['determinant_ratio']:.6f}`\n- Full-model RMSE: `{r['rmse_full']:.3e}`\n- Diagonal-model RMSE: `{r['rmse_diag']:.3e}`\n- Relative full-vs-diagonal improvement: `{100*r['full_vs_diag_improvement']:.2f}%`\n- Sampling-noise KL MAE (n=20,000): `{r['noise_mae']:.3e}`\n\n")
        f.write("## Scope\n\nThese results validate the mathematical/synthetic test harness only. They are not human-subject VR validation.\n")
    rs=next(r for r in results if r["architecture"]=="separable"); ra=next(r for r in results if r["architecture"]=="ablated")
    assert rs["hessian_fro_error"] < 1e-5 and rc["hessian_fro_error"] < 1e-5
    assert rs["interaction_fraction"] < 1e-4 and rc["interaction_fraction"] > .20 and ra["interaction_fraction"] < 1e-4
    assert rc["rmse_full"] < rc["rmse_diag"]*.35 and rs["rmse_full"] <= rs["rmse_diag"]*1.05
    assert rc["min_eigenvalue"] > -1e-6
    print((results_dir / "REPORT.md").read_text())

if __name__ == "__main__": main()
