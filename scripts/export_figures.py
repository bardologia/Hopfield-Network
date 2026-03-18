"""Extract PNG figures from the executed Hopfield demo notebook.

Reads the .ipynb JSON, finds every code cell with an image/png output,
and saves each image to  figures/<nn>_<slug>.png  with a descriptive name.
"""

import json, base64, pathlib, re, sys

NOTEBOOK = pathlib.Path(r"c:\Users\vice_vi\Desktop\Boltzman Machines\notebooks\hopfield_demo.ipynb")
OUT_DIR  = pathlib.Path(r"c:\Users\vice_vi\Desktop\Boltzman Machines\figures")

# Map code-cell execution order → descriptive filename slug
SLUG_MAP = {
    3:  "01_stored_patterns",
    4:  "02_recall_noise_levels",
    5:  "03_energy_trajectory",
    6:  "04_weight_matrix",
    7:  "05_overlap_dynamics",
    8:  "06_async_vs_sync_dynamics",
    9:  "07_async_vs_sync_comparison",
    10: "08_storage_capacity",
    11: "09_hebbian_vs_storkey",
    12: "10_basin_of_attraction",
    13: "11_spurious_states",
    14: "12_letter_patterns_stored",
    15: "13_letter_recall",
    16: "14_convergence_heatmap",
}

def extract():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_idx = 0            # 1-based index among code cells
    saved = 0

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        code_idx += 1
        exec_count = cell.get("execution_count")

        # Determine slug
        slug = SLUG_MAP.get(exec_count) or SLUG_MAP.get(code_idx)
        if slug is None:
            slug = f"cell_{code_idx:02d}"

        img_idx = 0
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            png_b64 = data.get("image/png")
            if png_b64 is None:
                continue
            # Handle list-of-lines vs single string
            if isinstance(png_b64, list):
                png_b64 = "".join(png_b64)
            img_bytes = base64.b64decode(png_b64)
            suffix = "" if img_idx == 0 else f"_{img_idx}"
            fname = OUT_DIR / f"{slug}{suffix}.png"
            fname.write_bytes(img_bytes)
            print(f"  ✓ {fname.name}  ({len(img_bytes)//1024} KB)")
            saved += 1
            img_idx += 1

    print(f"\nDone — saved {saved} figure(s) to {OUT_DIR}")

if __name__ == "__main__":
    extract()
