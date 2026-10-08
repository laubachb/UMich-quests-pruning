import os
import pandas as pd
import matplotlib.pyplot as plt
import math

RDF_FILENAME = "rdf_C_#2_C_[Cr_Co].csv"

root_dir = os.getcwd()
all_state_data = []

# Loop over state point directories
for state_dir in sorted(os.listdir(root_dir)):
    if not (state_dir.endswith("-pace") or state_dir.startswith("CASE")):
        continue

    state_path = os.path.join(root_dir, state_dir)
    if not os.path.isdir(state_path):
        continue

    state_data = []

    for model_dir in sorted(os.listdir(state_path)):
        model_path = os.path.join(state_path, model_dir)
        rdf_path = os.path.join(model_path, RDF_FILENAME)

        if os.path.isfile(rdf_path):
            try:
                df = pd.read_csv(
                    rdf_path,
                    comment="#",
                    sep=";",
                    engine="python"
                )
                df.columns = [col.strip() for col in df.columns]

                # Check if at least 2 columns exist
                if df.shape[1] < 2:
                    print(f"  ⚠️  Too few columns in {rdf_path}")
                    continue

                # Convert pm → Å
                df[df.columns[0]] = df[df.columns[0]].astype(float) / 100.0

                state_data.append((model_dir, df))

            except Exception as e:
                print(f"  ❌ Error reading {rdf_path}: {e}")

    if state_data:
        all_state_data.append((state_dir, state_data))

# Exit if no data
if not all_state_data:
    print("❌ No RDF data found.")
    exit()

# Determine subplot layout
num_plots = len(all_state_data)
cols = 2
rows = math.ceil(num_plots / cols)

fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)

for idx, (state_name, rdf_list) in enumerate(all_state_data):
    row, col = divmod(idx, cols)
    ax = axs[row][col]

    for model_name, df in rdf_list:
        x = df.iloc[:, 0]  # r in Å
        y = df.iloc[:, 1]  # g(r)
        ax.plot(x, y, label=model_name)

    ax.set_title(state_name, fontsize=10)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.legend(fontsize=6)

# Hide unused axes
for idx in range(len(all_state_data), rows * cols):
    row, col = divmod(idx, cols)
    fig.delaxes(axs[row][col])

plt.tight_layout()
plt.savefig("all_rdf_plots.png", dpi=300)
print("✅ Saved: all_rdf_plots.png")
