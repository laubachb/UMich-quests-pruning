import os
import pandas as pd
import json

RDF_FILENAME = "rdf_C_#2_C_[Cr_Co].csv"
OUTPUT_JSON = "rdf_data_summary.json"

root_dir = os.getcwd()
# Dictionary structure: { state_name: { model_name: { "r": [], "g_r": [] } } }
json_output_data = {}

for state_dir in sorted(os.listdir(root_dir)):
    if not (state_dir.endswith("-pace") or state_dir.startswith("CASE")):
        continue

    state_path = os.path.join(root_dir, state_dir)
    if not os.path.isdir(state_path):
        continue

    # Initialize sub-dictionary for this statepoint
    json_output_data[state_dir] = {}

    for model_dir in sorted(os.listdir(state_path)):
        model_path = os.path.join(state_path, model_dir)
        rdf_path = os.path.join(model_path, RDF_FILENAME)

        if os.path.isfile(rdf_path):
            try:
                df = pd.read_csv(rdf_path, comment="#", sep=";", engine="python")
                df.columns = [col.strip() for col in df.columns]

                if df.shape[1] < 2:
                    continue

                # Data processing: Convert pm to Å and extract lists
                r_values = (df.iloc[:, 0].astype(float) / 100.0).tolist()
                g_r_values = df.iloc[:, 1].astype(float).tolist()

                # Add to dictionary
                json_output_data[state_dir][model_dir] = {
                    "r": r_values,
                    "g_r": g_r_values
                }

            except Exception as e:
                print(f"  ❌ Error reading {rdf_path}: {e}")

# Check if we actually gathered data before writing
if json_output_data:
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        # indent=4 makes the file human-readable
        json.dump(json_output_data, f, indent=4)
    print(f"✅ Success: Data written to {OUTPUT_JSON}")
else:
    print("❌ No data found to save.")