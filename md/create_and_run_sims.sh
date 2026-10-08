# #!/bin/bash

# # Directory that contains the model files
# MODEL_DIR="compiled_models"

# # Check if the models directory exists
# if [ ! -d "$MODEL_DIR" ]; then
#     echo "Error: '$MODEL_DIR' directory does not exist."
#     exit 1
# fi

# # Get list of all model filenames (basename only)
# model_names=()
# while IFS= read -r -d '' file; do
#     filename=$(basename "$file")
#     model_names+=("$filename")
# done < <(find "$MODEL_DIR" -maxdepth 1 -type f -print0)

# # Loop through each relevant state point directory
# for dir in */; do
#     dir=${dir%/}  # Remove trailing slash
#     if [[ "$dir" =~ ^[0-9]+K.*-pace$ || "$dir" == CASE* ]]; then
#         echo "Processing: $dir"
        
#         # Find .in files in this state point directory
#         mapfile -t in_files < <(find "$dir" -maxdepth 1 -type f -name "*.in")

#         for model in "${model_names[@]}"; do
#             model_path="$MODEL_DIR/$model"
#             subdir="$dir/$model"

#             # Create subdirectory
#             mkdir -p "$subdir"
#             echo "  Created: $subdir"

#             for in_file in "${in_files[@]}"; do
#                 base_in_file=$(basename "$in_file")

#                 if [[ "$base_in_file" == "pace_lmp.in" ]]; then
#                     # Transform pace_lmp.in into nequip_lmp.in
#                     output_file="$subdir/nequip_lmp.in"
#                     awk -v model_file="$model" '
#                         BEGIN {
#                             print "newton          off"
#                             skip_coeff = 0
#                         }
#                         /pair_style/ {
#                             print "pair_style  nequip"
#                             next
#                         }
#                         /pair_coeff/ {
#                             if (skip_coeff == 0) {
#                                 print "pair_coeff  * * " model_file " C"
#                                 skip_coeff = 1
#                             }
#                             next
#                         }
#                         { print }
#                     ' "$in_file" > "$output_file"
#                     echo "    Transformed and copied: pace_lmp.in → nequip_lmp.in"
#                 else
#                     # Just copy the other .in files as-is
#                     cp "$in_file" "$subdir/"
#                     echo "    Copied: $base_in_file"
#                 fi
#             done

#             # Copy the model file into the subdir
#             cp "$model_path" "$subdir/"
#             echo "    Copied model file: $model"
#         done
#     fi
# done

# echo "✅ All done."

# # Path to your template submit script
# TEMPLATE_SUBMIT_SCRIPT="./lammps_submit.sh"

# # Check if it exists
# if [ ! -f "$TEMPLATE_SUBMIT_SCRIPT" ]; then
#     echo "Error: Submit script '$TEMPLATE_SUBMIT_SCRIPT' not found in current directory."
#     exit 1
# fi

# # Loop through each relevant state point directory
# for dir in */; do
#     dir=${dir%/}
#     if [[ "$dir" =~ ^[0-9]+K.*-pace$ || "$dir" == CASE* ]]; then

#         for model_subdir in "$dir"/*; do
#             if [ -d "$model_subdir" ]; then
#                 model_file=$(basename "$model_subdir")

#                 # Construct descriptive job name
#                 job_name="${dir}_${model_file}"

#                 # Destination for modified submit script
#                 submit_script_path="$model_subdir/lammps_submit.sh"

#                 # Copy and modify the job script
#                 awk -v job_name="$job_name" '
#                     /^#SBATCH -J/ {
#                         print "#SBATCH -J " job_name
#                         next
#                     }
#                     { print }
#                 ' "$TEMPLATE_SUBMIT_SCRIPT" > "$submit_script_path"

#                 echo "  ✔️  Created job script in: $model_subdir (Job Name: $job_name)"

#                 # Submit the job
#                 (cd "$model_subdir" && sbatch lammps_submit.sh)
#                 echo "  🚀  Submitted job: $job_name"
#             fi
#         done
#     fi
# done

# echo "✅ All jobs submitted."


#!/bin/bash

# Directory that contains the model files
MODEL_DIR="compiled_models"

# Check if the models directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: '$MODEL_DIR' directory does not exist."
    exit 1
fi

# Get list of all model filenames (basename only)
model_names=()
while IFS= read -r -d '' file; do
    filename=$(basename "$file")
    model_names+=("$filename")
done < <(find "$MODEL_DIR" -maxdepth 1 -type f -print0)

# Loop through each relevant state point directory
for dir in */; do
    dir=${dir%/}  # Remove trailing slash
    if [[ "$dir" =~ ^[0-9]+K.*-pace$ || "$dir" == CASE* ]]; then
        echo "Processing: $dir"

        # Find .in files in this state point directory
        mapfile -t in_files < <(find "$dir" -maxdepth 1 -type f -name "*.in")

        for model in "${model_names[@]}"; do
            model_path="$MODEL_DIR/$model"
            subdir="$dir/$model"

            # Create subdirectory
            mkdir -p "$subdir"
            echo "  Created: $subdir"

            for in_file in "${in_files[@]}"; do
                base_in_file=$(basename "$in_file")

                if [[ "$base_in_file" == "pace_lmp.in" ]]; then
                    # Transform pace_lmp.in into nequip_lmp.in
                    output_file="$subdir/nequip_lmp.in"
                    awk -v model_file="$model" '
                        BEGIN {
                            print "newton          off"
                            skip_coeff = 0
                        }
                        /pair_style/ {
                            print "pair_style  nequip"
                            next
                        }
                        /pair_coeff/ {
                            if (skip_coeff == 0) {
                                print "pair_coeff  * * " model_file " C"
                                skip_coeff = 1
                            }
                            next
                        }
                        { print }
                    ' "$in_file" > "$output_file"
                    echo "    Transformed and copied: pace_lmp.in → nequip_lmp.in"
                else
                    # Just copy the other .in files as-is
                    cp "$in_file" "$subdir/"
                    echo "    Copied: $base_in_file"
                fi
            done

            # Copy the model file into the subdir
            cp "$model_path" "$subdir/"
            echo "    Copied model file: $model"
        done
    fi
done

echo "✅ All done with directory setup."

# Path to your template submit script
TEMPLATE_SUBMIT_SCRIPT="./lammps_submit.sh"

# Check if it exists
if [ ! -f "$TEMPLATE_SUBMIT_SCRIPT" ]; then
    echo "Error: Submit script '$TEMPLATE_SUBMIT_SCRIPT' not found in current directory."
    exit 1
fi

# Loop through each relevant state point directory and create one job per statepoint
for dir in */; do
    dir=${dir%/}
    if [[ "$dir" =~ ^[0-9]+K.*-pace$ || "$dir" == CASE* ]]; then

        # Job name for this statepoint
        job_name="${dir}_all_models"

        # Path for the combined submit script
        submit_script_path="$dir/run_all_models.sh"

        # Start creating the new submit script
        # First, copy the SBATCH header lines and modify the job name
        awk -v job_name="$job_name" '
            /^#SBATCH -J/ {
                print "#SBATCH -J " job_name
                next
            }
            /^#SBATCH/ {
                print
                next
            }
            /^#!/ && NR==1 {
                print
                next
            }
            # Stop copying at the first non-header line
            /^[^#]/ || (/^#/ && !/^#SBATCH/ && NR>1) {
                exit
            }
        ' "$TEMPLATE_SUBMIT_SCRIPT" > "$submit_script_path"

        # Add the multi-model execution logic
        cat >> "$submit_script_path" << 'EOF'

# Get the statepoint directory
STATEPOINT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Loop through each model subdirectory
for model_dir in "$STATEPOINT_DIR"/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        echo "=========================================="
        echo "Running model: $model_name"
        echo "=========================================="

        cd "$model_dir"

EOF

        # Now append the actual execution commands from the template
        # (skip headers, just get the body)
        awk '
            BEGIN {
                header_done = 0
            }
            # Skip shebang and SBATCH lines
            /^#!/ && NR==1 { next }
            /^#SBATCH/ { next }
            # Skip comments at the top before real commands
            !header_done && /^#/ { next }
            !header_done && /^$/ { next }
            # Once we hit a real line, copy everything
            {
                header_done = 1
                print
            }
        ' "$TEMPLATE_SUBMIT_SCRIPT" >> "$submit_script_path"

        # Close the loop
        cat >> "$submit_script_path" << 'EOF'

        echo "Completed model: $model_name"
        echo ""
    fi
done

echo "=========================================="
echo "All models completed for this statepoint"
echo "=========================================="
EOF

        chmod +x "$submit_script_path"

        echo "  ✔️  Created combined job script: $submit_script_path (Job Name: $job_name)"

        # Submit the job from the statepoint directory
        (cd "$dir" && sbatch run_all_models.sh)
        echo "  🚀  Submitted job: $job_name"
    fi
done

echo "✅ All jobs submitted."
