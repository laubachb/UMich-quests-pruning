#!/bin/bash


# Replace with actual path to travis binary if not set in $PATH
TRAVIS_BIN="/p/lustre1/laubach2/nequip_models/becky_files/travis" # fallback to 'travis' if $travis not set

# Loop over all state point directories
for dir in */; do
    dir=${dir%/}
    if [[ "$dir" =~ ^[0-9]+K.*-pace$ || "$dir" == CASE* ]]; then
        echo "📁 Entering state point directory: $dir"

        # Loop over each model subdirectory
        for subdir in "$dir"/*/; do
            subdir=${subdir%/}
            if [ -d "$subdir" ]; then
                echo "  ➤ Processing: $subdir"

                if [[ -f "$subdir/traj.lammpstrj" && -f "$subdir/log.lammps" ]]; then
                    (
                        cd "$subdir" || exit
                        # --- Skip entire directory if rdf*.csv already exists ---
                        if ls rdf*.csv >/dev/null 2>&1; then
                            echo "    ⏭️  rdf*.csv found, skipping TRAVIS in this directory"
                            exit 0
                        fi
                        
                        # === 2. Run TRAVIS for msd, power, rdf ===
                        for j in msd power rdf; do
                            input_file="${j}.in"
                            echo "    🔄 Running TRAVIS for: $j"

                            if [ -f "$input_file" ]; then
                                $TRAVIS_BIN -p traj.lammpstrj.xyz -i "$input_file"
                                if [ -f travis.log ]; then
                                    mv travis.log "${j}.log"
                                    echo "    ✅ Saved ${j}.log"
                                else
                                    echo "    ⚠️  travis.log not found after $j run"
                                fi
                            else
                                echo "    ⚠️  Input file $input_file not found"
                            fi
                        done
                    )
                else
                    echo "    ⚠️  Missing traj.lammpstrj or log.lammps in $subdir"
                fi
            fi
        done
    fi
done

echo "✅ All analysis complete."
