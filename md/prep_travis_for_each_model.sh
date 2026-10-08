#!/bin/bash

# Full path to the lmp2xyz.py script
CONVERT_SCRIPT="/p/lustre1/laubach2/nequip_models/becky_files/lmp2xyz.py"

MISSING_FILE="missing_simulations.txt"

# Loop through each state point directory
for dir in */; do
    dir=${dir%/}
    if [[ "$dir" =~ ^[0-9]+K.*-pace$ || "$dir" == CASE* ]]; then
        echo "Entering state point directory: $dir"

        # Loop through each model subdirectory
        for subdir in "$dir"/*/; do
            subdir=${subdir%/}
            if [ -d "$subdir" ]; then
                echo "  ➤ Processing: $subdir"
                # 🔹 Skip if XYZ already exists
                if [[ -f "$subdir/traj.lammpstrj.xyz" ]]; then
                    echo "    ⏭  traj.lammpstrj.xyz already exists — skipping."
                    continue
                fi
                # Run the Python script if traj.lammpstrj exists
                if [[ -f "$subdir/traj.lammpstrj" && -f "$subdir/log.lammps" ]]; then
                    (
                        cd "$subdir" || exit
                        awk 'BEGIN{s=0;l=0}/Step/{s=1;sub($1,"# Step")}/Loop/{l=1}{if( (s==1)&&(l==0) ){print}}' log.lammps > lmp_statistics.out
                        awk 'BEGIN{s=0;ss=0;c=0}{if($1<5000){$6=$6/10000; s+=$6; ss+=$6*$6; c++}}END{a=s/c; print(a,sqrt( ss/c - a*a))}' lmp_statistics.out > pressure.5ps.dat
                        echo "    🧪 Running lmp2xyz..."
                        python3 "$CONVERT_SCRIPT" METAL traj.lammpstrj log.lammps

                        # Rename the output file
                        if [ -f "traj.lammpstrj.xyzf" ]; then
                            mv traj.lammpstrj.xyzf traj.lammpstrj.xyz
                            echo "    ✅ Converted and renamed output."
                        else
                            echo "    ⚠️  Conversion script did not produce traj.lammpstrj.xyzf"
                        fi
                    )
                else
                    echo "    ⚠️  Missing traj.lammpstrj or log.lammps in $subdir"
                    echo "$subdir" >> "$MISSING_FILE"
                fi
            fi
        done
    fi
done

echo "✅ All conversions complete."
