
#!/bin/bash

start=1264145647
end=1264188847

#Get frame files
if [[ -d ./frames-without-injs ]]
then
    echo -e "\\n\\n>> [`date`] Pre-existing strain files without injections found. Skipping fetching frame files."
else 
    mkdir frames-without-injs
    python get.py --gps-start-time $start --gps-end-time $end
fi

#Make frames with injections

## 1.Generate injections.hdf
if [[ -f injections.hdf ]]
then
    echo -e "\\n\\n>> [`date`] Pre-existing injections found"
else
    echo -e "\\n\\n>> [`date`] Generating injections"
    ### adjust time-step to control the number of injections
    pycbc_create_injections --config-file injection_config.ini \
			--gps-start-time $start \
			--gps-end-time $end \
			--time-step 200 \
			--time-window 20 \
			--output-file injections.hdf
fi

if [[ -d ./frames-with-injs ]]
then
    echo -e "\\n\\n>> [`date`] Pre-existing strain files with injections found. Skipping adding injections to the frame files."
else
    mkdir frames-with-injs 

    ## 2.Add injections to the strain files
    # Directories
    input_dir="./frames-without-injs"
    output_dir="./frames-with-injs"

    echo -e "\\n\\n>> [`date`] Adding injections to:"
    # Loop over detectors (H, L, V)
    for det in H L V; do
	# Loop over strain files for each detector
	echo "Detector-"$det
	for strain_file in $input_dir/${det}-*.gwf; do
	    # Extract filename without path
	    filename=$(basename "$strain_file")
	    echo $filename
	    # Check if there are any matching files
	    if [ ! -f "$strain_file" ]; then
		echo "No files found for detector: $det"
		continue
	    fi

	    # Example filename: H-H1_GWOSC_O3b_4KHZ_R1-1264168960-4096.gwf
	    # Extract start time and duration from the filename
	    start=$(echo "$filename" | cut -d '-' -f 3)

	    # Adjust the start and end time slightly if necessary (you can experiment with this)
	    adjusted_start=$((start + 4))  # Subtracting 4 seconds as an example
	    end=$((adjusted_start + duration))
	    duration=$(echo "$filename" | cut -d '-' -f 4 | cut -d '.' -f 1)
	    end=$((adjusted_start + duration - 10))

	    # Extract detector name (H1, L1, V1) from filename
	    detector=$(echo "$filename" | cut -d '-' -f 1)

	    # Set the channel name based on the detector
	    channel="${detector}1:GWOSC-4KHZ_R1_STRAIN"

	    # Output file path
	    output_file="${output_dir}/${filename}"

	    # Ensure output directory exists
	    mkdir -p "$(dirname "$output_file")"

	    # Run pycbc_condition_strain command
	    pycbc_condition_strain --frame-files $strain_file \
				   --low-frequency-cutoff 10.0 \
				   --gps-start-time $adjusted_start \
				   --gps-end-time $end \
				   --pad-data 2 \
				   --sample-rate 4096 \
				   --injection-file injections.hdf \
				   --channel-name $channel \
				   --output-strain-file $output_file
	done
    done
fi
