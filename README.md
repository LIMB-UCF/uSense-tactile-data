# uSense Tactile Dataset (Sample)


## Overview

This repository provides:

	•	A representative subset of the tactile texture dataset used in the uSense study.
	•	Code for data reformatting and segmentation.

The complete dataset is available upon request.

## ⚙️ Preprocessing (Run before the uSense pipeline)

Run the following scripts **in order** before running the end-to-end uSense pipeline:

```bash
!python src/csv_2_npz.py      # Step 1: Convert CSV → NPZ
!python src/segmentation.py   # Step 2: Segment data → segmented_data.npz
```

## Code Availability

The stochastic FFT implementation is available [here](https://github.com/UnaryLab/napl/blob/fft/src/napl/algorithm/fft/fft.py).

The end-to-end uSense pipeline is available [here](https://github.com/UnaryLab/napl/tree/fft/zoo/uSense).
