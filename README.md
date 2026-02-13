# Tile Barcode Pipeline for OPS (Optical Pooled Screening)

A Python pipeline for processing OPS imaging data with nuclei segmentation, image alignment, and two-color barcode calling optimized for NovaSeq sequencing chemistry.

## Overview

This pipeline processes multi-round fluorescence microscopy images to:
- Stack and align imaging rounds
- Segment nuclei using Cellpose
- Detect fluorescent spots
- Call barcodes using two-color intensity-based base calling with median correction

**Important**: This pipeline is specifically designed for **NovaSeq sequencing chemistry** with two-color (FITC and Cy3) fluorescence detection.

## Installation

```bash
pip install numpy pandas tifffile cellpose torch scipy scikit-image
```

Additionally, you need the `ops` package for image processing:
```bash
pip install ops-tools  # or install from source
```

## Usage

### 1. How to Make an Image Stack

The `make_image_stack()` function reads OME-TIFF files from multiple sequencing rounds and stacks them into a 4D array.

```python
from tile_barcode_pipeline import make_image_stack

# Define parameters
dir_to_read_file = "/path/to/your/data"
crop_y_start = 0
crop_x_start = 0
crop_size = 2048

# Create image stack
image_stack, mean_vals, background_vals = make_image_stack(
    dir_to_read_file=dir_to_read_file,
    crop_y_start=crop_y_start,
    crop_x_start=crop_x_start,
    crop_size=crop_size,
    rounds=range(2, 7),  # Process rounds 2-6
    channel_to_use=[
        'DAPI__FINAL_F',
        'FITC_NOVAseq-AC_FINAL_AFR_F',
        'Cy3_NOVAseq-TC_FINAL_AFR_F'
    ],
    csv_filename="ome_files_metadata.csv",
    region='R000'
)

# image_stack shape: (num_rounds, num_channels, crop_size, crop_size)
print(f"Stack shape: {image_stack.shape}")
```

**Key Parameters:**
- `dir_to_read_file`: Directory containing OME-TIFF files and metadata CSV
- `crop_y_start`, `crop_x_start`: Starting coordinates for cropping
- `crop_size`: Size of the square crop region
- `rounds`: Which sequencing rounds to process
- `channel_to_use`: List of channel names (must match metadata CSV)
- `csv_filename`: Metadata CSV with columns: round, region, channel_marker, full_path

**Metadata CSV Format:**
```csv
round,region,channel_marker,full_path
2,R000,DAPI__FINAL_F,/path/to/round2_dapi.ome.tif
2,R000,FITC_NOVAseq-AC_FINAL_AFR_F,/path/to/round2_fitc.ome.tif
2,R000,Cy3_NOVAseq-TC_FINAL_AFR_F,/path/to/round2_cy3.ome.tif
...
```

### 2. Full Tile Processing Pipeline

Process a complete tile through segmentation, alignment, and barcode calling:

```python
from tile_barcode_pipeline import process_tile_image

# Process tile
results = process_tile_image(
    tile_path="path/to/tile.ome.tif",
    threshold_peaks=150,
    min_threshold_intensity=400
)

# Access results
barcode_df = results['barcode_dataframe_cells']
spot_df = results['barcode_dataframe_spots']

# Save barcodes
barcode_df.to_csv('cell_barcodes.csv', index=False)
spot_df.to_csv('spot_barcodes.csv', index=False)
```

**Output DataFrames:**

*Cell-level barcodes* (`barcode_dataframe_cells`):
- `CellLabel`: Unique cell identifier
- `Barcode`: Best barcode sequence for the cell
- `Centroid_X`, `Centroid_Y`: Cell centroid coordinates
- `Quality`: Average quality score (0-1)
- `NumSpots`: Number of spots detected in the cell

*Spot-level barcodes* (`barcode_dataframe_spots`):
- `CellLabel`: Cell containing the spot
- `Barcode`: Barcode sequence for this spot
- `Spot_X`, `Spot_Y`: Spot coordinates
- `Quality`: Quality score for this spot
- `SpotIndex`: Unique spot identifier

### 3. How Base Calling Works (NovaSeq Data)

This pipeline uses **two-color base calling** optimized for NovaSeq sequencing chemistry.

#### NovaSeq Two-Color Chemistry

NovaSeq uses **two fluorophores** (FITC and Cy3) to encode four bases:

| Base | FITC (Green) | Cy3 (Red) | Description |
|------|--------------|-----------|-------------|
| **A** | High | Low | FITC only |
| **T** | Low | High | Cy3 only |
| **C** | High | High | Both channels |
| **G** | Low | Low | No signal |

#### Base Calling Algorithm

The `compute_barcodes_two_color()` function implements:

**Step 1: Median Correction**
- Corrects for channel crosstalk and intensity differences
- Uses spectral unmixing with median-based transformation
- Each cycle is corrected independently

**Step 2: Base Calling Logic**

For each spot in each cycle:

```python
I_FITC, I_Cy3 = corrected_intensities

if max(I_FITC, I_Cy3) < min_threshold_intensity:
    base = 'G'  # No signal
    
elif both channels >= threshold and ratio ~ 1:
    base = 'C'  # Both channels high
    
elif I_Cy3 >> I_FITC:
    base = 'T'  # Red dominant
    
elif I_FITC >> I_Cy3:
    base = 'A'  # Green dominant
    
else:
    base = best_guess()  # Ambiguous signal
```

**Quality Score Calculation:**
- Quality reflects signal-to-noise and channel separation
- Range: 0.0 (low quality) to 1.0 (high quality)
- Based on intensity ratios and threshold margins

**Step 3: Best Barcode Selection**
- Each cell may have multiple spots
- The spot with the highest average quality score is selected as the cell's barcode

#### Key Parameters

```python
compute_barcodes_two_color(
    intensity_values,      # Shape: (N_spots, num_cycles, 2)
    labels_mask,           # Cell labels for each spot
    positions,             # Spot (y, x) coordinates
    min_threshold_intensity=100  # Adjust based on your data
)
```

**Tuning `min_threshold_intensity`:**
- **Too low**: More false positives, noisy base calls
- **Too high**: Missing real signals, incomplete barcodes
- **Recommended**: 100-500 for NovaSeq data (depends on imaging conditions)
- Check your data's signal distribution to optimize

## NovaSeq Data Requirements

✅ **Required:**
- Two-channel fluorescence imaging (FITC + Cy3)
- OME-TIFF format images
- Metadata CSV linking rounds/channels to files
- DAPI channel for nuclei segmentation and alignment

✅ **Channel Naming Convention:**
- DAPI: `DAPI__FINAL_F`
- FITC (A/C detection): `FITC_NOVAseq-AC_FINAL_AFR_F`
- Cy3 (T/C detection): `Cy3_NOVAseq-TC_FINAL_AFR_F`

⚠️ **Not compatible with:**
- Four-color imaging systems
- Single-color imaging
- Illumina MiSeq/HiSeq chemistry (different fluorophore scheme)

## Pipeline Steps

1. **Load Image Stack**: Read multi-round OME-TIFF files
2. **Nuclei Segmentation**: Cellpose deep learning segmentation
3. **Alignment**: DAPI-based alignment across cycles
4. **Image Transformation**: LoG filtering and max filtering
5. **Peak Detection**: Identify fluorescent spots
6. **Cell Segmentation**: Watershed-based cell boundaries
7. **Spot Extraction**: Extract intensities at peak locations
8. **Barcode Calling**: Two-color median-corrected base calling

## Example Workflow

```python
from tile_barcode_pipeline import make_image_stack, process_tile_image

# Method 1: Process pre-stacked tile
results = process_tile_image(
    tile_path="tile_R000.ome.tif",
    threshold_peaks=150,
    min_threshold_intensity=400
)

# Method 2: Create stack then process
image_stack, _, _ = make_image_stack(
    dir_to_read_file="./imaging_data",
    crop_y_start=0,
    crop_x_start=0,
    crop_size=2048,
    rounds=range(2, 7)
)

# Save the stack for later use
import tifffile
tifffile.imwrite("tile_stack.ome.tif", image_stack)

# Then process it
results = process_tile_image("tile_stack.ome.tif")

# Extract barcodes
barcodes = results['barcode_dataframe_cells']
print(f"Found {len(barcodes)} cells with barcodes")
print(f"Mean quality: {barcodes['Quality'].mean():.3f}")
```

## Output Files

The pipeline can generate:
- `*_segmented.tif`: Nuclei segmentation masks (saved in `segmented/` subdirectory)
- `cell_barcodes.csv`: Best barcode per cell
- `spot_barcodes.csv`: All individual spot barcodes

## Troubleshooting

**Low barcode quality:**
- Adjust `min_threshold_intensity` parameter
- Check imaging quality and signal-to-noise ratio
- Verify channel alignment across cycles

**Few cells detected:**
- Check nuclei segmentation (may need to adjust `min_size` parameter)
- Verify DAPI channel quality

**Missing spots:**
- Lower `threshold_peaks` parameter
- Check that cycles are properly aligned

## Citation

If you use this pipeline, please cite the OPS framework and Cellpose:

- OPS: [Add OPS publication]
- Cellpose: Stringer, C., Wang, T., Michaelos, M. et al. Nat Methods 18, 100–106 (2021).

## License

[Add your license information]
