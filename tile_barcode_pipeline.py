"""
Tile Processing Pipeline for OPS (Optical Pooled Screening) data.

Contains functions for:
- Image stacking from OME-TIFF files
- Nuclei segmentation (Cellpose)
- Tile processing pipeline (alignment, filtering, peak detection, barcode calling)
- Two-color spot intensity barcode computation with median correction

Author:
    Md Abdul Kader Sagar
    High-Throughput Imaging Facility (HiTIF)
    National Cancer Institute (NCI)
    Email: sagarm2@nih.gov

Dependencies:
    pip install numpy pandas tifffile cellpose torch scipy scikit-image

Usage:
    from tile_barcode_pipeline import process_tile_image, make_image_stack
"""

import os
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from datetime import datetime

# Image processing
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.ndimage import sum as ndi_sum, maximum as ndi_max
from skimage.segmentation import expand_labels
from skimage.filters import threshold_otsu
from skimage.measure import regionprops

# Deep learning segmentation
import torch
from cellpose import models

# OPS pipeline
from ops.process import Align
from ops.firesnake import Snake


# =============================================================================
# Utility Functions
# =============================================================================

def ts_print(msg):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


# =============================================================================
# Segmentation Functions
# =============================================================================

def cellposeSegNucG(img, min_size=15):
    """
    Segment nuclei using Cellpose with GPU support if available.
    
    Parameters
    ----------
    img : ndarray
        2D grayscale image (e.g., DAPI channel).
    min_size : int, optional
        Minimum cell size in pixels (default: 15).
    
    Returns
    -------
    ndarray
        Label image where each nucleus has a unique integer label.
    """
    use_gpu = torch.cuda.is_available()
    model = models.Cellpose(gpu=use_gpu, model_type="nuclei")
    res, _, _, _ = model.eval(
        img,
        channels=[0, 0],
        diameter=None,
        min_size=min_size,
    )
    return res


def get_or_create_segmentation(tile_path, image_data, segment_nuclei_func=None):
    """
    Get segmentation masks from a pre-segmented file or create new segmentation.
    
    Looks for a pre-existing segmented file in a 'segmented/' subdirectory.
    If not found, runs the segmentation function and saves the result.
    
    Parameters
    ----------
    tile_path : str or Path
        Path to the tile image file.
    image_data : ndarray
        2D image to segment (e.g., DAPI channel).
    segment_nuclei_func : callable, optional
        Segmentation function. Defaults to cellposeSegNucG.
    
    Returns
    -------
    ndarray
        Label image of segmented nuclei.
    """
    if segment_nuclei_func is None:
        segment_nuclei_func = cellposeSegNucG

    tile_path = Path(tile_path)
    segmented_dir = tile_path.parent / "segmented"
    segmented_filename = tile_path.stem.replace('.ome', '') + '_segmented.tif'
    segmented_path = segmented_dir / segmented_filename

    if segmented_path.exists():
        ts_print(f"Loading pre-segmented masks from: {segmented_path.name}")
        nuclei_masks = tifffile.imread(segmented_path)
        ts_print(f"  Loaded masks shape: {nuclei_masks.shape}")
        ts_print(f"  Number of cells: {nuclei_masks.max()}")
    else:
        ts_print(f"No pre-segmented file found. Running segmentation...")
        nuclei_masks = segment_nuclei_func(image_data)
        ts_print(f"  Segmented masks shape: {nuclei_masks.shape}")
        ts_print(f"  Number of cells: {nuclei_masks.max()}")
        segmented_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(segmented_path, nuclei_masks)
        ts_print(f"  Saved segmentation to: {segmented_path}")

    return nuclei_masks


# =============================================================================
# Label Dilation
# =============================================================================

def dilate_labels_fast(labels, dilation_size):
    """
    Fast label dilation using skimage expand_labels.
    
    Parameters
    ----------
    labels : ndarray
        2D label image.
    dilation_size : int
        Number of pixels to expand each label.
    
    Returns
    -------
    ndarray
        Dilated label image.
    """
    dilated = expand_labels(labels, distance=dilation_size)
    return dilated


def dilate_labels(labels, dilation_size):
    """
    Dilate each label individually using binary dilation (slower).
    
    Parameters
    ----------
    labels : ndarray
        2D label image.
    dilation_size : int
        Number of dilation iterations.
    
    Returns
    -------
    ndarray
        Dilated label image.
    """
    dilated = np.zeros_like(labels)
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = labels == label
        dilated_mask = binary_dilation(mask, iterations=dilation_size)
        dilated[dilated_mask] = label
    return dilated


# =============================================================================
# Spot Extraction
# =============================================================================

def extract_base_intensity(maxed, peaks, cells, threshold_peaks):
    """
    Extract intensity values from maxed image at peak locations.
    
    Background spots (CellLabel = 0) are excluded.
    
    Parameters
    ----------
    maxed : ndarray
        Max-filtered image, shape (num_cycles, num_channels, H, W).
    peaks : ndarray
        2D peak intensity image.
    cells : ndarray
        2D cell label image.
    threshold_peaks : float
        Minimum peak intensity to include.
    
    Returns
    -------
    values : ndarray, shape (N_spots, num_cycles, num_channels)
        Intensity values at each spot location.
    labels : ndarray, shape (N_spots,)
        Cell label for each spot.
    positions : ndarray, shape (N_spots, 2)
        (y, x) coordinates for each spot.
    """
    read_mask = (peaks > threshold_peaks)
    values = maxed[:, :, read_mask].transpose([2, 0, 1])
    labels = cells[read_mask]
    positions = np.array(np.where(read_mask)).T

    # Exclude background spots
    valid_mask = labels > 0
    values = values[valid_mask]
    labels = labels[valid_mask]
    positions = positions[valid_mask]
    return values, labels, positions


# =============================================================================
# Barcode Computation (Two-Color Spot Intensity with Median Correction)
# =============================================================================

def compute_barcodes_two_color(
        intensity_values, labels_mask, positions, min_threshold_intensity=100):
    """
    Compute DNA barcodes from NovaSeq 2-color spot intensity data with median correction.
    
    This function implements a sophisticated barcode calling algorithm optimized for 
    NovaSeq sequencing chemistry with two-color (FITC and Cy3) fluorescence detection.
    It performs median-based spectral unmixing to correct channel crosstalk, assesses 
    each spot individually, and selects the best barcode per cell based on quality scores.
    
    NovaSeq Two-Color Base Calling Logic:
    --------------------------------------
    - **G (Guanine)**: No signal detected (both FITC and Cy3 below threshold)
    - **T (Thymine)**: Cy3 dominant (red channel >> green channel)
    - **A (Adenine)**: FITC dominant (green channel >> red channel)
    - **C (Cytosine)**: Both channels high (FITC ≈ Cy3, both above threshold)
    
    Algorithm Steps:
    ----------------
    1. **Median Correction**: Applies spectral unmixing to correct for channel crosstalk
       and intensity variations across cycles using median-based transformation matrix.
    2. **Individual Spot Assessment**: Each spot is evaluated independently across all 
       sequencing cycles to determine base calls and quality scores.
    3. **Best Barcode Selection**: For cells with multiple spots, the spot with the 
       highest average quality score is selected as the cell's barcode.
    
    Parameters
    ----------
    intensity_values : ndarray, shape (N_spots, num_cycles, 2)
        3D array of spot fluorescence intensities.
        - **N_spots**: Total number of detected spots across all cells
        - **num_cycles**: Number of sequencing cycles (typically 12-16 for barcode length)
        - **2 channels**: [FITC (channel 0), Cy3 (channel 1)]
        
        Example for 1000 spots with 12 cycles:
            intensity_values.shape = (1000, 12, 2)
            intensity_values[0, 0, 0] = FITC intensity for spot 0, cycle 0
            intensity_values[0, 0, 1] = Cy3 intensity for spot 0, cycle 0
        
        Expected value range: 0-65535 (uint16) or 0-100000+ (after processing)
        
    labels_mask : ndarray, shape (N_spots,), dtype int
        Cell label (ID) for each spot. Each spot is assigned to a single cell.
        - Values > 0: Valid cell IDs
        - Value = 0: Background (these should be filtered out before calling this function)
        
        Example:
            labels_mask = [1, 1, 1, 2, 2, 3, 3, 3, 3, ...]
            # Spots 0-2 belong to cell 1, spots 3-4 to cell 2, spots 5-8 to cell 3, etc.
        
    positions : ndarray, shape (N_spots, 2), dtype int
        Spatial (y, x) coordinates for each spot in the image.
        - positions[:, 0] = y-coordinates (row)
        - positions[:, 1] = x-coordinates (column)
        
        Example:
            positions[0] = [512, 1024]  # spot 0 at y=512, x=1024
        
        Used for: Calculating cell centroids and spatial spot distribution
        
    min_threshold_intensity : float, optional (default: 100)
        Minimum intensity threshold for signal detection after median correction.
        Signals below this threshold in both channels are called as 'G' (no signal).
        
        **How to tune this parameter:**
        - **Too low** (e.g., 50): More false positives, noisy base calls, lower quality
        - **Too high** (e.g., 1000): Missing real signals, incomplete barcodes
        - **Recommended starting values:**
            * 100-200: High-quality imaging with good signal-to-noise
            * 300-500: Average quality imaging
            * 500-800: Low signal or high background conditions
        
        **Optimization strategy:**
        1. Start with default (100)
        2. Check quality score distribution in output
        3. Visualize intensity distributions per cycle
        4. Adjust based on your data's signal characteristics
    
    Returns
    -------
    barcode_dataframe_cells : pandas.DataFrame
        Cell-level dataframe with the best barcode for each cell.
        
        **Columns:**
        - **CellLabel** (int): Unique cell identifier matching nuclei segmentation
        - **Barcode** (str): DNA barcode sequence (e.g., 'ATCGATCGATCG' for 12 cycles)
        - **Centroid_X** (float): X-coordinate of cell centroid (mean of all spots)
        - **Centroid_Y** (float): Y-coordinate of cell centroid (mean of all spots)
        - **Quality** (float): Average quality score (0.0-1.0), higher is better
        - **NumSpots** (int): Number of spots detected in this cell
        
        **Shape:** (num_unique_cells, 6)
        
        **Example:**
        ```
           CellLabel         Barcode  Centroid_X  Centroid_Y  Quality  NumSpots
        0          1  ATCGATCGATCG  1024.5      512.3      0.85         3
        1          2  GGCCTTAACCGG   890.2      723.8      0.72         2
        ```
        
    barcode_dataframe_spots : pandas.DataFrame
        Spot-level dataframe with all individual spot barcodes (before best selection).
        
        **Columns:**
        - **CellLabel** (int): Cell containing this spot
        - **Barcode** (str): DNA barcode sequence for this specific spot
        - **Spot_X** (int): X-coordinate of the spot
        - **Spot_Y** (int): Y-coordinate of the spot
        - **Quality** (float): Quality score for this spot (0.0-1.0)
        - **SpotIndex** (int): Unique spot identifier (0 to N_spots-1)
        
        **Shape:** (N_spots, 6)
        
        **Use cases:**
        - Quality control: Inspect variation in barcodes within cells
        - Spatial analysis: Map barcode quality across the image
        - Troubleshooting: Identify problematic spots or cycles
        
    base_calls_array : ndarray, shape (N_spots, num_cycles), dtype str
        Raw base calls for each spot and cycle before selection.
        Each element is a single character: 'A', 'T', 'C', or 'G'
        
        Example:
            base_calls_array[0] = ['A', 'T', 'C', 'G', 'A', 'T', ...]  # spot 0
        
    quality_scores : ndarray, shape (N_spots, num_cycles), dtype float
        Quality scores for each base call (0.0 to 1.0).
        Higher scores indicate more confident base calls.
        
        Quality score interpretation:
        - 0.9-1.0: Excellent signal separation
        - 0.7-0.9: Good quality
        - 0.5-0.7: Acceptable
        - 0.0-0.5: Poor quality, ambiguous signal
        
    corrected_intensity : ndarray, shape (N_spots, num_cycles, 2), dtype float
        Median-corrected intensity values after spectral unmixing.
        Same shape as input intensity_values but with crosstalk correction applied.
        Can be used for visualization or quality control.
    
    Notes
    -----
    - **Input data must be pre-filtered**: Remove background spots (labels_mask == 0) 
      before calling this function, as it only processes spots with valid cell labels.
      
    - **Median correction is automatic**: The function handles channel crosstalk 
      correction internally. No pre-correction needed.
      
    - **Memory usage**: For large datasets (>100k spots), consider processing in batches.
      Memory requirement: ~8 bytes × N_spots × num_cycles × 2 channels
      
    - **Performance**: Typical processing time is ~1-5 seconds per 1000 spots on CPU.
    
    Examples
    --------
    Basic usage with default parameters:
    
    >>> # After extracting spot intensities from images
    >>> val, lab, pos = extract_base_intensity(max_filtered, peaks, cells, threshold_peaks=150)
    >>> 
    >>> # Compute barcodes
    >>> cell_barcodes, spot_barcodes, bases, quality, corrected = \
    ...     compute_barcodes_two_color(val, lab, pos)
    >>> 
    >>> # Save results
    >>> cell_barcodes.to_csv('cell_barcodes.csv', index=False)
    >>> print(f"Found {len(cell_barcodes)} cells with barcodes")
    >>> print(f"Mean quality: {cell_barcodes['Quality'].mean():.3f}")
    
    Adjusting threshold for low-signal data:
    
    >>> # Use lower threshold for dim signals
    >>> cell_barcodes, spot_barcodes, bases, quality, corrected = \
    ...     compute_barcodes_two_color(val, lab, pos, min_threshold_intensity=50)
    
    Quality control workflow:
    
    >>> # Check quality distribution
    >>> import matplotlib.pyplot as plt
    >>> plt.hist(cell_barcodes['Quality'], bins=50)
    >>> plt.xlabel('Quality Score')
    >>> plt.ylabel('Number of Cells')
    >>> 
    >>> # Filter low-quality barcodes
    >>> high_quality = cell_barcodes[cell_barcodes['Quality'] > 0.7]
    >>> print(f"High-quality cells: {len(high_quality)} / {len(cell_barcodes)}")
    >>> 
    >>> # Check for cells with multiple spots
    >>> multi_spot = cell_barcodes[cell_barcodes['NumSpots'] > 1]
    >>> print(f"Cells with multiple spots: {len(multi_spot)}")
    
    See Also
    --------
    extract_base_intensity : Extract spot intensities from max-filtered images
    process_tile_image : Complete pipeline including this function
    
    References
    ----------
    NovaSeq two-color sequencing chemistry:
    Illumina NovaSeq System Guide, Document # 1000000019358
    """

    def transform_medians(X):
        """Estimate and correct differences in channel intensity and spectral overlap."""
        def get_medians(X):
            arr = []
            for i in range(X.shape[1]):
                channel_dominant = X[X.argmax(axis=1) == i]
                if len(channel_dominant) > 0:
                    arr += [np.median(channel_dominant, axis=0)]
                else:
                    arr += [np.zeros(X.shape[1])]
            M = np.array(arr)
            return M

        M = get_medians(X).T
        M = M / M.sum(axis=0)
        W = np.linalg.inv(M)
        Y = W.dot(X.T).T.astype(int)
        return Y, W

    def determine_best_guess_base(I_FITC, I_Cy3, min_thresh):
        """Determine the best guess base when signals are ambiguous."""
        if I_FITC > I_Cy3 * 1.2:
            return 'A'
        elif I_Cy3 > I_FITC * 1.2:
            return 'T'
        elif max(I_FITC, I_Cy3) >= min_thresh * 0.5:
            return 'C'
        else:
            return 'G'

    num_spots, num_cycles, num_channels = intensity_values.shape
    if num_channels != 2:
        raise ValueError(f"Expected 2 channels (FITC, Cy3), got {num_channels}")

    # MEDIAN CORRECTION
    print("Performing median correction with transform_medians...")
    print(f"  Original intensity shape: {intensity_values.shape}")

    corrected_intensity = np.zeros_like(intensity_values, dtype=float)
    all_W = []

    for cycle in range(num_cycles):
        X_cycle = intensity_values[:, cycle, :].astype(float)

        print(f"\n  Cycle {cycle}:")
        print(f"    Original range - FITC: [{X_cycle[:, 0].min():.1f}, {X_cycle[:, 0].max():.1f}], "
              f"Cy3: [{X_cycle[:, 1].min():.1f}, {X_cycle[:, 1].max():.1f}]")

        Y_cycle, W_cycle = transform_medians(X_cycle)
        Y_cycle = Y_cycle.astype(float)
        Y_cycle = np.maximum(Y_cycle, 0)
        corrected_intensity[:, cycle, :] = Y_cycle
        all_W.append(W_cycle)

        print(f"    Corrected range - FITC: [{Y_cycle[:, 0].min():.1f}, {Y_cycle[:, 0].max():.1f}], "
              f"Cy3: [{Y_cycle[:, 1].min():.1f}, {Y_cycle[:, 1].max():.1f}]")

    print(f"\n  Final corrected intensity range: [{corrected_intensity.min():.2f}, {corrected_intensity.max():.2f}]")
    print("Median correction complete.\n")

    intensity_values = corrected_intensity

    # Ratios for 2-color base calling
    ratio_single = 1
    ratio_both = 5

    # Initialize output arrays
    base_calls_by_spot = []
    quality_scores = np.zeros((num_spots, num_cycles), float)

    # INDIVIDUAL SPOT ASSESSMENT
    print("Assessing individual spots...")
    for spot_idx in range(num_spots):
        spot_bases = []
        for cycle in range(num_cycles):
            I_FITC, I_Cy3 = intensity_values[spot_idx, cycle, :]

            max_intensity = max(I_FITC, I_Cy3)
            min_intensity = min(I_FITC, I_Cy3)

            if max_intensity < min_threshold_intensity:
                base = 'G'
                q = 0.0
            elif (I_FITC >= min_threshold_intensity and I_Cy3 >= min_threshold_intensity
                  and (max_intensity / min_intensity) < ratio_both):
                base = 'C'
                q = max(0.0, 1.0 - (max_intensity / min_intensity) / ratio_both)
            elif I_Cy3 >= ratio_single * I_FITC:
                base = 'T'
                q = max(0.0, (I_Cy3 - ratio_single * I_FITC) / I_Cy3) if I_Cy3 > 0 else 0.0
            elif I_FITC >= ratio_single * I_Cy3:
                base = 'A'
                q = max(0.0, (I_FITC - ratio_single * I_Cy3) / I_FITC) if I_FITC > 0 else 0.0
            else:
                base = determine_best_guess_base(I_FITC, I_Cy3, min_threshold_intensity)
                q = 0.1

            spot_bases.append(base)
            quality_scores[spot_idx, cycle] = q

        base_calls_by_spot.append(spot_bases)

    base_calls_array = np.array(base_calls_by_spot)

    # CREATE SPOT-LEVEL DATAFRAME
    print("Creating spot-level dataframe...")
    spot_records = []
    for spot_idx in range(num_spots):
        cell_label = labels_mask[spot_idx]
        spot_barcode = ''.join(base_calls_array[spot_idx, :])
        spot_quality = np.mean(quality_scores[spot_idx, :])
        spot_y, spot_x = positions[spot_idx]

        spot_records.append({
            'CellLabel': cell_label,
            'Barcode': spot_barcode,
            'Spot_X': spot_x,
            'Spot_Y': spot_y,
            'Quality': spot_quality,
            'SpotIndex': spot_idx
        })

    barcode_dataframe_spots = pd.DataFrame(spot_records)

    # CREATE CELL-LEVEL DATAFRAME (best barcode per cell)
    print("Selecting best barcode per cell based on quality...")
    unique_cells = np.unique(labels_mask)
    unique_cells = unique_cells[unique_cells > 0]

    cell_records = []
    for cell_label in unique_cells:
        cell_spots_mask = labels_mask == cell_label
        cell_spot_indices = np.where(cell_spots_mask)[0]

        if len(cell_spot_indices) == 0:
            continue

        spot_mean_qualities = np.mean(quality_scores[cell_spot_indices, :], axis=1)
        best_spot_idx_within_cell = np.argmax(spot_mean_qualities)
        best_spot_global_idx = cell_spot_indices[best_spot_idx_within_cell]

        best_barcode = ''.join(base_calls_array[best_spot_global_idx, :])
        best_quality = spot_mean_qualities[best_spot_idx_within_cell]

        spot_positions = positions[cell_spot_indices]
        centroid_y = np.mean(spot_positions[:, 0])
        centroid_x = np.mean(spot_positions[:, 1])

        cell_records.append({
            'CellLabel': cell_label,
            'Barcode': best_barcode,
            'Centroid_X': centroid_x,
            'Centroid_Y': centroid_y,
            'Quality': best_quality,
            'NumSpots': len(cell_spot_indices)
        })

    barcode_dataframe_cells = pd.DataFrame(cell_records)

    print(f"\nProcessing complete!")
    print(f"  Total spots: {num_spots}")
    print(f"  Total cells: {len(unique_cells)}")
    print(f"  Cell-level dataframe: {barcode_dataframe_cells.shape}")
    print(f"  Spot-level dataframe: {barcode_dataframe_spots.shape}")

    return barcode_dataframe_cells, barcode_dataframe_spots, base_calls_array, quality_scores, intensity_values


# =============================================================================
# Image Stacking
# =============================================================================

def make_image_stack(dir_to_read_file, crop_y_start, crop_x_start, crop_size,
                     rounds=range(2, 7),
                     channel_to_use=None,
                     csv_filename="ome_files_metadata.csv",
                     region='R000'):
    """
    Read OME-TIFF files per round/channel, crop, and stack into a 4D array.
    
    Parameters
    ----------
    dir_to_read_file : str
        Directory containing the OME-TIFF files and metadata CSV.
    crop_y_start : int
        Y-coordinate of the top-left corner of the crop region.
    crop_x_start : int
        X-coordinate of the top-left corner of the crop region.
    crop_size : int
        Size of the square crop region.
    rounds : iterable, optional
        Round numbers to process (default: range(2, 7)).
    channel_to_use : list of str, optional
        Channel names to read. Defaults to DAPI, FITC, Cy3.
    csv_filename : str, optional
        Name of the metadata CSV file (default: "ome_files_metadata.csv").
    region : str, optional
        Region identifier in the metadata (default: 'R000').
    
    Returns
    -------
    image_stack : ndarray, shape (num_rounds, num_channels, crop_size, crop_size)
        Stacked and cropped image data.
    mean_vals : list
        Mean intensity values after background subtraction.
    background_vals : list
        Background (5th percentile) values for each image.
    """
    if channel_to_use is None:
        channel_to_use = [
            'DAPI__FINAL_F',
            'FITC_NOVAseq-AC_FINAL_AFR_F',
            'Cy3_NOVAseq-TC_FINAL_AFR_F'
        ]

    csv_path = os.path.join(dir_to_read_file, csv_filename)
    df = pd.read_csv(csv_path)

    mean_vals = []
    background_vals = []
    image_stack = []

    for round_num in rounds:
        round_channels = []
        for channel_idx, channel_name in enumerate(channel_to_use):
            file_query = df[
                (df['round'] == round_num) &
                (df['region'] == region) &
                (df['channel_marker'] == channel_name)
            ]
            if len(file_query) == 0:
                print(f"Warning: No file found for round={round_num}, channel={channel_name}")
                continue
            file_path = file_query['full_path'].values[0]

            with tifffile.TiffFile(file_path) as tif:
                image_full = tif.asarray()
                # Clamp crop coordinates
                cy_start = min(crop_y_start, image_full.shape[0] - crop_size)
                cx_start = min(crop_x_start, image_full.shape[1] - crop_size)
                cy_end = cy_start + crop_size
                cx_end = cx_start + crop_size
                image_cropped = image_full[cy_start:cy_end, cx_start:cx_end].copy()
                del image_full

            background = np.percentile(image_cropped, 5)
            background_vals.append(background)
            img_corrected = image_cropped.astype(np.float32) - background
            img_corrected = np.clip(img_corrected, 0, None)
            mean_vals.append(np.mean(img_corrected))
            round_channels.append(image_cropped)
            del image_cropped, img_corrected

        if len(round_channels) == len(channel_to_use):
            round_stack = np.stack(round_channels, axis=0)
            image_stack.append(round_stack)
        else:
            print(f"Warning: Skipping round {round_num} - only found "
                  f"{len(round_channels)}/{len(channel_to_use)} channels")

    image_stack = np.stack(image_stack, axis=0)  # (rounds, channels, height, width)
    print(f"Image stack shape: {image_stack.shape}")
    return image_stack, mean_vals, background_vals


# =============================================================================
# Main Tile Processing Pipeline
# =============================================================================

def process_tile_image(tile_path, segment_nuclei_func=None,
                       threshold_peaks=150, min_threshold_intensity=400):
    """
    Process a single tile image through the full OPS pipeline.
    
    Steps:
        1. Load tile from OME-TIFF
        2. Segment nuclei (Cellpose or pre-segmented)
        3. Align SBS cycles (DAPI-based)
        4. Log transform & max filter
        5. Compute std projection & find peaks
        6. Segment cells & dilate nuclei
        7. Extract spot intensities
        8. Compute barcodes with median correction
    
    Parameters
    ----------
    tile_path : str or Path
        Path to the tile OME-TIFF file.
    segment_nuclei_func : callable, optional
        Segmentation function. Defaults to cellposeSegNucG.
    threshold_peaks : int, optional
        Minimum peak intensity for spot detection (default: 150).
    min_threshold_intensity : int, optional
        Minimum intensity threshold for barcode calling (default: 400).
    
    Returns
    -------
    dict
        Dictionary containing all intermediate and final results:
        - image_stack : raw image data
        - nuclei_masks : segmentation labels
        - aligned_images : DAPI-aligned images
        - log_transformed : LoG-filtered images
        - max_filtered : max-filtered images
        - std_projection : standard deviation projection
        - barcode_peaks : detected peak locations
        - cell_masks : cell segmentation labels
        - nuclei_dilated : dilated nuclei labels
        - barcode_dataframe_cells : best barcode per cell (DataFrame)
        - barcode_dataframe_spots : all spot barcodes (DataFrame)
        - base_calls_array : raw base calls
        - quality_scores : quality scores per spot per cycle
    """
    if segment_nuclei_func is None:
        segment_nuclei_func = cellposeSegNucG

    tile_path = Path(tile_path)

    # 1. Load tile
    with tifffile.TiffFile(tile_path) as tif:
        image_stack = tif.asarray()

    data2 = image_stack

    # 2. Segment nuclei
    nuclei_masks = get_or_create_segmentation(
        tile_path, data2[0, 0], segment_nuclei_func=segment_nuclei_func
    )

    # 3. Align, transform, filter
    image_data = data2.astype(np.uint16)
    aligned_images = Snake._align_SBS(image_data, method='DAPI')
    log_transformed = Snake._transform_log(aligned_images, sigma=1, skip_index=0)
    max_filtered = Snake._max_filter(log_transformed, 3, remove_index=0)

    # 4. Std projection & peaks
    std_projection = Snake._compute_std(log_transformed, remove_index=0)
    barcode_peaks = Snake._find_peaks(std_projection)

    # 5. Cell segmentation & dilation
    cell_masks = Snake._segment_cells(image_data[0, 0], nuclei_masks, 100)
    nuclei_dilated = dilate_labels_fast(nuclei_masks, 5)

    # 6. Extract spot intensities
    val, lab, pos = extract_base_intensity(
        max_filtered, barcode_peaks, nuclei_dilated, threshold_peaks=threshold_peaks
    )

    # 7. Compute barcodes
    (barcode_dataframe_cells, barcode_dataframe_spots,
     base_calls_array, quality_scores, _) = \
        compute_barcodes_two_color(
            val, lab, pos, min_threshold_intensity=min_threshold_intensity
        )

    return {
        'image_stack': image_stack,
        'nuclei_masks': nuclei_masks,
        'aligned_images': aligned_images,
        'log_transformed': log_transformed,
        'max_filtered': max_filtered,
        'std_projection': std_projection,
        'barcode_peaks': barcode_peaks,
        'cell_masks': cell_masks,
        'nuclei_dilated': nuclei_dilated,
        'barcode_dataframe_cells': barcode_dataframe_cells,
        'barcode_dataframe_spots': barcode_dataframe_spots,
        'base_calls_array': base_calls_array,
        'quality_scores': quality_scores,
    }
