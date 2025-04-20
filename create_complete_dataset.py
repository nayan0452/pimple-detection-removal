import numpy as np
import cv2
import glob
import os
import re

def natural_sort_key(s):
    """Sort strings with numbers in a natural way (e.g., data1, data2, ..., data10)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    # Find all original data and label files (exclude the ones we created earlier)
    # We only want files like data1.npz, data2.npz, etc. (not data_all_combined.npz)
    data_files = glob.glob("data[0-9]*.npz")
    label_files = glob.glob("lebels[0-9]*.npz")
    
    # Sort files naturally so data10 comes after data9, not after data1
    data_files = sorted(data_files, key=natural_sort_key)
    label_files = sorted(label_files, key=natural_sort_key)
    
    print(f"Found {len(data_files)} data files and {len(label_files)} label files")
    
    # Make sure we have matching pairs
    pairs = []
    total_patches = 0
    total_grids = 0
    
    for data_file in data_files:
        # Extract the number from the data file (e.g., "data1.npz" -> "1")
        match = re.search(r'data(\d+)\.npz', data_file)
        if match:
            num = match.group(1)
            label_file = f"lebels{num}.npz"
            
            if label_file in label_files:
                pairs.append((data_file, label_file))
                
                # Count the actual number of patches
                data = np.load(data_file)['arr_0']
                total_patches += len(data)
                total_grids += len(data) // 225  # 15x15 grid
                
                print(f"Pair {len(pairs)}: {data_file} and {label_file} - {len(data)} patches, {len(data) // 225} grids")
    
    print(f"\nTotal valid pairs: {len(pairs)}")
    print(f"Total patches: {total_patches}")
    print(f"Expected total grids (15x15): {total_grids}")
    
    # Process each pair to create a dataset
    all_data = []
    all_labels = []
    
    for i, (data_file, label_file) in enumerate(pairs):
        print(f"\nProcessing pair {i+1}/{len(pairs)}: {data_file} and {label_file}")
        
        # Load data and labels
        data = np.load(data_file)['arr_0']
        labels = np.load(label_file)['arr_0']
        
        # Check if lengths match
        if len(data) != len(labels):
            print(f"  Warning: Data length ({len(data)}) doesn't match labels length ({len(labels)})")
            min_len = min(len(data), len(labels))
            data = data[:min_len]
            labels = labels[:min_len]
            print(f"  Using {min_len} elements")
        
        # Define grid dimensions
        n_row = 15
        n_col = 15
        cells_per_grid = n_row * n_col
        
        # Calculate how many complete grids we can form
        n_complete_grids = len(data) // cells_per_grid
        print(f"  Creating {n_complete_grids} complete image grids")
        
        # Process each grid
        for grid_idx in range(n_complete_grids):
            # Get patches and labels for this grid
            start_idx = grid_idx * cells_per_grid
            end_idx = start_idx + cells_per_grid
            
            grid_patches = data[start_idx:end_idx]
            grid_labels = labels[start_idx:end_idx]
            
            # Reshape labels to 15x15 grid
            grid_labels = grid_labels.reshape(n_row, n_col)
            
            # Reconstruct the full image by stitching patches
            patch_h, patch_w = grid_patches[0].shape[:2]
            reconstructed_img = np.zeros((patch_h * n_row, patch_w * n_col, 3), dtype=np.uint8)
            
            # Place each patch in the reconstructed image
            for idx, patch in enumerate(grid_patches):
                r = idx // n_col  # row in the grid
                c = idx % n_col   # column in the grid
                
                y1 = r * patch_h
                y2 = (r + 1) * patch_h
                x1 = c * patch_w
                x2 = (c + 1) * patch_w
                
                reconstructed_img[y1:y2, x1:x2] = patch
            
            # Add to our collections
            all_data.append(reconstructed_img)
            all_labels.append(grid_labels)
    
    # Convert lists to numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    print(f"\nFinal dataset:")
    print(f"Data shape: {all_data.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    # Create the dataset dictionary
    dataset = {
        'data': all_data,
        'labels': all_labels,
        'n_row': 15,
        'n_col': 15
    }
    
    # Save the dataset
    output_file = "dataset_complete.npz"
    np.savez(output_file, **dataset)
    print(f"Complete dataset saved to {output_file}")
    
    # Verify the saved file
    saved_data = np.load(output_file)
    print(f"\nVerification:")
    print(f"Keys in saved file: {list(saved_data.keys())}")
    print(f"Data shape: {saved_data['data'].shape}")
    print(f"Labels shape: {saved_data['labels'].shape}")

if __name__ == "__main__":
    main() 