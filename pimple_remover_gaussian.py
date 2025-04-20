import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def get_label_color(label):
    """
    Returns a color based on the label:
    0 (other) - Yellow
    1 (pimple) - Red
    2 (spotless skin) - Green
    """
    if label == 0:
        return (0, 255, 255)  # Yellow (BGR)
    elif label == 1:
        return (0, 0, 255)    # Red (BGR)
    elif label == 2:
        return (0, 255, 0)    # Green (BGR)
    else:
        return (255, 255, 255)  # White for any other value

def remove_pimples_gaussian(grid_data, labels, n_rows=15, n_cols=15, blur_kernel_size=(7, 7), blur_sigma=3):
    """
    Remove pimples by applying Gaussian blur selectively to pimple areas
    
    Parameters:
    - grid_data: Array of cell image data
    - labels: Array of cell labels (0=other, 1=pimple, 2=spotless skin)
    - n_rows, n_cols: Grid dimensions
    - blur_kernel_size: Kernel size for Gaussian blur (width, height)
    - blur_sigma: Sigma value for Gaussian blur
    
    Returns:
    - Array of processed cell image data with pimples removed
    """
    # Create a copy of the data for the result
    result_data = grid_data.copy()
    
    # Reshape the labels to n_rows x n_cols grid
    grid_labels = labels.reshape(n_rows, n_cols)
    
    # Process each cell labeled as pimple (1)
    for row in range(n_rows):
        for col in range(n_cols):
            if grid_labels[row, col] == 1:  # If it's a pimple (label 1)
                # Get the pimple index
                pimple_idx = row * n_cols + col
                pimple_cell = grid_data[pimple_idx]
                
                try:
                    # Apply Gaussian blur to the pimple cell
                    blurred_cell = cv.GaussianBlur(pimple_cell, blur_kernel_size, blur_sigma)
                    
                    # Apply the blurred result
                    result_data[pimple_idx] = blurred_cell
                    
                except Exception as e:
                    print(f"Error processing pimple at ({row},{col}): {e}")
    
    return result_data

def visualize_grid(grid_data, labels, n_rows=15, n_cols=15, cell_size=60):
    """
    Create a visualization of a grid with its labels
    """
    # Reshape labels
    grid_labels = labels.reshape(n_rows, n_cols)
    
    # Create a blank canvas
    display_size = n_rows * cell_size
    display_img = np.ones((display_size, display_size, 3), dtype=np.uint8) * 255
    
    # Place each cell in the display
    start_idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            # Get and resize the cell
            cell_data = grid_data[start_idx]
            cell_data = cv.resize(cell_data, (cell_size, cell_size))
            start_idx += 1
            
            # Position in display
            y1 = i * cell_size
            y2 = (i + 1) * cell_size
            x1 = j * cell_size
            x2 = (j + 1) * cell_size
            
            # Place cell in display
            display_img[y1:y2, x1:x2] = cell_data
            
            # Draw label
            label_value = grid_labels[i, j]
            text_pos = (x1 + cell_size//5, y1 + cell_size//5)
            cv.putText(
                display_img, 
                f"{label_value}", 
                text_pos,
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                thickness=1,
                color=get_label_color(label_value)
            )
            
            # Draw grid lines
            cv.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    
    return display_img

def main():
    # Find all label and data files
    label_files = sorted(glob.glob("lebels*.npz"))
    data_files = sorted(glob.glob("data*.npz"))
    
    if not label_files or not data_files:
        print("No data files found. Make sure label and data .npz files exist in the current directory.")
        return
    
    print(f"Found {len(label_files)} label files and {len(data_files)} data files.")
    
    # Load all datasets
    all_datasets = []
    for label_file, data_file in zip(label_files, data_files):
        print(f"Loading {label_file} and {data_file}...")
        
        # Load the data
        labels = np.load(label_file)['arr_0']  # Default array name in npz files
        data = np.load(data_file)['arr_0']
        
        # Calculate how many complete 15x15 grids we have
        n_cells = 15 * 15  # 225 cells per complete grid
        n_complete_grids = len(data) // n_cells
        
        print(f"Dataset contains {n_complete_grids} complete grids")
        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
        
        all_datasets.append({
            'labels': labels,
            'data': data,
            'n_grids': n_complete_grids,
            'label_file': label_file,
            'data_file': data_file
        })
    
    if not all_datasets:
        print("No valid datasets found.")
        return
    
    # Start browsing from the first dataset and first grid
    current_dataset_idx = 0
    current_grid_idx = 0
    
    # Default blur parameters
    blur_kernel_size = (7, 7)
    blur_sigma = 3
    
    # Main visualization loop
    while True:
        # Get current dataset and grid
        dataset = all_datasets[current_dataset_idx]
        n_cells = 15 * 15
        start_idx = current_grid_idx * n_cells
        end_idx = start_idx + n_cells
        
        # Get data for the current grid
        grid_data = dataset['data'][start_idx:end_idx]
        grid_labels = dataset['labels'][start_idx:end_idx]
        
        # Apply pimple removal with Gaussian blur
        processed_data = remove_pimples_gaussian(grid_data, grid_labels, 
                                               blur_kernel_size=blur_kernel_size, 
                                               blur_sigma=blur_sigma)
        
        # Create visualizations
        original_img = visualize_grid(grid_data, grid_labels)
        processed_img = visualize_grid(processed_data, grid_labels)
        
        # Create side-by-side comparison
        comparison = np.hstack((original_img, processed_img))
        
        # Show the comparison
        window_name = f"Dataset {current_dataset_idx+1}/{len(all_datasets)} - Grid {current_grid_idx+1}/{dataset['n_grids']} - Gaussian Blur"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.imshow(window_name, comparison)
        
        # Add text labels to indicate which is which
        h, w = original_img.shape[:2]
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(comparison, "Original", (w//4, 30), font, 1, (0, 0, 255), 2)
        cv.putText(comparison, f"Gaussian Blur ({blur_kernel_size}, sigma={blur_sigma})", (w + w//4, 30), font, 0.8, (0, 0, 255), 2)
        
        # Instructions
        print("\nViewing dataset", current_dataset_idx + 1, "of", len(all_datasets))
        print("Viewing grid", current_grid_idx + 1, "of", dataset['n_grids'])
        print(f"Current blur settings: kernel={blur_kernel_size}, sigma={blur_sigma}")
        print("Controls:")
        print("  'n' or Right arrow - Next grid")
        print("  'p' or Left arrow - Previous grid")
        print("  's' - Save current before/after comparison")
        print("  'j' or Down arrow - Next dataset")
        print("  'k' or Up arrow - Previous dataset")
        print("  '+' - Increase blur sigma")
        print("  '-' - Decrease blur sigma")
        print("  '>' - Increase kernel size")
        print("  '<' - Decrease kernel size")
        print("  'q' or ESC - Quit")
        
        # Wait for key press
        key = cv.waitKey(0) & 0xFF
        
        # Handle key press for navigation
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('s'):  # Save image
            save_name = f"pimple_removal_gaussian_dataset{current_dataset_idx+1}_grid{current_grid_idx+1}.png"
            cv.imwrite(save_name, comparison)
            print(f"Saved comparison as {save_name}")
        elif key == ord('n') or key == 83 or key == 3 or key == 100:  # 'n' or right arrow
            # Try different key codes for right arrow across platforms
            current_grid_idx += 1
            if current_grid_idx >= dataset['n_grids']:
                # Move to next dataset if at the end of current dataset's grids
                current_grid_idx = 0
                current_dataset_idx = (current_dataset_idx + 1) % len(all_datasets)
        elif key == ord('p') or key == 81 or key == 2 or key == 97:  # 'p' or left arrow
            # Try different key codes for left arrow across platforms
            current_grid_idx -= 1
            if current_grid_idx < 0:
                # Move to previous dataset if at the beginning of current dataset's grids
                current_dataset_idx = (current_dataset_idx - 1) % len(all_datasets)
                current_grid_idx = all_datasets[current_dataset_idx]['n_grids'] - 1
        elif key == ord('j') or key == 84 or key == 1 or key == 115:  # 'j' or down arrow
            # Next dataset, keep the same relative grid position
            current_dataset_idx = (current_dataset_idx + 1) % len(all_datasets)
            # Adjust grid index if it's beyond the range of the new dataset
            current_grid_idx = min(current_grid_idx, all_datasets[current_dataset_idx]['n_grids'] - 1)
        elif key == ord('k') or key == 82 or key == 0 or key == 119:  # 'k' or up arrow
            # Previous dataset, keep the same relative grid position
            current_dataset_idx = (current_dataset_idx - 1) % len(all_datasets)
            # Adjust grid index if it's beyond the range of the new dataset
            current_grid_idx = min(current_grid_idx, all_datasets[current_dataset_idx]['n_grids'] - 1)
        elif key == ord('+'):  # Increase blur sigma
            blur_sigma = min(20, blur_sigma + 1)
            print(f"Increased blur sigma to {blur_sigma}")
        elif key == ord('-'):  # Decrease blur sigma
            blur_sigma = max(1, blur_sigma - 1)
            print(f"Decreased blur sigma to {blur_sigma}")
        elif key == ord('>'):  # Increase kernel size
            # Kernel size must be odd
            kernel_size = max(blur_kernel_size[0], blur_kernel_size[1]) + 2
            blur_kernel_size = (kernel_size, kernel_size)
            print(f"Increased kernel size to {blur_kernel_size}")
        elif key == ord('<'):  # Decrease kernel size
            # Kernel size must be odd and >= 3
            kernel_size = max(3, min(blur_kernel_size[0], blur_kernel_size[1]) - 2)
            blur_kernel_size = (kernel_size, kernel_size)
            print(f"Decreased kernel size to {blur_kernel_size}")
        
        # Close the current window before showing the next
        cv.destroyWindow(window_name)
    
    cv.destroyAllWindows()
    print("Gaussian blur pimple removal visualization complete.")

if __name__ == "__main__":
    main() 