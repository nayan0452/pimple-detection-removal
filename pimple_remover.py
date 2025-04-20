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

def find_nearest_spotless_skin(labels, row, col, n_rows=15, n_cols=15, search_radius=3):
    """
    Find the nearest grid cells labeled as spotless skin (2) around a pimple (1)
    """
    spotless_skin_indices = []
    
    # Define search area
    row_start = max(0, row - search_radius)
    row_end = min(n_rows - 1, row + search_radius)
    col_start = max(0, col - search_radius)
    col_end = min(n_cols - 1, col + search_radius)
    
    # Search for spotless skin in surrounding area
    for r in range(row_start, row_end + 1):
        for c in range(col_start, col_end + 1):
            if labels[r, c] == 2:  # If it's spotless skin (label 2)
                spotless_skin_indices.append((r, c))
    
    return spotless_skin_indices

def remove_pimples(grid_data, labels, n_rows=15, n_cols=15):
    """
    Remove pimples by replacing them with the average of surrounding spotless skin areas
    Using advanced color processing for better results
    """
    # Create a copy of the data for the result
    result_data = grid_data.copy()
    
    # Reshape the labels to 15x15 grid
    grid_labels = labels.reshape(n_rows, n_cols)
    
    # First, find all spotless skin cells across the entire image
    all_spotless_cells = []
    for row in range(n_rows):
        for col in range(n_cols):
            if grid_labels[row, col] == 2:  # If it's spotless skin
                idx = row * n_cols + col
                all_spotless_cells.append((idx, grid_data[idx]))
    
    # If we don't have any spotless skin cells, we can't do much
    if not all_spotless_cells:
        print("Warning: No spotless skin cells found in this image.")
        return result_data
    
    # Calculate an average skin color from all spotless skin cells
    all_skin_avg = np.zeros_like(grid_data[0], dtype=np.float32)
    for idx, cell_data in all_spotless_cells:
        all_skin_avg += cell_data.astype(np.float32)
    all_skin_avg = all_skin_avg / len(all_spotless_cells)
    
    # Calculate average brightness of spotless skin
    avg_brightness = np.mean(all_skin_avg)
    print(f"Average spotless skin brightness: {avg_brightness}")
    
    # Process each cell labeled as pimple (1)
    for row in range(n_rows):
        for col in range(n_cols):
            if grid_labels[row, col] == 1:  # If it's a pimple (label 1)
                # Find surrounding spotless skin areas
                spotless_indices = find_nearest_spotless_skin(grid_labels, row, col)
                
                # Get the pimple index
                pimple_idx = row * n_cols + col
                pimple_cell = grid_data[pimple_idx].astype(np.float32)
                
                try:
                    # Convert pimple to HSV for better color manipulation
                    pimple_rgb = cv.cvtColor(np.uint8([pimple_cell]), cv.COLOR_BGR2RGB)[0]
                    pimple_hsv = cv.cvtColor(np.uint8([pimple_rgb]), cv.COLOR_RGB2HSV)[0]
                    
                    if spotless_indices and len(spotless_indices) >= 1:
                        # Calculate the average of surrounding spotless skin cells
                        avg_skin = np.zeros_like(pimple_cell, dtype=np.float32)
                        for r, c in spotless_indices:
                            spotless_idx = r * n_cols + c
                            avg_skin += grid_data[spotless_idx].astype(np.float32)
                        
                        avg_skin = avg_skin / len(spotless_indices)
                        
                        # Convert to HSV for better color blending
                        avg_skin_rgb = cv.cvtColor(np.uint8([avg_skin]), cv.COLOR_BGR2RGB)[0]
                        avg_skin_hsv = cv.cvtColor(np.uint8([avg_skin_rgb]), cv.COLOR_RGB2HSV)[0]
                        
                        # Create blended HSV
                        h, s, v = cv.split(avg_skin_hsv)
                        _, _, v_pimple = cv.split(pimple_hsv)
                        
                        # Blend the brightness (value) channel
                        v_blend = np.uint8(0.7 * v + 0.3 * v_pimple)
                        
                        # Boost brightness if too dark
                        if np.mean(v_blend) < 150:
                            v_blend = np.minimum(np.uint8(v_blend * 1.2), np.uint8(255))
                        
                        # Merge back to HSV
                        blended_hsv = cv.merge([h, s, v_blend])
                        
                        # Convert back to BGR
                        blended_rgb = cv.cvtColor(blended_hsv[np.newaxis, :, :], cv.COLOR_HSV2RGB)[0]
                        blended_bgr = cv.cvtColor(blended_rgb[np.newaxis, :, :], cv.COLOR_RGB2BGR)[0]
                        
                        # Apply the blended result
                        result_data[pimple_idx] = blended_bgr
                    else:
                        # If no nearby spotless skin found, use the global average with brightness adjustment
                        global_avg_rgb = cv.cvtColor(np.uint8([all_skin_avg]), cv.COLOR_BGR2RGB)[0]
                        global_avg_hsv = cv.cvtColor(np.uint8([global_avg_rgb]), cv.COLOR_RGB2HSV)[0]
                        
                        # Extract channels
                        h, s, v = cv.split(global_avg_hsv)
                        _, _, v_pimple = cv.split(pimple_hsv)
                        
                        # Blend the brightness (value) channel
                        v_blend = np.uint8(0.65 * v + 0.35 * v_pimple)
                        
                        # Boost brightness if needed
                        v_blend = np.minimum(np.uint8(v_blend * 1.25), np.uint8(255))
                        
                        # Merge back to HSV
                        blended_hsv = cv.merge([h, s, v_blend])
                        
                        # Convert back to BGR
                        blended_rgb = cv.cvtColor(blended_hsv[np.newaxis, :, :], cv.COLOR_HSV2RGB)[0]
                        blended_bgr = cv.cvtColor(blended_rgb[np.newaxis, :, :], cv.COLOR_RGB2BGR)[0]
                        
                        # Apply the blended result
                        result_data[pimple_idx] = blended_bgr
                except Exception as e:
                    print(f"Error processing pimple at ({row},{col}): {e}")
                    # Fallback to simple averaging if HSV processing fails
                    if spotless_indices:
                        avg_skin = np.zeros_like(pimple_cell, dtype=np.float32)
                        for r, c in spotless_indices:
                            spotless_idx = r * n_cols + c
                            avg_skin += grid_data[spotless_idx].astype(np.float32)
                        
                        avg_skin = avg_skin / len(spotless_indices)
                        result_data[pimple_idx] = np.uint8(avg_skin)
    
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
        
        # Apply pimple removal
        processed_data = remove_pimples(grid_data, grid_labels)
        
        # Create visualizations
        original_img = visualize_grid(grid_data, grid_labels)
        processed_img = visualize_grid(processed_data, grid_labels)
        
        # Create side-by-side comparison
        comparison = np.hstack((original_img, processed_img))
        
        # Show the comparison
        window_name = f"Dataset {current_dataset_idx+1}/{len(all_datasets)} - Grid {current_grid_idx+1}/{dataset['n_grids']} - {dataset['label_file']}"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.imshow(window_name, comparison)
        
        # Add text labels to indicate which is which
        h, w = original_img.shape[:2]
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(comparison, "Original", (w//4, 30), font, 1, (0, 0, 255), 2)
        cv.putText(comparison, "Pimples Removed", (w + w//4, 30), font, 1, (0, 0, 255), 2)
        
        # Instructions
        print("\nViewing dataset", current_dataset_idx + 1, "of", len(all_datasets))
        print("Viewing grid", current_grid_idx + 1, "of", dataset['n_grids'])
        print("Controls:")
        print("  'n' or Right arrow - Next grid")
        print("  'p' or Left arrow - Previous grid")
        print("  's' - Save current before/after comparison")
        print("  'j' or Down arrow - Next dataset")
        print("  'k' or Up arrow - Previous dataset")
        print("  'q' or ESC - Quit")
        
        # Wait for key press
        key = cv.waitKey(0) & 0xFF
        
        # Handle key press for navigation
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('s'):  # Save image
            save_name = f"pimple_removal_dataset{current_dataset_idx+1}_grid{current_grid_idx+1}.png"
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
        
        # Close the current window before showing the next
        cv.destroyWindow(window_name)
    
    cv.destroyAllWindows()
    print("Pimple removal visualization complete.")

if __name__ == "__main__":
    main() 