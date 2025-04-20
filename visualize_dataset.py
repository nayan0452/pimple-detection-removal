import cv2 as cv
import numpy as np
import os
import glob

def get_label_color(label):
    """
    Returns a color based on the label:
    0 (non-skin) - Yellow
    1 (custom) - Red
    2 (skin) - Black
    """
    if label == 0:
        return (0, 255, 255)  # Yellow (BGR)
    elif label == 1:
        return (0, 0, 255)    # Red (BGR)
    elif label == 2:
        return (0, 0, 0)      # Black (BGR)
    else:
        return (255, 255, 255)  # White for any other value

def display_grid(grid_data, label, n_cells=15, cell_size=40):
    """
    Creates a visualization for a single grid of 15x15 cells
    with each cell colored according to its label
    """
    # Create a large enough canvas to display all cells
    display_size = n_cells * cell_size
    display_img = np.ones((display_size, display_size, 3), dtype=np.uint8) * 255
    
    # Reshape the labels to the original grid structure
    grid_labels = label.reshape(n_cells, n_cells)
    
    # Calculate indices for the data array
    start_idx = 0
    
    # Place each grid cell in the display image
    for i in range(n_cells):
        for j in range(n_cells):
            # Get the cell data and resize it
            cell_data = grid_data[start_idx]
            cell_data = cv.resize(cell_data, (cell_size, cell_size))
            start_idx += 1
            
            # Calculate position in display image
            y1 = i * cell_size
            y2 = (i + 1) * cell_size
            x1 = j * cell_size
            x2 = (j + 1) * cell_size
            
            # Place the cell in the display image
            display_img[y1:y2, x1:x2] = cell_data
            
            # Draw the label on top of the cell
            label_value = grid_labels[i, j]
            text_pos = (x1 + cell_size//4, y1 + cell_size//2)
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
    
    # Load all datasets at once
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
        
        # Create visualization
        display_img = display_grid(grid_data, grid_labels)
        
        # Show the visualization
        window_name = f"Dataset {current_dataset_idx+1}/{len(all_datasets)} - Grid {current_grid_idx+1}/{dataset['n_grids']} - {dataset['label_file']}"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.imshow(window_name, display_img)
        
        # Instructions
        print("\nViewing dataset", current_dataset_idx + 1, "of", len(all_datasets))
        print("Viewing grid", current_grid_idx + 1, "of", dataset['n_grids'])
        print("Controls:")
        print("  'n' or Right arrow - Next grid")
        print("  'p' or Left arrow - Previous grid")
        print("  'j' or Down arrow - Next dataset")
        print("  'k' or Up arrow - Previous dataset")
        print("  'q' or ESC - Quit")
        
        # Wait for key press
        key = cv.waitKey(0) & 0xFF
        
        # Handle key press for navigation
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
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
    print("Visualization complete.")

if __name__ == "__main__":
    main() 