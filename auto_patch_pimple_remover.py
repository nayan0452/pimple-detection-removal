import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature

def get_label_color(label):
    """
    Returns a color based on the label:
    0 (no skin/no pimple) - Yellow
    1 (pimple) - Red
    2 (usable skin) - Green
    """
    if label == 0:
        return (0, 255, 255)  # Yellow (BGR)
    elif label == 1:
        return (0, 0, 255)    # Red (BGR)
    elif label == 2:
        return (0, 255, 0)    # Green (BGR)
    else:
        return (255, 255, 255)  # White for any other value

def find_nearest_similar_skin(grid_data, labels, pimple_row, pimple_col, n_rows=15, n_cols=15, search_radius=3):
    """
    Find the nearest cell labeled as usable skin (2) that has similar characteristics to the pimple
    
    Parameters:
    - grid_data: Array of cell image data
    - labels: Reshaped labels array (n_rows x n_cols)
    - pimple_row, pimple_col: Position of the pimple to find skin for
    - search_radius: How many cells to search around (increase for larger search area)
    
    Returns:
    - (row, col) of the best matching skin cell, or None if none found
    """
    # Get the pimple cell
    pimple_idx = pimple_row * n_cols + pimple_col
    pimple_cell = grid_data[pimple_idx]
    
    # Get surrounding context of the pimple (neighboring cells)
    context_cells = []
    for r_offset in range(-1, 2):  # -1, 0, 1
        for c_offset in range(-1, 2):  # -1, 0, 1
            if r_offset == 0 and c_offset == 0:
                continue  # Skip the pimple cell itself
                
            r = (pimple_row + r_offset) % n_rows
            c = (pimple_col + c_offset) % n_cols
            
            # Only consider valid skin cells for context
            if labels[r, c] == 2:  # If it's usable skin
                context_cells.append(grid_data[r * n_cols + c])
    
    # If we have context cells, compute their average color features
    context_color_features = {}
    if context_cells:
        # Convert to different color spaces and compute averages
        for cell in context_cells:
            # BGR (original)
            if 'bgr' not in context_color_features:
                context_color_features['bgr'] = np.zeros(3)
            context_color_features['bgr'] += np.mean(cell, axis=(0, 1))
            
            # HSV (better for skin tone)
            hsv_cell = cv.cvtColor(cell, cv.COLOR_BGR2HSV)
            if 'hsv' not in context_color_features:
                context_color_features['hsv'] = np.zeros(3)
            context_color_features['hsv'] += np.mean(hsv_cell, axis=(0, 1))
            
            # LAB (perceptually uniform)
            lab_cell = cv.cvtColor(cell, cv.COLOR_BGR2Lab)
            if 'lab' not in context_color_features:
                context_color_features['lab'] = np.zeros(3)
            context_color_features['lab'] += np.mean(lab_cell, axis=(0, 1))
        
        # Average the features
        for key in context_color_features:
            context_color_features[key] /= len(context_cells)
    
    # Convert pimple cell to different color spaces
    pimple_hsv = cv.cvtColor(pimple_cell, cv.COLOR_BGR2HSV)
    pimple_lab = cv.cvtColor(pimple_cell, cv.COLOR_BGR2Lab)
    
    # Calculate pimple features
    pimple_features = {
        'bgr_mean': np.mean(pimple_cell, axis=(0, 1)),
        'hsv_mean': np.mean(pimple_hsv, axis=(0, 1)),
        'lab_mean': np.mean(pimple_lab, axis=(0, 1)),
        'brightness': np.mean(pimple_cell),
    }
    
    # Create texture descriptors for the pimple
    pimple_gray = cv.cvtColor(pimple_cell, cv.COLOR_BGR2GRAY)
    pimple_texture = feature.local_binary_pattern(pimple_gray, P=8, R=1, method='uniform')
    pimple_hist, _ = np.histogram(pimple_texture.ravel(), bins=10, range=(0, 10), density=True)
    pimple_features['texture_hist'] = pimple_hist
    
    # Search for suitable skin cells
    best_match = None
    best_score = float('inf')
    
    # Collect all candidate scores for global optimization
    candidate_scores = []
    
    # Define search area with wraparound (circular buffer style)
    for r_offset in range(-search_radius, search_radius + 1):
        for c_offset in range(-search_radius, search_radius + 1):
            # Skip the pimple cell itself
            if r_offset == 0 and c_offset == 0:
                continue
                
            # Calculate target position with wraparound
            r = (pimple_row + r_offset) % n_rows
            c = (pimple_col + c_offset) % n_cols
            
            # Check if this is a usable skin cell (label 2)
            if labels[r, c] == 2:
                # Get the skin cell
                skin_idx = r * n_cols + c
                skin_cell = grid_data[skin_idx]
                
                # Convert skin cell to different color spaces for better comparison
                skin_hsv = cv.cvtColor(skin_cell, cv.COLOR_BGR2HSV)
                skin_lab = cv.cvtColor(skin_cell, cv.COLOR_BGR2Lab)
                
                # Calculate skin features
                skin_bgr_mean = np.mean(skin_cell, axis=(0, 1))
                skin_hsv_mean = np.mean(skin_hsv, axis=(0, 1))
                skin_lab_mean = np.mean(skin_lab, axis=(0, 1))
                skin_brightness = np.mean(skin_cell)
                
                # Calculate texture features for skin cell
                skin_gray = cv.cvtColor(skin_cell, cv.COLOR_BGR2GRAY)
                skin_texture = feature.local_binary_pattern(skin_gray, P=8, R=1, method='uniform')
                skin_hist, _ = np.histogram(skin_texture.ravel(), bins=10, range=(0, 10), density=True)
                
                # Calculate feature differences
                bgr_diff = np.sum(np.abs(skin_bgr_mean - pimple_features['bgr_mean']))
                
                # For HSV, handle the circular nature of Hue
                h_diff = min(abs(skin_hsv_mean[0] - pimple_features['hsv_mean'][0]),
                            360 - abs(skin_hsv_mean[0] - pimple_features['hsv_mean'][0])) / 180.0
                sv_diff = np.sum(np.abs(skin_hsv_mean[1:] - pimple_features['hsv_mean'][1:])) / 255.0
                hsv_diff = h_diff + sv_diff
                
                # LAB difference (perceptually uniform)
                lab_diff = np.sum(np.abs(skin_lab_mean - pimple_features['lab_mean'])) / 255.0
                
                # Brightness difference
                brightness_diff = abs(skin_brightness - pimple_features['brightness']) / 255.0
                
                # Texture difference (using histogram intersection)
                texture_diff = np.sum(np.abs(skin_hist - pimple_features['texture_hist']))
                
                # Context similarity if we have context
                context_score = 0
                if context_cells:
                    # Compare skin cell with context features
                    context_bgr_diff = np.sum(np.abs(skin_bgr_mean - context_color_features['bgr']))
                    
                    # For HSV context
                    context_h_diff = min(abs(skin_hsv_mean[0] - context_color_features['hsv'][0]),
                                        360 - abs(skin_hsv_mean[0] - context_color_features['hsv'][0])) / 180.0
                    context_sv_diff = np.sum(np.abs(skin_hsv_mean[1:] - context_color_features['hsv'][1:])) / 255.0
                    context_hsv_diff = context_h_diff + context_sv_diff
                    
                    # LAB context difference
                    context_lab_diff = np.sum(np.abs(skin_lab_mean - context_color_features['lab'])) / 255.0
                    
                    # Combined context score
                    context_score = (context_bgr_diff + context_hsv_diff + context_lab_diff) / 3.0
                
                # Distance factor - prefer closer cells, but not too much weight
                distance = np.sqrt(r_offset**2 + c_offset**2) / search_radius
                
                # Calculate combined score (weighted average of all factors)
                # Adjust weights based on importance
                score = (
                    0.15 * bgr_diff +         # BGR color difference
                    0.20 * hsv_diff +         # HSV color difference (good for skin)
                    0.20 * lab_diff +         # LAB color difference (perceptual)
                    0.15 * brightness_diff +  # Brightness difference
                    0.15 * texture_diff +     # Texture difference
                    0.10 * context_score +    # Context similarity
                    0.05 * distance           # Distance factor (small weight)
                )
                
                # Store candidate with score
                candidate_scores.append((score, r, c))
                
                # Update best match if this is better
                if score < best_score:
                    best_score = score
                    best_match = (r, c)
    
    # If we have enough candidates, do a global optimization
    # This helps avoid local minima and find better overall patches
    if len(candidate_scores) >= 5:
        # Sort by score
        candidate_scores.sort()
        # Take top 3 and re-evaluate with additional criteria
        top_candidates = candidate_scores[:3]
        
        best_global_match = None
        best_global_score = float('inf')
        
        for _, r, c in top_candidates:
            skin_idx = r * n_cols + c
            skin_cell = grid_data[skin_idx]
            
            # Evaluate edge continuity - how well the edges of the skin patch
            # would continue with the edges around the pimple
            edge_score = evaluate_edge_continuity(grid_data, labels, skin_cell, pimple_row, pimple_col, r, c, n_rows, n_cols)
            
            # Evaluate variance - prefer skin patches with similar variance to surrounding area
            variance_score = evaluate_variance_similarity(grid_data, labels, skin_cell, pimple_row, pimple_col, r, c, n_rows, n_cols)
            
            # Global score combining multiple factors
            global_score = 0.5 * candidate_scores[candidate_scores.index((_, r, c))][0] + 0.3 * edge_score + 0.2 * variance_score
            
            if global_score < best_global_score:
                best_global_score = global_score
                best_global_match = (r, c)
        
        # Use the globally optimized match if it's good
        if best_global_match:
            best_match = best_global_match
    
    return best_match

def evaluate_edge_continuity(grid_data, labels, skin_cell, pimple_row, pimple_col, skin_row, skin_col, n_rows, n_cols):
    """
    Evaluate how well the edges of a skin patch would continue with the edges around a pimple
    Returns a score where lower is better
    """
    # Simple edge detection on skin cell
    gray_skin = cv.cvtColor(skin_cell, cv.COLOR_BGR2GRAY)
    edges_skin = cv.Canny(gray_skin, 50, 150)
    
    # Check surrounding cells of pimple for edge patterns
    edge_continuity_score = 0
    count = 0
    
    for r_offset in range(-1, 2):
        for c_offset in range(-1, 2):
            if r_offset == 0 and c_offset == 0:
                continue  # Skip pimple itself
                
            r = (pimple_row + r_offset) % n_rows
            c = (pimple_col + c_offset) % n_cols
            
            # Only consider valid skin cells
            if labels[r, c] == 2:
                neighbor_idx = r * n_cols + c
                neighbor_cell = grid_data[neighbor_idx]
                
                # Detect edges in neighbor
                gray_neighbor = cv.cvtColor(neighbor_cell, cv.COLOR_BGR2GRAY)
                edges_neighbor = cv.Canny(gray_neighbor, 50, 150)
                
                # Compare edge patterns (use normalized hamming distance)
                edge_diff = np.sum(np.abs(edges_skin - edges_neighbor)) / edges_skin.size
                edge_continuity_score += edge_diff
                count += 1
    
    # Return average score or 0 if no valid neighbors
    return edge_continuity_score / max(1, count)

def evaluate_variance_similarity(grid_data, labels, skin_cell, pimple_row, pimple_col, skin_row, skin_col, n_rows, n_cols):
    """
    Evaluate how similar the variance of a skin patch is to the variance of the area around the pimple
    Returns a score where lower is better
    """
    # Calculate variance of skin cell
    skin_var = np.var(skin_cell)
    
    # Calculate average variance of valid surrounding cells
    surrounding_var = 0
    count = 0
    
    for r_offset in range(-1, 2):
        for c_offset in range(-1, 2):
            if r_offset == 0 and c_offset == 0:
                continue  # Skip pimple itself
                
            r = (pimple_row + r_offset) % n_rows
            c = (pimple_col + c_offset) % n_cols
            
            if labels[r, c] == 2:
                neighbor_idx = r * n_cols + c
                neighbor_cell = grid_data[neighbor_idx]
                
                surrounding_var += np.var(neighbor_cell)
                count += 1
    
    if count == 0:
        return 0  # No valid surrounding cells
    
    # Calculate variance difference (normalized)
    surrounding_var /= count
    variance_diff = abs(skin_var - surrounding_var) / max(skin_var, surrounding_var, 1)
    
    return variance_diff

def apply_circular_patch(source_cell, target_cell, radius_factor=0.9, feather_amount=15):
    """
    Apply a circular patch from source cell to target cell with improved blending
    
    Parameters:
    - source_cell: The healthy skin cell to copy from
    - target_cell: The pimple cell to patch
    - radius_factor: Controls the size of the circular patch (0-1)
    - feather_amount: Controls the smoothness of the transition (higher = more feathering)
    
    Returns:
    - Patched cell
    """
    # Create a copy of the target cell
    result = target_cell.copy()
    
    # Get dimensions
    h, w = target_cell.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(w, h) * radius_factor // 2)
    
    # Create a circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv.circle(mask, center, radius, 255, -1)
    
    # Create a feathered mask with stronger blending at the edges
    feathered_mask = cv.GaussianBlur(mask.astype(float), (feather_amount, feather_amount), 0) / 255.0
    
    # Color correction to better match the source with the target surroundings
    # Convert to LAB color space for better color matching
    source_lab = cv.cvtColor(source_cell, cv.COLOR_BGR2Lab)
    target_lab = cv.cvtColor(target_cell, cv.COLOR_BGR2Lab)
    
    # Split LAB channels
    source_l, source_a, source_b = cv.split(source_lab)
    target_l, target_a, target_b = cv.split(target_lab)
    
    # Create inverse mask for the border region only
    border_width = max(3, int(feather_amount / 3))
    border_mask = np.zeros_like(mask)
    cv.circle(border_mask, center, radius, 255, -1)
    cv.circle(border_mask, center, radius - border_width, 0, -1)
    border_mask = border_mask.astype(float) / 255.0
    
    # Get stats of the border region in both source and target
    border_source_l = source_l * border_mask
    border_target_l = target_l * border_mask
    
    # Calculate adjustment needed (brightness only)
    l_sum_source = np.sum(border_source_l)
    l_sum_target = np.sum(border_target_l)
    
    # Calculate multiplier if there are enough border pixels
    if l_sum_source > 0 and np.sum(border_mask) > 0:
        l_ratio = np.mean(target_l[border_mask > 0]) / np.mean(source_l[border_mask > 0])
        l_ratio = max(0.8, min(1.2, l_ratio))  # Limit adjustment to 20%
    else:
        l_ratio = 1.0
    
    # Apply adjustment to the L channel of source
    adjusted_source_l = np.clip(source_l * l_ratio, 0, 255).astype(np.uint8)
    
    # Merge back to LAB
    adjusted_source_lab = cv.merge([adjusted_source_l, source_a, source_b])
    
    # Convert back to BGR
    adjusted_source_bgr = cv.cvtColor(adjusted_source_lab, cv.COLOR_Lab2BGR)
    
    # Create a gradual non-linear transition for smoother blending
    # Enhance edges by applying a non-linear curve to the mask
    nonlinear_mask = np.power(feathered_mask, 1.5)  # Adjust power for different transition effects
    
    # Expand dimensions for proper broadcasting
    nonlinear_mask = np.expand_dims(nonlinear_mask, axis=2)
    
    # Apply the patch with enhanced feathered edges
    result = adjusted_source_bgr * nonlinear_mask + target_cell * (1 - nonlinear_mask)
    
    # Apply a slight blur at the boundary for more natural transition
    final_result = result.copy()
    
    # Create a ring mask just for the border region
    ring_mask = np.zeros((h, w), dtype=np.uint8)
    outer_radius = radius + border_width//2
    inner_radius = radius - border_width//2
    cv.circle(ring_mask, center, outer_radius, 255, -1)
    cv.circle(ring_mask, center, inner_radius, 0, -1)
    
    # Apply a subtle blur ONLY to the border region
    blurred_result = cv.GaussianBlur(result, (3, 3), 0)
    
    # Only use the blurred version at the border
    ring_mask_3ch = np.stack([ring_mask.astype(float) / 255.0] * 3, axis=2)
    final_result = blurred_result * ring_mask_3ch + final_result * (1 - ring_mask_3ch)
    
    return final_result.astype(np.uint8)

def auto_patch_pimples(grid_data, labels, n_rows=15, n_cols=15, search_radius=5, feather_amount=15):
    """
    Automatically patch all pimples using nearby similar skin
    
    Parameters:
    - grid_data: Array of cell image data
    - labels: Array of cell labels (0=no skin, 1=pimple, 2=usable skin)
    - n_rows, n_cols: Grid dimensions
    - search_radius: How far to search for matching skin
    - feather_amount: Controls blending smoothness (higher = more feathering)
    
    Returns:
    - Array of processed cell image data with pimples patched
    - List of (pimple_pos, skin_pos) tuples for visualization
    """
    # Create a copy of the data for the result
    result_data = grid_data.copy()
    
    # Reshape the labels to n_rows x n_cols grid
    grid_labels = labels.reshape(n_rows, n_cols)
    
    # Statistics for reporting
    pimples_found = 0
    pimples_patched = 0
    
    # Store patch information for visualization
    patch_info = []
    
    # Process each cell labeled as pimple (1)
    for row in range(n_rows):
        for col in range(n_cols):
            if grid_labels[row, col] == 1:  # If it's a pimple (label 1)
                pimples_found += 1
                
                # Find the best matching skin cell
                skin_pos = find_nearest_similar_skin(grid_data, grid_labels, row, col, 
                                                   n_rows, n_cols, search_radius)
                
                # If a suitable skin cell was found, apply the patch
                if skin_pos:
                    skin_row, skin_col = skin_pos
                    
                    # Get the pimple and skin cells
                    pimple_idx = row * n_cols + col
                    skin_idx = skin_row * n_cols + skin_col
                    
                    pimple_cell = grid_data[pimple_idx]
                    skin_cell = grid_data[skin_idx]
                    
                    # Apply circular patch with enhanced blending
                    try:
                        patched_cell = apply_circular_patch(skin_cell, pimple_cell, 
                                                           feather_amount=feather_amount)
                        result_data[pimple_idx] = patched_cell
                        pimples_patched += 1
                        
                        # Store the patch information
                        patch_info.append(((row, col), (skin_row, skin_col)))
                    except Exception as e:
                        print(f"Error patching pimple at ({row},{col}): {e}")
                        patch_info.append(((row, col), None))
                else:
                    patch_info.append(((row, col), None))
    
    print(f"Found {pimples_found} pimples, successfully patched {pimples_patched}")
    return result_data, patch_info

def visualize_grid(grid_data, labels, n_rows=15, n_cols=15, cell_size=60, show_matches=False, patch_info=None):
    """
    Create a visualization of a grid with its labels
    
    Parameters:
    - grid_data: Array of cell image data
    - labels: Array of cell labels
    - n_rows, n_cols: Grid dimensions
    - cell_size: Size of each cell in the visualization
    - show_matches: Whether to show lines between matched pimples and skin
    - patch_info: List of (pimple_pos, skin_pos) tuples if show_matches is True
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
            
            # Add grid cell number in the bottom-right corner of each cell
            cell_number = i * n_cols + j + 1  # 1-based numbering
            number_pos = (x2 - cell_size//4, y2 - cell_size//10)
            cv.putText(
                display_img,
                f"#{cell_number}",
                number_pos,
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.3,
                thickness=1,
                color=(0, 0, 0)  # Black text
            )
            
            # Draw grid lines
            cv.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    
    # If showing matches and patch info is provided, draw lines between matched pairs
    if show_matches and patch_info:
        # Create a dictionary to track skin cells used as patches
        skin_use_count = {}
        
        for pimple_pos, skin_pos in patch_info:
            if pimple_pos and skin_pos:
                # Calculate center points
                pimple_row, pimple_col = pimple_pos
                skin_row, skin_col = skin_pos
                
                # Calculate cell numbers for display
                pimple_cell_num = pimple_row * n_cols + pimple_col + 1  # 1-based
                skin_cell_num = skin_row * n_cols + skin_col + 1  # 1-based
                
                # Track how many times this skin cell is used
                if skin_pos in skin_use_count:
                    skin_use_count[skin_pos] += 1
                else:
                    skin_use_count[skin_pos] = 1
                
                pimple_center = (pimple_col * cell_size + cell_size//2, 
                                pimple_row * cell_size + cell_size//2)
                
                skin_center = (skin_col * cell_size + cell_size//2,
                              skin_row * cell_size + cell_size//2)
                
                # Draw a line connecting them
                cv.line(display_img, pimple_center, skin_center, (255, 0, 255), 2)
                
                # Add text to show the patch relationship
                # Put text near the pimple
                mid_x = (pimple_center[0] + skin_center[0]) // 2
                mid_y = (pimple_center[1] + skin_center[1]) // 2
                cv.putText(
                    display_img,
                    f"P{pimple_cell_num}→S{skin_cell_num}",
                    (mid_x - 20, mid_y - 5),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    thickness=1,
                    color=(255, 0, 0)  # Blue text
                )
        
        # Highlight skin cells that are used as patches with count
        for skin_pos, count in skin_use_count.items():
            skin_row, skin_col = skin_pos
            skin_cell_num = skin_row * n_cols + skin_col + 1  # 1-based
            
            # Draw a rectangle around used skin cells
            x1 = skin_col * cell_size
            y1 = skin_row * cell_size
            x2 = (skin_col + 1) * cell_size
            y2 = (skin_row + 1) * cell_size
            
            # Thicker border for skin cells used more times
            thickness = min(count + 1, 3)
            cv.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Add count if used multiple times
            if count > 1:
                cv.putText(
                    display_img,
                    f"Used {count}x",
                    (x1 + 5, y1 + 15),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    thickness=1,
                    color=(0, 0, 255)  # Red text
                )
    
    return display_img

def main():
    # Find all label and data files
    label_files = sorted(glob.glob("lebels*.npz") + glob.glob("labels*.npz"))
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
    
    # Patch settings
    search_radius = 5  # How far to search for matching skin
    feather_amount = 15  # Controls the blending smoothness
    show_patch_lines = True  # Always show patch match lines for better visibility
    
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
        
        # Apply auto-patching with improved blending and get patch info
        processed_data, patch_info = auto_patch_pimples(grid_data, grid_labels, 
                                         search_radius=search_radius,
                                         feather_amount=feather_amount)
        
        # Create visualizations
        original_img = visualize_grid(grid_data, grid_labels, 
                                    show_matches=show_patch_lines, 
                                    patch_info=patch_info)
        
        processed_img = visualize_grid(processed_data, grid_labels)
        
        # Create side-by-side comparison
        comparison = np.hstack((original_img, processed_img))
        
        # Show the comparison
        window_name = f"Dataset {current_dataset_idx+1}/{len(all_datasets)} - Grid {current_grid_idx+1}/{dataset['n_grids']} - Auto Patch"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.imshow(window_name, comparison)
        
        # Add text labels to indicate which is which
        h, w = original_img.shape[:2]
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(comparison, "Original with Patch Sources", (w//8, 30), font, 0.8, (0, 0, 255), 2)
        cv.putText(comparison, f"Auto-Patched (radius={search_radius}, feather={feather_amount})", 
                 (w + w//8, 30), font, 0.7, (0, 0, 255), 2)
        
        # Instructions
        print("\nViewing dataset", current_dataset_idx + 1, "of", len(all_datasets))
        print("Viewing grid", current_grid_idx + 1, "of", dataset['n_grids'])
        print(f"Current settings: search radius={search_radius}, feather amount={feather_amount}")
        print("Controls:")
        print("  'n' or Right arrow - Next grid")
        print("  'p' or Left arrow - Previous grid")
        print("  's' - Save current before/after comparison")
        print("  'j' or Down arrow - Next dataset")
        print("  'k' or Up arrow - Previous dataset")
        print("  '+' - Increase search radius")
        print("  '-' - Decrease search radius")
        print("  'f' - Increase feather amount (smoother transitions)")
        print("  'd' - Decrease feather amount (sharper transitions)")
        print("  'l' - Toggle patch match lines")
        print("  'q' or ESC - Quit")
        
        # Wait for key press
        key = cv.waitKey(0) & 0xFF
        
        # Handle key press for navigation
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('s'):  # Save image
            save_name = f"pimple_removal_patch_dataset{current_dataset_idx+1}_grid{current_grid_idx+1}.png"
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
        elif key == ord('+'):  # Increase search radius
            search_radius = min(10, search_radius + 1)
            print(f"Increased search radius to {search_radius}")
        elif key == ord('-'):  # Decrease search radius
            search_radius = max(1, search_radius - 1)
            print(f"Decreased search radius to {search_radius}")
        elif key == ord('f'):  # Increase feather amount
            feather_amount = min(31, feather_amount + 2)
            print(f"Increased feather amount to {feather_amount} (smoother transitions)")
        elif key == ord('d'):  # Decrease feather amount
            feather_amount = max(3, feather_amount - 2)
            print(f"Decreased feather amount to {feather_amount} (sharper transitions)")
        elif key == ord('l'):  # Toggle patch lines
            show_patch_lines = not show_patch_lines
            print(f"Patch match lines: {'ON' if show_patch_lines else 'OFF'}")
        
        # Close the current window before showing the next
        cv.destroyWindow(window_name)
    
    cv.destroyAllWindows()
    print("Auto patch pimple removal visualization complete.")

if __name__ == "__main__":
    main() 