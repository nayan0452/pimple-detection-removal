import cv2 as cv
import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
from scipy import ndimage

class BrushPimpleRemover:
    def __init__(self):
        self.image = None
        self.original = None
        self.mask = None
        self.brush_size = 10
        self.healing_mode = False  # Kept for backward compatibility
        self.patch_mode = True     # Patch mode is now the default and only mode
        self.source_point = None
        self.clone_mode = cv.NORMAL_CLONE
        self.drawing = False     # General drawing state flag
        self.window_name = "Patch Tool Pimple Remover"  # Updated window name
        self.last_x, self.last_y = -1, -1
        self.result_saved = False
        self.show_help_overlay = True  # Start with help visible
        self.status_message = "Welcome! Draw around a pimple, then drag to sample healthy skin"
        
        # Image navigation attributes
        self.image_files = []
        self.current_image_index = 0
        self.edited_images = {}  # Store edited versions of images
        self.edited_masks = {}   # Store masks for each image
        
        # Patch tool attributes
        self.patch_selection = None  # The patch selection contour
        self.patch_mask = None  # Mask for the patch selection
        self.patch_rect = None  # Bounding rectangle of the patch selection
        self.patch_dragging = False  # Whether the patch is being dragged
        self.patch_start_pos = None  # Starting position of the patch drag
        self.patch_current_pos = None  # Current position of the patch drag
        self.patch_selection_ready = False  # Whether a patch selection has been made
        self.patch_selection_points = []  # Points for the patch selection
        
        # Comparison view flag
        self.show_comparison = False  # Whether to show before/after comparison view
        self.comparison_window_name = "Before and After Comparison"
        
    def load_image_directory(self, path):
        """Load all images from a directory"""
        if os.path.isdir(path):
            # Find all image files in the directory
            extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
            self.image_files = []
            for ext in extensions:
                self.image_files.extend(glob.glob(os.path.join(path, f"*.{ext}")))
                self.image_files.extend(glob.glob(os.path.join(path, f"*.{ext.upper()}")))
            
            # Sort the files for consistent navigation
            self.image_files.sort()
            
            if not self.image_files:
                print(f"No image files found in {path}")
                return False
                
            print(f"Found {len(self.image_files)} images in directory")
            self.current_image_index = 0
            return self.load_current_image()
        elif os.path.isfile(path):
            # If a single file is provided, treat its directory as the source
            directory = os.path.dirname(path)
            if not directory:
                directory = "."
            filename = os.path.basename(path)
            
            # Load all images from the same directory
            extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
            self.image_files = []
            for ext in extensions:
                self.image_files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
                self.image_files.extend(glob.glob(os.path.join(directory, f"*.{ext.upper()}")))
            
            # Sort the files for consistent navigation
            self.image_files.sort()
            
            if not self.image_files:
                # If no other images found, just use the one provided
                self.image_files = [path]
                self.current_image_index = 0
            else:
                # Find the index of the provided file
                try:
                    self.current_image_index = self.image_files.index(path)
                except ValueError:
                    # If not in list (path format difference), just use the first image
                    self.current_image_index = 0
            
            # Print debug info
            print(f"Working with {len(self.image_files)} images from directory {directory}")
            print(f"Starting at image index {self.current_image_index}: {os.path.basename(self.image_files[self.current_image_index])}")
            
            return self.load_current_image()
        else:
            print(f"Invalid path: {path}")
            return False
    
    def load_current_image(self):
        """Load the current image based on the index"""
        if not self.image_files:
            print("No image files available")
            return False
            
        # Make sure the index is valid
        if self.current_image_index < 0 or self.current_image_index >= len(self.image_files):
            print(f"Invalid image index {self.current_image_index}, resetting to 0")
            self.current_image_index = 0
            
        image_path = self.image_files[self.current_image_index]
        print(f"Loading image {self.current_image_index + 1}/{len(self.image_files)}: {os.path.basename(image_path)}")
        
        # Save current image state if we're switching from an image
        self.save_current_state()
        
        # Reset healing and patch tool attributes
        self.source_point = None
        self.source_preview = None
        self.heal_preview_active = False
        self.patch_selection = None
        self.patch_mask = None
        self.patch_rect = None
        self.patch_dragging = False
        self.patch_start_pos = None
        self.patch_current_pos = None
        self.patch_selection_ready = False
        self.patch_selection_points = []
        
        # Check if we have already edited this image
        if image_path in self.edited_images:
            print(f"Loading previously edited version of {os.path.basename(image_path)}")
            # Restore the edited version and mask
            self.original = cv.imread(image_path)
            if self.original is None:
                print(f"Error: Could not load image from {image_path}")
                return False
                
            # Resize if too large
            max_size = 1200
            h, w = self.original.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                self.original = cv.resize(self.original, (int(w * scale), int(h * scale)))
                
            self.image = self.edited_images[image_path].copy()
            self.mask = self.edited_masks[image_path].copy()
        else:
            print(f"Loading fresh image: {os.path.basename(image_path)}")
            # Load as a new image
            self.original = cv.imread(image_path)
            if self.original is None:
                print(f"Error: Could not load image from {image_path}")
                return False
                
            # Resize if too large
            max_size = 1200
            h, w = self.original.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                self.original = cv.resize(self.original, (int(w * scale), int(h * scale)))
                
            self.image = self.original.copy()
            h, w = self.image.shape[:2]
            self.mask = np.zeros((h, w), dtype=np.uint8)
            
        self.display_image = self.image.copy()
        self.status_message = f"Loaded image {os.path.basename(image_path)} ({self.current_image_index + 1}/{len(self.image_files)})"
        return True
    
    def save_current_state(self):
        """Save the current image state and mask"""
        if not self.image_files or not 0 <= self.current_image_index < len(self.image_files):
            print("No valid image to save state for")
            return
            
        current_path = self.image_files[self.current_image_index]
        if self.image is not None and self.mask is not None:
            # Only save if modifications have been made
            if not np.array_equal(self.image, self.original):
                print(f"Saving edited state for {os.path.basename(current_path)}")
                self.edited_images[current_path] = self.image.copy()
                self.edited_masks[current_path] = self.mask.copy()
            else:
                print(f"No edits to save for {os.path.basename(current_path)}")
    
    def next_image(self):
        """Navigate to the next image"""
        if not self.image_files:
            print("No image files available for navigation")
            return False
            
        # Calculate the next index with wraparound
        next_index = (self.current_image_index + 1) % len(self.image_files)
        print(f"Navigating from image {self.current_image_index + 1} to {next_index + 1}")
        
        # Update the current index and load the image
        self.current_image_index = next_index
        return self.load_current_image()
    
    def prev_image(self):
        """Navigate to the previous image"""
        if not self.image_files:
            print("No image files available for navigation")
            return False
            
        # Calculate the previous index with wraparound
        prev_index = (self.current_image_index - 1) % len(self.image_files)
        print(f"Navigating from image {self.current_image_index + 1} to {prev_index + 1}")
        
        # Update the current index and load the image
        self.current_image_index = prev_index
        return self.load_current_image()
    
    def load_image(self, image_path):
        """Load an image for editing - superseded by load_image_directory"""
        return self.load_image_directory(image_path)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for patch tool interaction"""
        # Directly use patch tool events for all interactions
        self.handle_patch_tool_events(event, x, y, flags)
    
    def handle_patch_tool_events(self, event, x, y, flags):
        """Handle mouse events for the patch tool"""
        # Don't process events outside the image boundaries
        if x < 0 or y < 0 or x >= self.image.shape[1] or y >= self.image.shape[0]:
            return
            
        if event == cv.EVENT_LBUTTONDOWN:
            if not self.patch_selection_ready:
                # Check if we are starting a brand new selection (no points yet)
                if not self.patch_selection_points:
                    # Start a new selection
                    self.patch_selection_points = []
                    self.patch_selection_points.append((x, y))
                    self.patch_selection = None  # Initialize as None until selection is complete
                    self.drawing = True  # Start drawing mode
                    self.status_message = "Draw around the blemish area - release to complete selection"
                else:
                    # The user is clicking to manually add points (click-by-click mode)
                    # Add the point if it's not too close to the last one
                    last_x, last_y = self.patch_selection_points[-1]
                    dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                    if dist > 2:  # Even lower threshold for more precision
                        self.patch_selection_points.append((x, y))
                        # Check if we're closing the selection by clicking near the start
                        start_x, start_y = self.patch_selection_points[0]
                        dist_to_start = np.sqrt((x - start_x)**2 + (y - start_y)**2)
                        
                        if dist_to_start < 10 and len(self.patch_selection_points) > 2:
                            # They're clicking near the start point - complete the selection
                            self.complete_selection()
                        else:
                            # Calculate total perimeter to guide the user
                            perimeter = 0
                            for i in range(1, len(self.patch_selection_points)):
                                p1 = self.patch_selection_points[i-1]
                                p2 = self.patch_selection_points[i]
                                perimeter += np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                            
                            self.status_message = f"Added point {len(self.patch_selection_points)}. Continue adding points or double-click to complete."
            
            elif self.patch_selection_ready and not self.patch_dragging:
                # Start dragging the selection if clicking within the selection area
                if self.patch_mask is not None and self.patch_mask[y, x] > 0:
                    self.patch_dragging = True
                    self.patch_start_pos = (x, y)
                    self.patch_current_pos = (x, y)
                    self.status_message = "Drag to position over healthy skin, then release"
                else:
                    # If clicking outside the selection, start a new selection
                    self.patch_selection_points = []
                    self.patch_selection_points.append((x, y))
                    self.patch_selection = None  # Initialize as None
                    self.patch_selection_ready = False
                    self.drawing = True  # Start drawing mode
                    self.status_message = "Drawing new selection"
        
        elif event == cv.EVENT_LBUTTONDBLCLK:
            # Double-click to complete selection
            if len(self.patch_selection_points) > 2 and not self.patch_selection_ready:
                self.complete_selection()
                
        elif event == cv.EVENT_MOUSEMOVE:
            # Always track the current mouse position for preview
            self.last_x, self.last_y = x, y
            
            if len(self.patch_selection_points) > 0 and not self.patch_selection_ready and self.drawing:
                # Continue drawing the selection if in drawing mode
                # Add point if it's far enough from the last one - but not too often
                last_x, last_y = self.patch_selection_points[-1]
                dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                
                # Use a variable threshold based on speed - add points more frequently when moving slowly
                # This allows for both quick rough selections and fine-tuned precise movements
                min_dist = 5  # Base minimum distance
                
                if dist > min_dist:
                    self.patch_selection_points.append((x, y))
                    
                    # Check if we have enough points to show instruction about closing the selection
                    if len(self.patch_selection_points) >= 3:
                        # Get distance to start point for visualization
                        start_x, start_y = self.patch_selection_points[0]
                        dist_to_start = np.sqrt((x - start_x)**2 + (y - start_y)**2)
                        if dist_to_start < 15:
                            self.status_message = f"Release to close the selection ({len(self.patch_selection_points)} points)"
                        else:
                            self.status_message = f"Drawing selection - {len(self.patch_selection_points)} points"
                    else:
                        self.status_message = f"Drawing selection - {len(self.patch_selection_points)} points. Need at least 3."
            
            elif self.patch_dragging:
                # Update current drag position
                self.patch_current_pos = (x, y)
                
                # Calculate offset for preview
                dx = self.patch_current_pos[0] - self.patch_start_pos[0]
                dy = self.patch_current_pos[1] - self.patch_start_pos[1]
                
                # Update status with offset information
                self.status_message = f"Dragging to source - Offset: ({dx}, {dy})"
            
            # Always update display to show preview
            self.update_display()
        
        elif event == cv.EVENT_LBUTTONUP:
            # Stop drawing mode
            if self.drawing:
                self.drawing = False
                
            if len(self.patch_selection_points) > 0 and not self.patch_selection_ready:
                # Complete the selection if we were in drawing mode and have enough points
                if len(self.patch_selection_points) > 2:
                    # Check if the end point is close to the start point to auto-close
                    start_x, start_y = self.patch_selection_points[0]
                    dist_to_start = np.sqrt((x - start_x)**2 + (y - start_y)**2)
                    
                    # If end point is close to start, automatically complete the selection
                    if dist_to_start < 15:
                        self.complete_selection()
                    else:
                        # If we have enough points, ask the user to close the selection
                        self.status_message = f"Selection in progress ({len(self.patch_selection_points)} points). Double-click to complete or click near start point."
                else:
                    # Not enough points yet, but keep the existing points
                    # This allows point-by-point construction of the selection
                    remaining = 3 - len(self.patch_selection_points)
                    self.status_message = f"Need {remaining} more point(s). Click to add points."
            
            elif self.patch_dragging:
                # Finalize the patch operation
                self.patch_dragging = False
                
                # Calculate the offset between start and end positions
                dx = self.patch_current_pos[0] - self.patch_start_pos[0]
                dy = self.patch_current_pos[1] - self.patch_start_pos[1]
                
                # Only apply if there was a meaningful drag
                if abs(dx) > 5 or abs(dy) > 5:
                    # Apply the patch with this offset
                    self.apply_patch(dx, dy)
                    self.status_message = "Patch applied successfully"
                else:
                    self.status_message = "Patch cancelled - drag distance too small"
                
                # Keep the selection active but stop dragging
                self.patch_dragging = False
        
        # Update the display to show the current state
        self.update_display()
    
    def complete_selection(self):
        """Complete the patch selection and create the mask"""
        # Make sure we have enough points
        if len(self.patch_selection_points) <= 2:
            self.status_message = "Need at least 3 points for a valid selection"
            return
            
        # Close the contour by adding the first point again
        if self.patch_selection_points[0] != self.patch_selection_points[-1]:
            self.patch_selection_points.append(self.patch_selection_points[0])
        
        # Create a mask from the selection
        h, w = self.image.shape[:2]
        self.patch_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert points to numpy array for drawing
        points = np.array(self.patch_selection_points, dtype=np.int32)
        
        # Draw filled contour
        cv.fillPoly(self.patch_mask, [points], 255)
        
        # Get the bounding rectangle for the selection
        x, y, w, h = cv.boundingRect(points)
        self.patch_rect = (x, y, w, h)
        
        # Store the selection contour for visualization
        self.patch_selection = np.array([points], dtype=np.int32)
        
        # Mark selection as ready
        self.patch_selection_ready = True
        self.status_message = "Selection complete. Click and drag to position over healthy skin"
    
    def apply_patch(self, dx, dy):
        """Apply the patch from source to destination area"""
        if self.patch_mask is None or self.patch_rect is None:
            self.status_message = "No valid patch selection to apply"
            return
            
        # Get the patch rectangle (destination area with blemish)
        x, y, w, h = self.patch_rect
        
        # Define source rectangle (healthy skin we're sampling from)
        src_x = x + dx
        src_y = y + dy
        
        # Make sure source rectangle is within image bounds
        if (src_x < 0 or src_y < 0 or 
            src_x + w > self.image.shape[1] or 
            src_y + h > self.image.shape[0]):
            self.status_message = "Source area outside image bounds - patch cancelled"
            return
            
        # Copy the source region (healthy skin)
        source_region = self.image[src_y:src_y+h, src_x:src_x+w].copy()
        
        # Get the destination area (blemish) and its mask
        dest_region = self.image[y:y+h, x:x+w].copy()
        dest_mask = self.patch_mask[y:y+h, x:x+w].copy()
        
        # Create a mask with a border feather for smoother blending
        feather_mask = cv.GaussianBlur(dest_mask.astype(float), (21, 21), 0)
        feather_mask = np.clip(feather_mask, 0, 255).astype(np.uint8)
        
        # Try to use seamless cloning if the regions are large enough
        if w > 10 and h > 10 and np.sum(dest_mask) > 100:
            try:
                # Create center point for the clone relative to the destination image
                center = (w // 2, h // 2)
                
                # Apply seamless cloning - source_region is what we're copying FROM (healthy skin)
                # dest_region is where we're applying TO (blemish area)
                result = cv.seamlessClone(
                    source_region, dest_region, feather_mask, center, self.clone_mode)
                
                # Copy the result back to the full image
                self.image[y:y+h, x:x+w] = result
                self.status_message = "Patch applied with seamless blending"
            except Exception as e:
                # Fallback to alpha blending if seamless clone fails
                print(f"Seamless cloning failed: {e}")
                self.apply_alpha_blend_patch(x, y, w, h, src_x, src_y)
        else:
            # For small regions, just use alpha blending
            self.apply_alpha_blend_patch(x, y, w, h, src_x, src_y)
    
    def apply_alpha_blend_patch(self, x, y, w, h, src_x, src_y):
        """Apply the patch using alpha blending for small regions"""
        # Get the destination area and mask
        dest_region = self.image[y:y+h, x:x+w].copy()
        dest_mask = self.patch_mask[y:y+h, x:x+w].astype(float) / 255.0
        
        # Create a gradient blend factor at the edges
        blend_mask = cv.GaussianBlur(dest_mask, (21, 21), 0)
        
        # Expand dimensions for proper broadcasting
        blend_mask = np.expand_dims(blend_mask, axis=2)
        
        # Copy the source region
        source_region = self.image[src_y:src_y+h, src_x:src_x+w].copy()
        
        # Blend the source and destination
        blended = source_region * blend_mask + dest_region * (1 - blend_mask)
        
        # Copy the result back to the full image
        self.image[y:y+h, x:x+w] = blended.astype(np.uint8)
        self.status_message = "Patch applied with alpha blending"
    
    def update_display(self):
        """Update the display image with current mask and image"""
        # Create a copy of the image for display
        self.display_image = self.image.copy()
        
        # In patch tool mode
        # Draw the patch selection points
        if self.patch_selection_points and not self.patch_selection_ready:
            # If we're still drawing the selection
            pts = np.array(self.patch_selection_points, dtype=np.int32)
            cv.polylines(self.display_image, [pts], False, (0, 255, 0), 2)
            
            # Draw the points
            for pt in self.patch_selection_points:
                cv.circle(self.display_image, pt, 3, (0, 0, 255), -1)
            
            # If we have at least 3 points, show a preview line to the starting point
            if len(self.patch_selection_points) >= 3 and self.last_x >= 0 and self.last_y >= 0:
                # Draw a dashed line from current mouse position to the first point
                start_pt = self.patch_selection_points[0]
                current_pt = (self.last_x, self.last_y)
                
                # Calculate distance to start point to determine color
                dist_to_start = np.sqrt((current_pt[0] - start_pt[0])**2 + (current_pt[1] - start_pt[1])**2)
                
                # Use yellow if close to start point (can close selection), otherwise use light blue
                line_color = (0, 255, 255) if dist_to_start < 20 else (255, 255, 0)
                
                # Draw dashed preview line
                pt1 = current_pt
                pt2 = start_pt
                dist = np.sqrt(np.sum((np.array(pt1) - np.array(pt2))**2))
                if dist > 0:
                    unit_vec = (np.array(pt2) - np.array(pt1)) / dist
                    dash_length = 5
                    num_segments = int(dist / (2 * dash_length))
                    for j in range(num_segments):
                        start = np.array(pt1) + j * 2 * dash_length * unit_vec
                        end = start + dash_length * unit_vec
                        start = tuple(map(int, start))
                        end = tuple(map(int, end))
                        cv.line(self.display_image, start, end, line_color, 1)
                        
                # If close enough to close, show a hint circle around the start point
                if dist_to_start < 20:
                    cv.circle(self.display_image, start_pt, 10, (0, 255, 255), 1)
        
        # Draw the completed patch selection
        if self.patch_selection_ready and self.patch_selection is not None:
            # Draw the selection contour
            cv.drawContours(self.display_image, [self.patch_selection], 0, (0, 255, 0), 2)
            
            # If dragging, show the destination position
            if self.patch_dragging and self.patch_start_pos and self.patch_current_pos:
                # Calculate offset
                dx = self.patch_current_pos[0] - self.patch_start_pos[0]
                dy = self.patch_current_pos[1] - self.patch_start_pos[1]
                
                # Draw the shifted contour
                shifted_contour = self.patch_selection.copy()
                shifted_contour[0,:,0] += dx
                shifted_contour[0,:,1] += dy
                
                # Draw the shifted contour with dashed lines
                # Need to manually draw dashed lines since cv.LINE_DASH doesn't exist
                points = shifted_contour[0]
                for i in range(len(points) - 1):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i+1])
                    # Draw dashed line by skipping every 5 pixels
                    dist = np.sqrt(np.sum((np.array(pt1) - np.array(pt2))**2))
                    if dist > 0:
                        unit_vec = (np.array(pt2) - np.array(pt1)) / dist
                        dash_length = 5
                        num_segments = int(dist / (2 * dash_length))
                        for j in range(num_segments):
                            start = np.array(pt1) + j * 2 * dash_length * unit_vec
                            end = start + dash_length * unit_vec
                            start = tuple(map(int, start))
                            end = tuple(map(int, end))
                            cv.line(self.display_image, start, end, (255, 255, 0), 2)
                
                # Draw an arrow indicating the drag
                cv.arrowedLine(self.display_image, self.patch_start_pos, self.patch_current_pos, 
                             (0, 255, 255), 2, tipLength=0.2)
        
        # Simplified UI showing only patch mode info
        mode_text = "Mode: PATCH TOOL"
        clone_text = "Clone: MIXED" if self.clone_mode == cv.MIXED_CLONE else "Clone: NORMAL"
        
        # Add image navigation info if we have multiple images
        if len(self.image_files) > 1:
            image_name = os.path.basename(self.image_files[self.current_image_index])
            nav_text = f"Image: {self.current_image_index+1}/{len(self.image_files)} - {image_name}"
            info_text = f"{mode_text} | Brush: {self.brush_size} | {clone_text} | {nav_text}"
        else:
            info_text = f"{mode_text} | Brush Size: {self.brush_size} | {clone_text}"
        
        # Add status text at bottom
        h, w = self.display_image.shape[:2]
        
        # Add background for text
        cv.rectangle(self.display_image, (0, 0), (w, 30), (0, 0, 0), -1)
        cv.putText(self.display_image, info_text, (10, 20), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        
        # Add status message at bottom
        cv.rectangle(self.display_image, (0, h-30), (w, h), (0, 0, 0), -1)
        cv.putText(self.display_image, self.status_message, (10, h-10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        
        # Add navigation buttons if we have multiple images
        if len(self.image_files) > 1:
            # Previous button (left)
            cv.rectangle(self.display_image, (10, h//2-25), (50, h//2+25), (40, 40, 40), -1)
            cv.putText(self.display_image, "<", (20, h//2+10), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv.LINE_AA)
            
            # Next button (right)
            cv.rectangle(self.display_image, (w-50, h//2-25), (w-10, h//2+25), (40, 40, 40), -1)
            cv.putText(self.display_image, ">", (w-40, h//2+10), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv.LINE_AA)
        
        # Add help overlay if enabled
        if self.show_help_overlay:
            self.draw_help_overlay()
        
        # Display the current image with overlays
        cv.imshow(self.window_name, self.display_image)
        
        # Update the comparison view if enabled
        if self.show_comparison:
            self.update_comparison_view()
    
    def draw_help_overlay(self):
        """Draw help information directly on the image"""
        h, w = self.display_image.shape[:2]
        
        # Create semi-transparent overlay
        overlay = self.display_image.copy()
        cv.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        
        # Help text - simplified for patch tool only
        help_lines = [
            "PATCH TOOL PIMPLE REMOVER - CONTROLS",
            "",
            "MOUSE:",
            "• Draw closed selection: Click and drag around a pimple",
            "• Double-click: Complete selection",
            "• Click & drag inside selection: Sample healthy skin",
            "• Click < and > buttons: Navigate between images",
            "",
            "KEYBOARD:",
            "• +/=: Increase brush size",
            "• -: Decrease brush size",
            "• c: Toggle clone mode (Normal/Mixed)",
            "• n: Next image",
            "• p: Previous image",
            "• r: Reset to original image",
            "• s: Save result",
            "• a: Save all edited images",
            "• v: Toggle before/after comparison view",
            "• q: Quit",
            "• ?: Toggle this help display",
            "",
            "PATCH TOOL USAGE:",
            "1. Click around the blemish to create a selection (minimum 3 points)",
            "2. Either drag continuously, or click point-by-point to build selection",
            "3. Click near the starting point to close the selection",
            "4. Click inside your selection to start dragging",
            "5. Drag to a nearby area with healthy skin texture",
            "6. Release to apply the patch with seamless blending",
            "",
            "Press any key to close this help"
        ]
        
        # Calculate text positions
        text_y = h // 2 - len(help_lines) * 12
        
        # Add background for better readability
        alpha = 0.7
        self.display_image = cv.addWeighted(overlay, alpha, self.display_image, 1 - alpha, 0)
        
        # Add text
        for i, line in enumerate(help_lines):
            y = text_y + i * 24
            if i == 0:  # Title
                cv.putText(self.display_image, line, (w//2 - 180, y), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            elif "PATCH TOOL USAGE:" in line:  # Section header
                cv.putText(self.display_image, line, (w//4, y), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv.LINE_AA)
            else:
                cv.putText(self.display_image, line, (w//4, y), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    def toggle_clone_mode(self):
        """Toggle between normal and mixed clone modes"""
        if self.clone_mode == cv.NORMAL_CLONE:
            self.clone_mode = cv.MIXED_CLONE
            self.status_message = "Switched to Mixed Clone mode (better color matching)"
        else:
            self.clone_mode = cv.NORMAL_CLONE
            self.status_message = "Switched to Normal Clone mode (better texture preservation)"
        self.update_display()
        
    def adjust_brush_size(self, change):
        """Increase or decrease brush size"""
        self.brush_size = max(1, min(50, self.brush_size + change))
        self.status_message = f"Brush size adjusted to: {self.brush_size}"
        self.update_display()
        
    def save_result(self, filename=None):
        """Save the current result to a file"""
        if filename is None:
            # Get current image path
            if self.image_files and 0 <= self.current_image_index < len(self.image_files):
                base_path = self.image_files[self.current_image_index]
                basename = os.path.basename(base_path)
                name, ext = os.path.splitext(basename)
                filename = f"healed_{name}{ext}"
            else:
                timestamp = cv.getTickCount()
                filename = f"healed_image_{timestamp}.jpg"
            
        cv.imwrite(filename, self.image)
        self.status_message = f"Result saved to {filename}"
        self.result_saved = True
        self.update_display()
    
    def save_all_edited(self):
        """Save all edited images"""
        if not self.edited_images:
            self.status_message = "No edited images to save"
            self.update_display()
            return
            
        # Create output directory
        output_dir = "healed_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the current state first
        self.save_current_state()
        
        # Save all edited images
        saved_count = 0
        for img_path, img_data in self.edited_images.items():
            basename = os.path.basename(img_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(output_dir, f"healed_{name}{ext}")
            cv.imwrite(output_path, img_data)
            saved_count += 1
            
        self.status_message = f"Saved {saved_count} edited images to {output_dir}/ directory"
        self.result_saved = True
        self.update_display()
        
    def reset_to_original(self):
        """Reset to the original image"""
        self.image = self.original.copy()
        self.mask = np.zeros_like(self.mask)
        self.status_message = "Reset to original image"
        self.update_display()
        
    def toggle_help_overlay(self):
        """Toggle the help overlay on/off"""
        self.show_help_overlay = not self.show_help_overlay
        self.update_display()
        
    def check_button_click(self, x, y):
        """Check if a navigation button was clicked"""
        h, w = self.display_image.shape[:2]
        
        # Check if we have multiple images
        if len(self.image_files) <= 1:
            return False
            
        # Previous button (left)
        if 10 <= x <= 50 and h//2-25 <= y <= h//2+25:
            print("Previous button clicked")
            return self.prev_image()
            
        # Next button (right)
        if w-50 <= x <= w-10 and h//2-25 <= y <= h//2+25:
            print("Next button clicked")
            return self.next_image()
            
        return False
        
    def update_comparison_view(self):
        """Create and display a side-by-side before and after comparison"""
        if self.original is None or self.image is None:
            return
        
        # Get dimensions
        h, w = self.original.shape[:2]
        
        # Create copies for display
        before_img = self.original.copy()
        after_img = self.image.copy()
        
        # Add labels at the top of each image
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(before_img, "BEFORE", (w//2 - 60, 30), font, 0.8, (0, 0, 255), 2)
        cv.putText(after_img, "AFTER", (w//2 - 50, 30), font, 0.8, (0, 255, 0), 2)
        
        # Create side-by-side comparison
        comparison = np.hstack((before_img, after_img))
        
        # Display the comparison
        cv.namedWindow(self.comparison_window_name, cv.WINDOW_NORMAL)
        cv.imshow(self.comparison_window_name, comparison)
    
    def toggle_comparison_view(self):
        """Toggle the before/after comparison view"""
        self.show_comparison = not self.show_comparison
        
        if self.show_comparison:
            self.status_message = "Before/After comparison view enabled"
            self.update_comparison_view()
        else:
            self.status_message = "Before/After comparison view disabled"
            cv.destroyWindow(self.comparison_window_name)
        
        self.update_display()

    def run(self, image_path):
        """Main function to run the application"""
        if not self.load_image(image_path):
            return False
        
        # Create window and set callback
        cv.namedWindow(self.window_name)
        
        # Create a custom mouse callback that checks for button clicks
        def custom_mouse_callback(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                # Check if a navigation button was clicked
                if self.check_button_click(x, y):
                    return
                
            # If not a button click, use the regular mouse callback
            self.mouse_callback(event, x, y, flags, param)
            
        # Set the custom callback
        cv.setMouseCallback(self.window_name, custom_mouse_callback)
        
        # Show initial image with help overlay
        self.update_display()
        
        # Main loop - simplified for patch mode only
        while True:
            key = cv.waitKey(50) & 0xFF  # Use a shorter wait time (50ms) for more responsive UI
            
            if key == ord('q'):
                # Ask to save if changes were made and not saved
                if not self.result_saved and not np.array_equal(self.image, self.original):
                    self.status_message = "You have unsaved changes. Press 's' to save or 'q' again to quit."
                    self.update_display()
                    if cv.waitKey(0) & 0xFF == ord('s'):
                        self.save_result()
                break
                
            elif key == ord('+') or key == ord('='):
                self.adjust_brush_size(1)
                
            elif key == ord('-'):
                self.adjust_brush_size(-1)
                
            elif key == ord('c'):
                self.toggle_clone_mode()
                
            elif key == ord('r'):
                self.reset_to_original()
                
            elif key == ord('s'):
                self.save_result()
                
            elif key == ord('a'):
                self.save_all_edited()
                
            elif key == ord('n'):
                self.next_image()
                
            elif key == ord('p'):  # Use the 'p' key for previous image
                self.prev_image()
                
            elif key == ord('?'):
                self.toggle_help_overlay()
                
            elif key == ord('v'):  # Use 'v' key to toggle comparison view
                self.toggle_comparison_view()
                
            # If any key is pressed and help is visible, hide it
            elif self.show_help_overlay and key != 255:  # 255 is no key pressed
                self.show_help_overlay = False
                self.update_display()
        
        # Clean up all windows when closing
        cv.destroyAllWindows()
        return True

def main():
    # Check if an image path was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Please provide the path to an image or directory as a command-line argument.")
        print("Usage: python brush_pimple_remover.py <image_path_or_directory>")
        return
    
    # Create and run the application
    app = BrushPimpleRemover()
    app.run(image_path)

if __name__ == "__main__":
    main() 