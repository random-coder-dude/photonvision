import cv2
import numpy as np
import time

# ========================
# CONFIGURATION CONSTANTS
# ========================

# Color Detection Thresholds
RED_FACTOR = 2.0  # Red must be this many times larger than blue and green
BLUE_FACTOR = 1.7  # Blue must be this many times larger than red and green

# Bumper Detection Parameters
MIN_BUMPER_AREA = 1000  # Minimum area in pixels of a bumper
MIN_ASPECT_RATIO = 1.5  # Minimum width:height of a bumper
MAX_ASPECT_RATIO = 4.0  # Maximum width:height of a bumper

# Morphology Parameters
MORPH_KERNEL_SIZE = 7  # Gaussian Blur (Ask me)
MORPH_KERNEL_SHAPE = cv2.MORPH_ELLIPSE  # (Ask me)
MORPH_OPERATION = cv2.MORPH_CLOSE  # (Ask me )
MORPH_ITERATIONS = 1  # How many times to run the blur

# Metallic Detection
METALLIC_THRESHOLD = 0.25  # Minimum metallic score
METALLIC_SPREAD_WEIGHT = 0.7  # How much rgb effects the metallic value
COLOR_CHANNEL_WEIGHT = 0.3333333  # 1/3

# Robot Detection
ROBOT_SEARCH_HEIGHT_MULTIPLIER = 1.0  # Search region is bumper_height * this value above bumper

# Display Configuration
DEBUG_SHOW_OVERLAY = True  # Show FPS counter
DEBUG_VERBOSE = True  # Values printed on screen

# Performance Tracking
FPS_SMOOTHING_FACTOR = 0.2  # Weight for new FPS value
MAX_FRAMES_TRACKING = 1000000  # Don't worry about it

# Drawing Parameters
BBOX_THICKNESS = 2  # Line weight
ROBOT_BOX_COLOR = (0, 255, 0)  # BGR: Green
RED_BUMPER_COLOR = (0, 0, 255)  # BGR: Red
BLUE_BUMPER_COLOR = (255, 0, 0)  # BGR: Blue
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX  # Self explanatory
TEXT_SCALE_LARGE = 0.8  # Self explanatory
TEXT_SCALE_MEDIUM = 0.6  # Self explanatory
TEXT_SCALE_SMALL = 0.5  # Self explanatory
TEXT_THICKNESS_BOLD = 2  # Self explanatory
TEXT_THICKNESS_NORMAL = 1  # Self explanatory
TEXT_COLOR_FPS = (0, 255, 0)  # Self explanatory
TEXT_COLOR_TIMING = (255, 255, 0)  # Self explanatory
TEXT_COLOR_DETAILS = (200, 200, 200)  # Self explanatory

# Overlay Layout
OVERLAY_START_Y = 30  # Display configs
OVERLAY_LINE_SPACING_LARGE = 25  # Display configs
OVERLAY_LINE_SPACING_MEDIUM = 20  # Display configs
OVERLAY_LINE_SPACING_SMALL = 18  # Display configs
OVERLAY_X_OFFSET = 10  # Display configs
OVERLAY_TEXT_Y_OFFSET = 5  # Display configs

# Connected Components
CC_CONNECTIVITY = 8  # How close 2 objects need to be for merging


# ========================
# BumperDetector Class
# ========================
class BumperDetector:
    def __init__(self, red_factor=RED_FACTOR, blue_factor=BLUE_FACTOR, min_area=MIN_BUMPER_AREA):
        self.red_factor = red_factor
        self.blue_factor = blue_factor
        self.min_area = min_area
        self.morph_kernel = cv2.getStructuringElement(MORPH_KERNEL_SHAPE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

        # Pre-allocate for statistics
        self.frame_count = 0
        self.max_frames = MAX_FRAMES_TRACKING

        # Performance tracking with pre-allocated arrays
        self.timing_history = {
            "color_masks": np.zeros(self.max_frames, dtype=np.float32),
            "metallic": np.zeros(self.max_frames, dtype=np.float32),
            "morphology": np.zeros(self.max_frames, dtype=np.float32),
            "bbox": np.zeros(self.max_frames, dtype=np.float32),
            "robot": np.zeros(self.max_frames, dtype=np.float32),
            "total": np.zeros(self.max_frames, dtype=np.float32),
        }

        # For real-time FPS smoothing
        self.last_fps = 0.0

        # Pre-compute constants
        self.inv3 = COLOR_CHANNEL_WEIGHT
        self.spread_weight = METALLIC_SPREAD_WEIGHT

    # --------------------
    # Color mask computation
    # --------------------
    def compute_color_masks(self, img):
        # Direct indexing
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        # Pre-compute scaled values
        g_red_scaled = g * self.red_factor
        b_red_scaled = b * self.red_factor
        g_blue_scaled = g * self.blue_factor
        r_blue_scaled = r * self.blue_factor

        # Compute masks with pre-scaled values
        red_mask = (r >= g_red_scaled) & (r >= b_red_scaled)
        blue_mask = (b >= g_blue_scaled) & (b >= r_blue_scaled)

        return red_mask, blue_mask

    # --------------------
    # Metallic buffer computation
    # --------------------
    def compute_metallic_buffer(self, img):
        # Keep channels as uint8
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        # Use int32 to avoid overflow, then convert to float
        avg_rgb = (r.astype(np.int32) + g.astype(np.int32) + b.astype(np.int32)) * self.inv3

        # Calculate spread
        r_f = r.astype(np.float32)
        g_f = g.astype(np.float32)
        b_f = b.astype(np.float32)
        avg_f = avg_rgb.astype(np.float32)

        spread = (np.abs(r_f - avg_f) + np.abs(g_f - avg_f) + np.abs(b_f - avg_f)) * self.inv3

        # Metallic score
        metallic_raw = avg_f - self.spread_weight * spread

        # Fast normalization using numpy operations
        minv = metallic_raw.min()
        maxv = metallic_raw.max()
        denom = maxv - minv
        if denom < 1e-6:
            return np.zeros_like(metallic_raw)

        metallic_buffer = (metallic_raw - minv) / denom

        return metallic_buffer

    # --------------------
    # Morphology
    # --------------------
    def postprocess_mask(self, mask_bool):
        # Direct conversion
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        return cv2.morphologyEx(mask_uint8, MORPH_OPERATION, self.morph_kernel, iterations=MORPH_ITERATIONS)

    # --------------------
    # Bounding box detection
    # --------------------
    def detect_bboxes(self, mask, min_aspect=MIN_ASPECT_RATIO, max_aspect=MAX_ASPECT_RATIO):
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=CC_CONNECTIVITY)

        if num_labels <= 1:
            return []

        # Filtering
        bboxes = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue

            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            if h == 0:
                continue

            aspect = w / h
            if min_aspect <= aspect <= max_aspect:
                bboxes.append((stats[i, cv2.CC_STAT_LEFT],
                              stats[i, cv2.CC_STAT_TOP],
                              w, h))

        return bboxes

    # --------------------
    # Draw bounding boxes
    # --------------------
    def draw_bboxes(self, img, bboxes, color=ROBOT_BOX_COLOR, thickness=BBOX_THICKNESS, label=None):
        for (x, y, w, h) in bboxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            if label:
                cv2.putText(img, label, (x, y - OVERLAY_TEXT_Y_OFFSET), TEXT_FONT,
                            TEXT_SCALE_MEDIUM, color, TEXT_THICKNESS_BOLD)

    # --------------------
    # Robot detection
    # --------------------
    def detect_robots(self, img, bumper_bboxes, metallic_buffer):
        output_img = img.copy()
        robot_count = 0

        h_img = metallic_buffer.shape[0]

        for (x, y, w, h) in bumper_bboxes:
            # Clamp region bounds
            search_height = int(h * ROBOT_SEARCH_HEIGHT_MULTIPLIER)
            y_top = max(y - search_height, 0)
            x_end = min(x + w, metallic_buffer.shape[1])

            if y_top >= y:
                continue

            # Extract region above bumper
            region_above = metallic_buffer[y_top:y, x:x_end]

            if region_above.size == 0:
                continue

            # Average calculation
            region_avg = float(np.mean(region_above))

            if region_avg > METALLIC_THRESHOLD:
                robot_count += 1
                cv2.rectangle(output_img, (x, y_top), (x + w, y + h), ROBOT_BOX_COLOR, BBOX_THICKNESS)
                cv2.putText(output_img, "Robot", (x, y_top - OVERLAY_TEXT_Y_OFFSET),
                            TEXT_FONT, TEXT_SCALE_MEDIUM, ROBOT_BOX_COLOR, TEXT_THICKNESS_BOLD)

        return output_img, robot_count

    # --------------------
    # Full detection pipeline
    # --------------------
    def detect_and_visualize(self, img):
        t0 = time.perf_counter()
        idx = self.frame_count

        # ---- Color masks ----
        t_color0 = time.perf_counter()
        red_mask_bool, blue_mask_bool = self.compute_color_masks(img)
        self.timing_history["color_masks"][idx] = time.perf_counter() - t_color0

        # ---- Metallic ----
        t_metal0 = time.perf_counter()
        metallic_buffer = self.compute_metallic_buffer(img)
        self.timing_history["metallic"][idx] = time.perf_counter() - t_metal0

        # ---- Morphology ----
        t_morph0 = time.perf_counter()
        red_mask = self.postprocess_mask(red_mask_bool)
        blue_mask = self.postprocess_mask(blue_mask_bool)
        self.timing_history["morphology"][idx] = time.perf_counter() - t_morph0

        # ---- BBox ----
        t_bbox0 = time.perf_counter()
        red_bboxes = self.detect_bboxes(red_mask)
        blue_bboxes = self.detect_bboxes(blue_mask)
        all_bboxes = red_bboxes + blue_bboxes
        self.timing_history["bbox"][idx] = time.perf_counter() - t_bbox0

        # ---- Robot detection ----
        t_robot0 = time.perf_counter()
        if len(red_bboxes) > 0 or len(blue_bboxes) > 0:
            output_img = img.copy()
            self.draw_bboxes(output_img, red_bboxes, color=RED_BUMPER_COLOR)
            self.draw_bboxes(output_img, blue_bboxes, color=BLUE_BUMPER_COLOR)
            output_img, robot_count = self.detect_robots(output_img, all_bboxes, metallic_buffer)
        else:
            output_img = img
            robot_count = 0
        self.timing_history["robot"][idx] = time.perf_counter() - t_robot0

        # Total time
        total_time = time.perf_counter() - t0
        self.timing_history["total"][idx] = total_time

        # Smooth FPS calculation
        fps = 1.0 / total_time
        self.last_fps = fps * FPS_SMOOTHING_FACTOR + self.last_fps * (1.0 - FPS_SMOOTHING_FACTOR)

        # ----- REAL-TIME OVERLAY -----
        if DEBUG_SHOW_OVERLAY:
            if output_img is img:
                output_img = img.copy()
            self._draw_overlay(output_img, idx)

        # Visualization of Color Masks and Metallic Buffer
        if DEBUG_VERBOSE:
            # Display Red Mask - show white for 1, black for 0
            red_mask_display = red_mask_bool.astype(np.uint8) * 255
            cv2.imshow("Red Mask", red_mask_display)  # Show white for 1, black for 0
            
            # Display Blue Mask - show white for 1, black for 0
            blue_mask_display = blue_mask_bool.astype(np.uint8) * 255
            cv2.imshow("Blue Mask", blue_mask_display)  # Show white for 1, black for 0

            # Display Metallic Buffer (scaled for better visualization)
            metallic_display = cv2.normalize(metallic_buffer, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imshow("Metallic Buffer", metallic_display.astype(np.uint8))

        self.frame_count += 1
        return output_img

    # --------------------
    # Draw performance overlay
    # --------------------
    def _draw_overlay(self, img, idx):
        y = OVERLAY_START_Y
        cv2.putText(img, f"FPS: {self.last_fps:.1f}", (OVERLAY_X_OFFSET, y),
                    TEXT_FONT, TEXT_SCALE_LARGE, TEXT_COLOR_FPS, TEXT_THICKNESS_BOLD)

        y += OVERLAY_LINE_SPACING_LARGE
        total_ms = self.timing_history["total"][idx] * 1000
        cv2.putText(img, f"Frame Time: {total_ms:.2f} ms", (OVERLAY_X_OFFSET, y),
                    TEXT_FONT, TEXT_SCALE_MEDIUM, TEXT_COLOR_TIMING, TEXT_THICKNESS_BOLD)

        # Individual component times
        components = [
            ("Color", "color_masks"),
            ("Metallic", "metallic"),
            ("Morph", "morphology"),
            ("BBox", "bbox"),
            ("Robot", "robot")
        ]
        
        y += OVERLAY_LINE_SPACING_MEDIUM
        for label, key in components:
            y += OVERLAY_LINE_SPACING_SMALL
            ms = self.timing_history[key][idx] * 1000
            cv2.putText(img, f"{label}: {ms:.2f} ms", (OVERLAY_X_OFFSET, y),
                        TEXT_FONT, TEXT_SCALE_SMALL, TEXT_COLOR_DETAILS, TEXT_THICKNESS_NORMAL)

    # --------------------
    # Print performance summary
    # --------------------
    def print_summary(self):
        print("\n=== PERFORMANCE SUMMARY ===")
        n = self.frame_count
        
        for key, label in [
            ("color_masks", "Color Mask"),
            ("metallic", "Metallic"),
            ("morphology", "Morphology"),
            ("bbox", "BBox"),
            ("robot", "Robot"),
            ("total", "TOTAL Frame")
        ]:
            avg_ms = np.mean(self.timing_history[key][:n]) * 1000
            print(f"Avg {label:12s} Time: {avg_ms:6.2f} ms")
        
        avg_total = np.mean(self.timing_history["total"][:n])
        print(f"Avg FPS:              {1/avg_total:.2f}")


# ========================
# Single-threaded Video Pipeline
# ========================
def video_pipeline(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video!")
        return

    detector = BumperDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect robots and draw overlay
        output_img = detector.detect_and_visualize(frame)

        # Show result
        cv2.imshow("Detected Robots", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print summary statistics
    detector.print_summary()


# ========================
# Main Entry Point
# ========================
if __name__ == "__main__":
    video_file = "video.mp4"
    video_pipeline(video_file)
