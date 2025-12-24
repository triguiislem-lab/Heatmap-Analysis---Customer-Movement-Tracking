#!/usr/bin/env python3
"""
Minimalist Heatmap Analysis - Customer Movement Tracking
Uses YOLOv8 for person detection and generates heatmaps to analyze foot traffic patterns.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for heatmap analysis."""
    model_name: str = "yolov8n.pt"  # YOLOv8 nano (fastest)
    grid_size: Tuple[int, int] = (10, 8)
    confidence: float = 0.5
    skip_frames: int = 2
    output_dir: str = "output"

class HeatmapAnalyzer:
    """Heatmap analyzer using YOLOv8 for person detection."""

    def __init__(self, config: Config = None):
        """Initialize with configuration."""
        self.config = config or Config()
        self.model = None
        self.heatmap = None
        self.frame_size = None
        self.last_frame = None

    def _load_model(self) -> bool:
        """Load YOLOv8 model (auto-downloads if needed)."""
        try:
            self.model = YOLO(self.config.model_name)
            logger.info(f"Loaded YOLO model: {self.config.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def _detect_feet(self, results) -> List[Tuple[int, int]]:
        """Extract foot positions from YOLO detections."""
        feet_positions = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Filter for person class (class 0 in COCO)
                    if int(box.cls) == 0 and float(box.conf) > self.config.confidence:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Calculate foot position (bottom center of bounding box)
                        foot_x = (x1 + x2) // 2
                        foot_y = y2  # Bottom of the box

                        feet_positions.append((foot_x, foot_y))

        return feet_positions

    def _update_heatmap(self, feet_positions: List[Tuple[int, int]]):
        """Update heatmap with foot positions."""
        for x, y in feet_positions:
            if 0 <= x < self.frame_size[1] and 0 <= y < self.frame_size[0]:
                self.heatmap[y, x] += 1

    def _create_zones(self) -> dict:
        """Create grid zones for analysis."""
        h, w = self.frame_size
        zone_h, zone_w = h // self.config.grid_size[1], w // self.config.grid_size[0]

        zones = {}
        for i in range(self.config.grid_size[0]):
            for j in range(self.config.grid_size[1]):
                x1, y1 = i * zone_w, j * zone_h
                x2, y2 = x1 + zone_w, y1 + zone_h
                zones[f"{i},{j}"] = ((x1, y1), (x2, y2))

        return zones

    def _analyze_zones(self, feet_positions: List[Tuple[int, int]]) -> dict:
        """Analyze zone visits."""
        zones = self._create_zones()
        zone_visits = {zone_id: 0 for zone_id in zones}

        for x, y in feet_positions:
            for zone_id, ((x1, y1), (x2, y2)) in zones.items():
                if x1 <= x < x2 and y1 <= y < y2:
                    zone_visits[zone_id] += 1
                    break

        return zone_visits

    def analyze_video(self, video_path: str) -> bool:
        """Process video and generate heatmap analysis."""
        if not self._load_model():
            return False

        # Validate video path
        video_file = Path(video_path)
        if not video_file.exists():
            logger.error(f"Video file not found: {video_path}")
            return False

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False

        # Get video properties
        ret, first_frame = cap.read()
        if not ret:
            logger.error("Cannot read first frame")
            cap.release()
            return False

        self.frame_size = first_frame.shape[:2]  # (height, width)
        self.heatmap = np.zeros(self.frame_size, dtype=np.float32)

        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        logger.info(f"Processing video: {video_file.name}")
        logger.info(f"Frame size: {self.frame_size[1]}x{self.frame_size[0]}")

        frame_count = 0
        all_feet_positions = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.last_frame = frame.copy()
                frame_count += 1

                # Skip frames for performance
                if frame_count % self.config.skip_frames != 0:
                    continue

                # YOLO detection
                results = self.model(frame, verbose=False)

                # Extract foot positions
                feet_positions = self._detect_feet(results)
                all_feet_positions.extend(feet_positions)

                # Update heatmap
                self._update_heatmap(feet_positions)

                # Progress update
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames, detected {len(feet_positions)} people")

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")

        finally:
            cap.release()

        logger.info(f"Processing complete. Total frames: {frame_count}")
        logger.info(f"Total foot detections: {len(all_feet_positions)}")

        # Generate outputs
        self._save_results(all_feet_positions)

        return True

    def _save_results(self, all_feet_positions: List[Tuple[int, int]]):
        """Save heatmap and analysis results."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Zone analysis
        zone_visits = self._analyze_zones(all_feet_positions)
        top_zones = sorted(zone_visits.items(), key=lambda x: x[1], reverse=True)[:5]

        # Create professional visualizations
        self._create_last_frame_overlays(output_dir, zone_visits)

        # Print summary to console
        logger.info("\n" + "=" * 50)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total detections: {len(all_feet_positions)}")
        logger.info(f"Grid size: {self.config.grid_size}")
        logger.info("\nTop 5 Most Visited Zones:")
        for i, (zone_id, visits) in enumerate(top_zones, 1):
            logger.info(f"  {i}. Zone {zone_id}: {visits} visits")
        logger.info("=" * 50)

    def _create_last_frame_overlays(self, output_dir: Path, zone_visits: dict):
        """Create heatmap overlay on the last frame with zone highlights."""
        if self.last_frame is None or np.max(self.heatmap) == 0:
            logger.warning("No last frame or heatmap data available")
            return

        try:
            # Smooth and normalize heatmap
            smoothed_heatmap = cv2.GaussianBlur(self.heatmap, (51, 51), 0)
            normalized_heatmap = cv2.normalize(smoothed_heatmap, None, 0, 255, cv2.NORM_MINMAX)
            normalized_heatmap = normalized_heatmap.astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

            # Create overlay (60% original + 40% heatmap)
            overlay = cv2.addWeighted(self.last_frame, 0.6, colored_heatmap, 0.4, 0)

            # Add zone visualization
            overlay_with_zones = self._add_zone_visualization(overlay.copy(), zone_visits)
            cv2.imwrite(str(output_dir / 'zones_analysis.png'), overlay_with_zones)

            # Create comprehensive 3-panel comparison
            self._create_comprehensive_comparison(output_dir, overlay, overlay_with_zones, zone_visits)

            logger.info(f"Zone analysis visualization saved to {output_dir / 'zones_analysis.png'}")
            logger.info(f"Comprehensive comparison saved to {output_dir / 'comprehensive_comparison.png'}")

        except Exception as e:
            logger.error(f"Error creating overlays: {e}")

    def _add_zone_visualization(self, image: np.ndarray, zone_visits: dict) -> np.ndarray:
        """Add zone grid and highlight top zones."""
        h, w = self.frame_size
        zone_w, zone_h = w // self.config.grid_size[0], h // self.config.grid_size[1]

        # Get top 5 zones
        top_zones = sorted(zone_visits.items(), key=lambda x: x[1], reverse=True)[:5]

        # Define colors for top zones (BGR format)
        zone_colors = [
            (0, 255, 255),    # Yellow (top zone)
            (0, 165, 255),    # Orange (2nd)
            (0, 0, 255),      # Red (3rd)
            (255, 0, 255),    # Magenta (4th)
            (255, 255, 0),    # Cyan (5th)
        ]

        # Draw all zone grid lines (subtle)
        for i in range(self.config.grid_size[0] + 1):
            x = i * zone_w
            cv2.line(image, (x, 0), (x, h), (128, 128, 128), 1)

        for j in range(self.config.grid_size[1] + 1):
            y = j * zone_h
            cv2.line(image, (0, y), (w, y), (128, 128, 128), 1)

        # Highlight and label top zones
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        for rank, (zone_id, visits) in enumerate(top_zones):
            if visits == 0:
                continue

            # Parse zone coordinates
            zone_x, zone_y = map(int, zone_id.split(','))

            # Calculate zone boundaries
            x1 = zone_x * zone_w
            y1 = zone_y * zone_h
            x2 = min(x1 + zone_w, w)
            y2 = min(y1 + zone_h, h)

            # Get color for this rank
            color = zone_colors[rank] if rank < len(zone_colors) else (255, 255, 255)

            # Draw thick border for top zones
            border_thickness = 4 if rank == 0 else 3
            cv2.rectangle(image, (x1, y1), (x2, y2), color, border_thickness)

            # Create label combining rank and visits
            label = f"#{rank+1} ({visits})"
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

            # Position label in center of zone
            zone_center_x = (x1 + x2) // 2
            zone_center_y = (y1 + y2) // 2
            label_x = zone_center_x - label_size[0] // 2
            label_y = zone_center_y + label_size[1] // 2

            # Ensure label stays within zone boundaries
            label_x = max(x1 + 5, min(label_x, x2 - label_size[0] - 5))
            label_y = max(y1 + label_size[1] + 5, min(label_y, y2 - 5))

            # Draw label background
            bg_padding = 4
            cv2.rectangle(image,
                         (label_x - bg_padding, label_y - label_size[1] - bg_padding),
                         (label_x + label_size[0] + bg_padding, label_y + bg_padding),
                         (0, 0, 0), -1)

            # Draw white border around background
            cv2.rectangle(image,
                         (label_x - bg_padding, label_y - label_size[1] - bg_padding),
                         (label_x + label_size[0] + bg_padding, label_y + bg_padding),
                         (255, 255, 255), 1)

            # Draw label text
            cv2.putText(image, label, (label_x, label_y), font, font_scale, color, thickness)

        # Add legend
        self._add_legend(image, top_zones, zone_colors)

        return image

    def _add_legend(self, image: np.ndarray, top_zones: List[Tuple[str, int]], zone_colors: List[Tuple[int, int, int]]):
        """Add legend showing top zones."""
        h, w = image.shape[:2]

        # Legend background
        legend_width = 250
        legend_height = min(200, 30 + len(top_zones) * 25)
        legend_x = w - legend_width - 10
        legend_y = 10

        # Draw legend background
        cv2.rectangle(image, (legend_x, legend_y),
                     (legend_x + legend_width, legend_y + legend_height),
                     (0, 0, 0), -1)
        cv2.rectangle(image, (legend_x, legend_y),
                     (legend_x + legend_width, legend_y + legend_height),
                     (255, 255, 255), 2)

        # Legend title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Top Zones", (legend_x + 10, legend_y + 25),
                   font, 0.7, (255, 255, 255), 2)

        # Legend entries
        for i, (zone_id, visits) in enumerate(top_zones[:5]):
            if visits == 0:
                continue

            y_pos = legend_y + 50 + i * 25
            color = zone_colors[i] if i < len(zone_colors) else (255, 255, 255)

            # Draw color box
            cv2.rectangle(image, (legend_x + 10, y_pos - 10),
                         (legend_x + 25, y_pos + 5), color, -1)
            cv2.rectangle(image, (legend_x + 10, y_pos - 10),
                         (legend_x + 25, y_pos + 5), (255, 255, 255), 1)

            # Draw text
            text = f"#{i+1}: Zone {zone_id} ({visits})"
            cv2.putText(image, text, (legend_x + 35, y_pos),
                       font, 0.45, (255, 255, 255), 1)

    def _create_comprehensive_comparison(self, output_dir: Path, overlay: np.ndarray,
                                        overlay_with_zones: np.ndarray, zone_visits: dict):
        """Create a comprehensive 3-panel comparison."""
        h, w = self.last_frame.shape[:2]

        # Create 3-panel comparison
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = self.last_frame
        comparison[:, w:2*w] = overlay
        comparison[:, 2*w:] = overlay_with_zones

        # Add panel labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (255, 255, 255)
        thickness = 2

        cv2.putText(comparison, "Original Last Frame", (10, 40), font, font_scale, color, thickness)
        cv2.putText(comparison, "With Heatmap", (w + 10, 40), font, font_scale, color, thickness)
        cv2.putText(comparison, "With Top Zones", (2*w + 10, 40), font, font_scale, color, thickness)

        # Add summary statistics
        top_zones = sorted(zone_visits.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_zones and top_zones[0][1] > 0:
            summary_y = h - 60
            cv2.putText(comparison, f"Top Zone: {top_zones[0][0]} ({top_zones[0][1]} visits)",
                       (10, summary_y), font, 0.7, (0, 255, 255), 2)

            if len(top_zones) > 1 and top_zones[1][1] > 0:
                cv2.putText(comparison, f"2nd: {top_zones[1][0]} ({top_zones[1][1]} visits)",
                           (10, summary_y + 25), font, 0.7, (0, 165, 255), 2)

        cv2.imwrite(str(output_dir / 'comprehensive_comparison.png'), comparison)


def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(description='Heatmap Analysis for Customer Movement Tracking')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model name (default: yolov8n.pt)')
    parser.add_argument('--grid', type=int, nargs=2, default=[10, 8], help='Grid size (cols rows, default: 10 8)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--skip-frames', type=int, default=2, help='Number of frames to skip (default: 2)')
    parser.add_argument('--output', type=str, default='output', help='Output directory (default: output)')

    args = parser.parse_args()

    logger.info("🔥 HEATMAP ANALYZER")
    logger.info("Using YOLOv8 for Person Detection")
    logger.info("=" * 40)

    # Create configuration
    config = Config(
        model_name=args.model,
        grid_size=tuple(args.grid),
        confidence=args.confidence,
        skip_frames=args.skip_frames,
        output_dir=args.output
    )

    # Run analysis
    analyzer = HeatmapAnalyzer(config)
    success = analyzer.analyze_video(args.video_path)

    if success:
        logger.info("✅ Analysis completed successfully!")
        logger.info(f"📁 Results saved to: {config.output_dir}/")
    else:
        logger.error("❌ Analysis failed!")

    return success


if __name__ == "__main__":
    main()
