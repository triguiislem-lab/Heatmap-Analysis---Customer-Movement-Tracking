#!/usr/bin/env python3
"""
Example Usage of Heatmap Analyzer

This script demonstrates how to use the heatmap analyzer programmatically.
"""

from heatmap_analyzer import HeatmapAnalyzer, Config
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic usage example with default configuration."""
    print("=" * 50)
    print("EXAMPLE 1: Basic Usage (Default Settings)")
    print("=" * 50)
    
    # Specify your video path here
    video_path = "your_video.mp4"
    
    # Check if file exists
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        logger.info("Please update the video_path variable with a valid video file")
        return
    
    # Create analyzer with default configuration
    analyzer = HeatmapAnalyzer()
    
    # Run analysis
    success = analyzer.analyze_video(video_path)
    
    if success:
        print("\n✅ Analysis completed!")
        print("Results saved to: output/")


def example_custom_config():
    """Example with custom configuration."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 50)
    
    video_path = "your_video.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Create custom configuration
    config = Config(
        model_name="yolov8n.pt",    # Fast model (yolov8s.pt for better accuracy)
        grid_size=(15, 10),          # 15x10 grid (more zones)
        confidence=0.6,              # Higher confidence threshold
        skip_frames=3,               # Skip more frames for faster processing
        output_dir="custom_output"   # Custom output directory
    )
    
    # Create analyzer with custom config
    analyzer = HeatmapAnalyzer(config)
    
    # Run analysis
    success = analyzer.analyze_video(video_path)
    
    if success:
        print("\n✅ Analysis completed with custom settings!")
        print(f"Results saved to: {config.output_dir}/")


def example_high_accuracy():
    """Example optimized for accuracy over speed."""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: High Accuracy Mode")
    print("=" * 50)
    
    video_path = "your_video.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Configuration for better accuracy
    config = Config(
        model_name="yolov8m.pt",    # Medium model (more accurate)
        grid_size=(10, 8),
        confidence=0.4,              # Lower threshold to catch more detections
        skip_frames=1,               # Process every frame
        output_dir="high_accuracy_output"
    )
    
    analyzer = HeatmapAnalyzer(config)
    
    print("⚠️ Warning: This will be slower but more accurate")
    success = analyzer.analyze_video(video_path)
    
    if success:
        print("\n✅ High accuracy analysis completed!")


def example_fast_mode():
    """Example optimized for speed."""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Fast Processing Mode")
    print("=" * 50)
    
    video_path = "your_video.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Configuration for faster processing
    config = Config(
        model_name="yolov8n.pt",    # Nano model (fastest)
        grid_size=(8, 6),            # Fewer zones
        confidence=0.6,              # Higher threshold
        skip_frames=5,               # Skip more frames
        output_dir="fast_output"
    )
    
    analyzer = HeatmapAnalyzer(config)
    
    print("⚡ Processing in fast mode...")
    success = analyzer.analyze_video(video_path)
    
    if success:
        print("\n✅ Fast analysis completed!")


def main():
    """Run all examples."""
    print("\n🔥 HEATMAP ANALYZER - USAGE EXAMPLES\n")
    
    # Uncomment the example you want to run:
    
    # example_basic_usage()
    # example_custom_config()
    # example_high_accuracy()
    # example_fast_mode()
    
    print("\n" + "=" * 50)
    print("💡 TIP: Update video_path in each example before running")
    print("=" * 50)
    print("\nAvailable configurations:")
    print("  - model_name: yolov8n.pt (fast), yolov8s.pt, yolov8m.pt (accurate)")
    print("  - grid_size: (cols, rows) - e.g., (10, 8) for 10x8 grid")
    print("  - confidence: 0.0-1.0 (higher = fewer but more confident detections)")
    print("  - skip_frames: 1-10 (higher = faster but less data)")
    print("  - output_dir: directory name for results")
    print("\nOutput files generated:")
    print("  - heatmap.png: Main heatmap visualization")
    print("  - last_frame_with_heatmap.png: Last frame with overlay")
    print("  - analysis.txt: Zone statistics and summary")


if __name__ == "__main__":
    main()
