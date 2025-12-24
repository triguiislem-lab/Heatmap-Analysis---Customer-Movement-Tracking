# 🚀 Quick Start Guide

## Installation (2 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/triguiislem-lab/Heatmap-Analysis---Customer-Movement-Tracking.git
cd Heatmap-Analysis---Customer-Movement-Tracking

# 2. Install dependencies
pip install -r requirements.txt
```

## Run Your First Analysis (1 command)

```bash
python heatmap_analyzer.py your_video.mp4
```

That's it! Results will be in the `output/` folder.

## What You'll Get

After running the analysis, check the `output/` folder:

- **`zones_analysis.png`** - Professional visualization with heatmap and top zones highlighted
- **`comprehensive_comparison.png`** - 3-panel comparison (Original | Heatmap | Zones)
- **Console output** - Statistics about most visited zones

## Common Use Cases

### 1. Retail Store Analysis
```bash
python heatmap_analyzer.py store_video.mp4 --grid 12 8
```
Analyze customer movement patterns with a 12x8 grid.

### 2. Fast Preview
```bash
python heatmap_analyzer.py video.mp4 --skip-frames 5
```
Quick analysis for large videos.

### 3. High Accuracy
```bash
python heatmap_analyzer.py video.mp4 --model yolov8m.pt --skip-frames 1
```
Most accurate detection (slower).

## Troubleshooting

**Problem**: Script is too slow
```bash
# Solution: Skip more frames
python heatmap_analyzer.py video.mp4 --skip-frames 5
```

**Problem**: Not detecting people accurately
```bash
# Solution: Lower confidence threshold
python heatmap_analyzer.py video.mp4 --confidence 0.3
```

**Problem**: Need more detailed zones
```bash
# Solution: Increase grid size
python heatmap_analyzer.py video.mp4 --grid 15 12
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [example_usage.py](example_usage.py) for programmatic usage
- Experiment with different parameters

## Support

If you encounter issues:
1. Check that your video file exists and is readable
2. Ensure Python 3.8+ is installed
3. Verify all dependencies are installed: `pip list`

Happy analyzing! 🎉
