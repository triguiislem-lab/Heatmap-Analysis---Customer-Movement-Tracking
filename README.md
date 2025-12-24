# 🔥 Heatmap Analysis - Customer Movement Tracking

**AI-powered customer movement analysis using YOLOv8 person detection and heatmap visualization for retail analytics.**

Analyze customer foot traffic patterns in retail environments using computer vision and generate professional visualizations with zone-based analytics.

## 🎯 Features

- ✅ Automatic person detection using YOLOv8
- ✅ Foot position tracking
- ✅ Heatmap generation for foot traffic analysis
- ✅ Zone-based analysis
- ✅ Last frame overlay with heatmap
- ✅ Comprehensive results export
- ✅ Command-line interface support
- ✅ Configurable parameters

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

The YOLOv8 model will be automatically downloaded on first run.

## 🚀 Usage

### Method 1: Command Line (Recommended)

```bash
# Basic usage
python heatmap_analyzer.py path/to/your/video.mp4

# With custom parameters
python heatmap_analyzer.py video.mp4 --confidence 0.6 --skip-frames 3

# High accuracy mode
python heatmap_analyzer.py video.mp4 --model yolov8m.pt --skip-frames 1

# Custom output directory
python heatmap_analyzer.py video.mp4 --output my_results
```

### Method 2: Programmatic Usage

```python
from heatmap_analyzer import HeatmapAnalyzer, Config

# Create configuration
config = Config(
    model_name="yolov8n.pt",
    grid_size=(10, 8),
    confidence=0.5,
    skip_frames=2,
    output_dir="output"
)

# Create analyzer and run
analyzer = HeatmapAnalyzer(config)
analyzer.analyze_video("your_video.mp4")
```

See [`example_usage.py`](example_usage.py) for more examples.

## ⚙️ Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `model_name` | YOLO model to use | `yolov8n.pt` | `yolov8n.pt` (fastest), `yolov8s.pt`, `yolov8m.pt` (most accurate) |
| `grid_size` | Grid size (columns, rows) | `(10, 8)` | Any tuple of integers |
| `confidence` | Detection confidence threshold | `0.5` | `0.0` to `1.0` |
| `skip_frames` | Frames to skip for speed | `2` | `1` (slowest, most data) to `10` (fastest) |
| `output_dir` | Output directory | `output` | Any valid directory name |

## 📊 Output Files

The analyzer generates the following professional visualizations in the output directory:

1. **`zones_analysis.png`** - Professional visualization with heatmap overlay and top zones highlighted with color-coded labels
2. **`comprehensive_comparison.png`** - 3-panel comparison showing: Original Frame | Heatmap Overlay | Zone Analysis

Statistics are displayed in the console output including total detections and top 5 most visited zones.

## 📁 Project Structure

```
heatmap-analyzer/
├── heatmap_analyzer.py      # Main analyzer module
├── example_usage.py         # Usage examples
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── output/                 # Generated results (created automatically)
    ├── heatmap.png
    ├── last_frame_with_heatmap.png
    └── analysis.txt
```

## 🔧 Command Line Arguments

```bash
python heatmap_analyzer.py --help

usage: heatmap_analyzer.py [-h] [--model MODEL] [--grid GRID GRID]
                          [--confidence CONFIDENCE] [--skip-frames SKIP_FRAMES]
                          [--output OUTPUT]
                          video_path

positional arguments:
  video_path            Path to video file

optional arguments:
  -h, --help            Show this help message and exit
  --model MODEL         YOLO model name (default: yolov8n.pt)
  --grid GRID GRID      Grid size (cols rows, default: 10 8)
  --confidence CONFIDENCE
                        Detection confidence threshold (default: 0.5)
  --skip-frames SKIP_FRAMES
                        Number of frames to skip (default: 2)
  --output OUTPUT       Output directory (default: output)
```

## 💡 Usage Tips

### For Fast Processing
```bash
python heatmap_analyzer.py video.mp4 --model yolov8n.pt --skip-frames 5
```

### For High Accuracy
```bash
python heatmap_analyzer.py video.mp4 --model yolov8m.pt --skip-frames 1 --confidence 0.4
```

### For Fine-Grained Analysis
```bash
python heatmap_analyzer.py video.mp4 --grid 15 12
```

## 🛠️ Technologies

- **YOLOv8** (Ultralytics) - Person detection
- **OpenCV** - Video processing
- **NumPy** - Numerical computations
- **Matplotlib** - Visualizations
- **Seaborn** - Heatmap styling

## 📝 Example Results

### Heatmap Visualization
The heatmap shows areas with high foot traffic in warmer colors (red/yellow) and low traffic areas in cooler colors (blue/purple).

### Zone Analysis
```
HEATMAP ANALYSIS RESULTS
==============================

Total detections: 1247
Grid size: (10, 8)

Top 5 Most Visited Zones:
1. Zone 4,3: 156 visits
2. Zone 5,3: 143 visits
3. Zone 3,4: 128 visits
4. Zone 6,2: 112 visits
5. Zone 4,2: 98 visits
```

## 🐛 Troubleshooting

### Video file not found
Make sure the video path is correct. Use absolute paths if needed:
```bash
python heatmap_analyzer.py /full/path/to/video.mp4
```

### Out of memory
Try increasing `skip_frames` or using a smaller model:
```bash
python heatmap_analyzer.py video.mp4 --skip-frames 5
```

### Slow processing
Use the nano model and skip more frames:
```bash
python heatmap_analyzer.py video.mp4 --model yolov8n.pt --skip-frames 5
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

Retail analytics project for customer movement analysis.

---

**Note**: The YOLOv8 nano model is optimized for speed. For better accuracy, use `yolov8s.pt` (small) or `yolov8m.pt` (medium), though processing will be slower.
