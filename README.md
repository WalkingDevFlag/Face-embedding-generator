# Face Embedding Visualization Project

This project uses a pre-trained FaceNet model to generate face embeddings and visualize them on facial images.

## 📁 Project Structure

```
EZ pic/
├── models/                          # Pre-trained models and model files
│   ├── 20180402-114759/            # FaceNet model directory
│   │   ├── 20180402-114759.pb      # Frozen TensorFlow model
│   │   ├── model-20180402-114759.ckpt-275.data-00000-of-00001
│   │   ├── model-20180402-114759.ckpt-275.index
│   │   └── model-20180402-114759.meta
│   └── 20180402-114759.zip         # Original model archive
├── images/                          # Input images for processing
│   ├── sample.png                  # Main test image
│   └── unnamed.png                 # Additional image
├── scripts/                         # Python scripts
│   ├── trial_script.py             # Main embedding generation script
│   └── embedding_visualizer.py     # Visualization script
├── outputs/                         # Generated visualizations and results
│   ├── embedding_visualization.png # Comprehensive visualization
│   └── embedding_analysis.png      # Detailed analysis charts
└── README.md                       # This file
```

## 🚀 Usage

### 1. Generate Face Embeddings
```bash
cd scripts
python trial_script.py
```
This script:
- Loads the FaceNet model from `../models/20180402-114759/20180402-114759.pb`
- Processes `../images/sample.png`
- Generates a 512-dimensional face embedding vector

### 2. Visualize Embeddings
```bash
cd scripts
python embedding_visualizer.py
```
This script:
- Runs the embedding generation
- Creates visual mappings of embeddings onto the face
- Saves comprehensive visualizations to `../outputs/`

## 📊 Output Visualizations

### embedding_visualization.png
A 6-panel comprehensive view showing:
- **Original Image**: Your input face image
- **Heatmap Overlay**: Which facial regions contribute most to the embedding
- **Pure Heatmap**: Raw embedding intensity map
- **Feature Points**: Key embedding dimensions marked on the face
- **Statistics**: Embedding metrics and properties
- **Dimension Plot**: First 100 embedding dimensions

### embedding_analysis.png
Detailed analysis charts:
- **Full Vector Plot**: Complete 512D embedding visualization
- **Value Distribution**: Histogram of embedding values
- **Top Features**: 20 most significant embedding dimensions

## 🎯 Understanding the Results

- **Heatmap Colors**: 
  - Red/Yellow = High embedding values (strong facial features)
  - Blue/Purple = Low embedding values (less significant areas)
- **Feature Points**: Numbered circles show the most important embedding dimensions
- **Embedding Vector**: 512 floating-point numbers that uniquely represent the face

## 🔧 Requirements

- Python 3.x
- TensorFlow 2.x (with v1 compatibility)
- OpenCV (`cv2`)
- PIL/Pillow
- NumPy
- Matplotlib

## 📝 Technical Details

- **Model**: Pre-trained FaceNet (20180402-114759)
- **Input Size**: 160x160 pixels
- **Output**: 512-dimensional embedding vector
- **Preprocessing**: Image resizing and pre-whitening normalization

## 🎨 Customization

To use your own images:
1. Place new images in the `images/` folder
2. Update `IMAGE_PATH` in `scripts/trial_script.py`
3. Run the scripts to generate new visualizations

## 🔬 Research Applications

This visualization can help understand:
- Which facial features are most important for face recognition
- How different faces map to embedding space
- The distribution and patterns of facial embeddings
- Feature importance for identity verification

---

*Generated on October 3, 2025*