import tensorflow.compat.v1 as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from org_trial_script import load_model_and_get_embedding, MODEL_PATH, IMAGE_PATH

class EmbeddingVisualizer:
    def __init__(self, model_path, image_path):
        self.model_path = model_path
        self.image_path = image_path
        self.embedding = None
        self.original_image = None
        
    def generate_embedding(self):
        """Generate embedding using the existing script"""
        print("Generating face embedding...")
        self.embedding = load_model_and_get_embedding(self.model_path, self.image_path)
        if self.embedding is not None:
            print(f"✅ Embedding generated: {self.embedding.shape[0]}D vector")
            return True
        else:
            print("❌ Failed to generate embedding")
            return False
    
    def load_image(self):
        """Load the original image"""
        self.original_image = cv2.imread(self.image_path)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        return self.original_image is not None
    
    def create_heatmap_overlay(self):
        """Create a heatmap overlay based on embedding values"""
        if self.embedding is None or self.original_image is None:
            return None
        
        # Normalize embedding values to 0-1 range
        normalized_embedding = (self.embedding - self.embedding.min()) / (self.embedding.max() - self.embedding.min())
        
        # Create a grid to map embedding values
        img_height, img_width = self.original_image.shape[:2]
        
        # Calculate grid size (make it roughly square)
        embedding_size = len(normalized_embedding)
        grid_size = int(np.sqrt(embedding_size))
        
        # Reshape embedding to fit grid (pad if necessary)
        padded_size = grid_size * grid_size
        if embedding_size < padded_size:
            padded_embedding = np.pad(normalized_embedding, (0, padded_size - embedding_size), 'constant')
        else:
            padded_embedding = normalized_embedding[:padded_size]
        
        # Reshape to grid
        embedding_grid = padded_embedding.reshape(grid_size, grid_size)
        
        # Resize to match image dimensions
        heatmap = cv2.resize(embedding_grid, (img_width, img_height))
        
        # Apply colormap
        heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Blend with original image
        alpha = 0.4
        overlay = cv2.addWeighted(self.original_image, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlay, heatmap_colored
    
    def create_feature_points(self):
        """Create visualization showing key feature points based on embedding"""
        if self.embedding is None or self.original_image is None:
            return None
        
        img_copy = self.original_image.copy()
        height, width = img_copy.shape[:2]
        
        # Select top embedding values as "key features"
        top_indices = np.argsort(np.abs(self.embedding))[-20:]  # Top 20 features
        
        # Map these to facial regions (simplified mapping)
        # This is a simplified approach - real face mapping would need proper face detection
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, idx in enumerate(top_indices):
            # Map embedding index to image coordinates
            x = int((idx % 32) * width / 32)  # Assuming 32x32 grid mapping
            y = int((idx // 32) * height / 32)
            
            # Ensure coordinates are within image bounds
            x = max(5, min(width-5, x))
            y = max(5, min(height-5, y))
            
            # Draw circle with size based on embedding value
            radius = int(abs(self.embedding[idx]) * 10) + 3
            color = colors[i % len(colors)]
            
            cv2.circle(img_copy, (x, y), radius, color, 2)
            cv2.putText(img_copy, f'{i+1}', (x-5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return img_copy
    
    def create_embedding_chart(self):
        """Create a chart showing embedding values"""
        if self.embedding is None:
            return None
        
        plt.figure(figsize=(15, 6))
        
        # Plot 1: Full embedding vector
        plt.subplot(1, 3, 1)
        plt.plot(self.embedding)
        plt.title('Full Embedding Vector')
        plt.xlabel('Dimension')
        plt.ylabel('Value')
        plt.grid(True)
        
        # Plot 2: Embedding histogram
        plt.subplot(1, 3, 2)
        plt.hist(self.embedding, bins=50, alpha=0.7)
        plt.title('Embedding Value Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Plot 3: Top 20 values
        plt.subplot(1, 3, 3)
        top_20_indices = np.argsort(np.abs(self.embedding))[-20:]
        top_20_values = self.embedding[top_20_indices]
        plt.bar(range(20), top_20_values)
        plt.title('Top 20 Embedding Features')
        plt.xlabel('Feature Rank')
        plt.ylabel('Value')
        plt.xticks(range(20), [f'{i+1}' for i in range(20)])
        plt.grid(True)
        
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_all(self):
        """Create comprehensive visualization"""
        print("🎨 Creating embedding visualizations...")
        
        # Generate embedding if not already done
        if self.embedding is None:
            if not self.generate_embedding():
                return False
        
        # Load image
        if not self.load_image():
            print("❌ Failed to load image")
            return False
        
        # Create visualizations
        print("Creating heatmap overlay...")
        overlay, heatmap = self.create_heatmap_overlay()
        
        print("Creating feature points...")
        feature_points = self.create_feature_points()
        
        print("Creating embedding charts...")
        chart_fig = self.create_embedding_chart()
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Heatmap overlay
        if overlay is not None:
            axes[0, 1].imshow(overlay)
            axes[0, 1].set_title('Embedding Heatmap Overlay')
            axes[0, 1].axis('off')
        
        # Pure heatmap
        if heatmap is not None:
            axes[0, 2].imshow(heatmap)
            axes[0, 2].set_title('Pure Embedding Heatmap')
            axes[0, 2].axis('off')
        
        # Feature points
        if feature_points is not None:
            axes[1, 0].imshow(feature_points)
            axes[1, 0].set_title('Key Feature Points')
            axes[1, 0].axis('off')
        
        # Embedding statistics
        axes[1, 1].text(0.1, 0.8, f'Embedding Dimensions: {len(self.embedding)}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'Min Value: {self.embedding.min():.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Max Value: {self.embedding.max():.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'Mean: {self.embedding.mean():.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Std Dev: {self.embedding.std():.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, f'L2 Norm: {np.linalg.norm(self.embedding):.4f}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Embedding Statistics')
        axes[1, 1].axis('off')
        
        # Embedding vector visualization
        axes[1, 2].plot(self.embedding[:100])  # Plot first 100 dimensions
        axes[1, 2].set_title('First 100 Embedding Dimensions')
        axes[1, 2].set_xlabel('Dimension')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('../outputs/embedding_visualization.png', dpi=300, bbox_inches='tight')
        print("📁 Saved: ../outputs/embedding_visualization.png")
        plt.close()
        
        # Show detailed chart
        if chart_fig:
            chart_fig.savefig('../outputs/embedding_analysis.png', dpi=300, bbox_inches='tight')
            print("📁 Saved: ../outputs/embedding_analysis.png")
            plt.close(chart_fig)
        
        print("✅ Visualization complete!")
        print("📁 Saved files: '../outputs/embedding_visualization.png' and '../outputs/embedding_analysis.png'")
        
        return True

def main():
    """Main function to run the visualization"""
    print("🎯 Face Embedding Visualizer")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image file not found: {IMAGE_PATH}")
        return
    
    # Create visualizer and run
    visualizer = EmbeddingVisualizer(MODEL_PATH, IMAGE_PATH)
    success = visualizer.visualize_all()
    
    if success:
        print("\n🎉 Success! Check the generated visualization images.")
        print("\nInterpretation Guide:")
        print("• Heatmap shows which image regions contribute most to the embedding")
        print("• Feature points mark the most significant embedding dimensions")
        print("• Charts show the distribution and pattern of embedding values")
        print("• Higher intensity areas represent stronger facial features")
    else:
        print("\n❌ Visualization failed. Please check your model and image files.")

if __name__ == "__main__":
    main()