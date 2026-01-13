#!/usr/bin/env python3
"""
Feature Visualization for YOLO Model
ดูว่าโมเดลใช้ฟีเจอร์อะไรในการตัดสินใจ
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


class FeatureVisualizer:
    """
    Visualize what features the model learns
    """
    
    def __init__(self, model_path):
        """
        Load YOLO model
        
        Args:
            model_path: path to .pt file
        """
        self.model = YOLO(model_path)
        self.model_torch = self.model.model
        self.device = next(self.model_torch.parameters()).device
        
        # Storage for activations
        self.activations = {}
        self.gradients = {}
        
        print(f"✓ Loaded model: {model_path}")
        print(f"✓ Device: {self.device}")
    
    # ========================================================================
    # 1. Feature Maps - ดูว่าแต่ละ layer เห็นอะไร
    # ========================================================================
    
    def visualize_feature_maps(self, image_path, output_dir='feature_maps'):
        """
        Visualize feature maps from different layers
        แสดงว่าแต่ละ layer ของโมเดลเห็นรูปภาพยังไง
        """
        
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("🔍 Visualizing Feature Maps")
        print("="*60)
        
        # Read and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare for model (YOLO preprocessing)
        img_resized = cv2.resize(image_rgb, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Hook to capture activations
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks on different layers
        layers_to_visualize = []
        for i, (name, module) in enumerate(self.model_torch.model.named_modules()):
            # เลือก Conv layers สำคัญๆ
            if isinstance(module, nn.Conv2d):
                if i % 5 == 0:  # ทุก 5 layers
                    module.register_forward_hook(get_activation(name))
                    layers_to_visualize.append(name)
        
        print(f"✓ Registered hooks on {len(layers_to_visualize)} layers")
        
        # Forward pass
        with torch.no_grad():
            _ = self.model_torch(img_tensor)
        
        # Visualize each layer
        for layer_name in layers_to_visualize[:8]:  # แค่ 8 layers แรก
            
            if layer_name not in activations:
                continue
            
            feature_map = activations[layer_name][0].cpu().numpy()
            
            # Plot
            n_features = min(16, feature_map.shape[0])  # แสดงแค่ 16 filters
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f'Layer: {layer_name}\nShape: {feature_map.shape}', 
                        fontsize=14, fontweight='bold')
            
            for i in range(n_features):
                ax = axes[i // 4, i % 4]
                
                # Get feature map
                feat = feature_map[i]
                
                # Normalize
                feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
                
                # Plot
                im = ax.imshow(feat, cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Filter {i}', fontsize=8)
            
            plt.tight_layout()
            
            # Save
            safe_name = layer_name.replace('/', '_').replace('.', '_')
            save_path = f'{output_dir}/layer_{safe_name}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved: {save_path}")
        
        print(f"\n✅ Feature maps saved to: {output_dir}/")
    
    # ========================================================================
    # 2. Grad-CAM - ดูว่าโมเดลโฟกัสที่ไหน
    # ========================================================================
    
    def visualize_gradcam(self, image_path, output_dir='gradcam'):
        """
        Grad-CAM visualization
        แสดงว่าโมเดลมอง/ใส่ใจส่วนไหนของรูปในการตัดสินใจ
        """
        
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("🔥 Grad-CAM Visualization")
        print("="*60)
        
        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # Preprocess (YOLO style)
        img_resized = cv2.resize(image_rgb, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True
        
        # Get target layer (last conv layer)
        target_layer = None
        for name, module in reversed(list(self.model_torch.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                target_layer_name = name
                break
        
        print(f"✓ Target layer: {target_layer_name}")
        
        # Hook to get activations and gradients
        activations = {}
        gradients = {}
        
        def forward_hook(module, input, output):
            activations['value'] = output
        
        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0]
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        output = self.model_torch(img_tensor)
        
        # Get loss (sum of all outputs for simplicity)
        if isinstance(output, (list, tuple)):
            # YOLO returns tuple of tensors
            loss = 0
            for o in output:
                if isinstance(o, torch.Tensor):
                    loss = loss + o.sum()
                elif isinstance(o, (list, tuple)):
                    for item in o:
                        if isinstance(item, torch.Tensor):
                            loss = loss + item.sum()
        else:
            loss = output.sum()
        
        # Backward pass
        self.model_torch.zero_grad()
        loss.backward()
        
        # Get activations and gradients
        acts = activations['value'].detach().cpu()
        grads = gradients['value'].detach().cpu()
        
        # Compute weights (global average pooling of gradients)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU
        
        # Normalize
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to original image size
        cam_resized = cv2.resize(cam, (original_size[1], original_size[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save
        save_path = f'{output_dir}/gradcam_{Path(image_path).stem}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
        print(f"\n✅ Grad-CAM saved to: {output_dir}/")
        
        return cam_resized
    
    # ========================================================================
    # 3. Filter Visualization - ดูว่าแต่ละ filter เรียนรู้ pattern อะไร
    # ========================================================================
    
    def visualize_filters(self, layer_idx=0, output_dir='filters'):
        """
        Visualize learned filters (weights)
        ดูว่าแต่ละ filter เรียนรู้ pattern อะไร (ขอบ, สี, texture, etc.)
        """
        
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("🎨 Visualizing Learned Filters")
        print("="*60)
        
        # Get Conv layers
        conv_layers = []
        for name, module in self.model_torch.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        if layer_idx >= len(conv_layers):
            layer_idx = 0
        
        layer_name, layer = conv_layers[layer_idx]
        weights = layer.weight.data.cpu()
        
        print(f"✓ Layer: {layer_name}")
        print(f"✓ Weight shape: {weights.shape}")
        print(f"  - Filters: {weights.shape[0]}")
        print(f"  - Input channels: {weights.shape[1]}")
        print(f"  - Kernel size: {weights.shape[2]}x{weights.shape[3]}")
        
        # Visualize first layer (usually learns edges, colors, etc.)
        n_filters = min(64, weights.shape[0])
        n_cols = 8
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*2))
        fig.suptitle(f'Learned Filters - Layer: {layer_name}', 
                    fontsize=16, fontweight='bold')
        
        for i in range(n_filters):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Get filter weights
            filt = weights[i]
            
            # If 3 channels (RGB), show as color
            if filt.shape[0] == 3:
                filt_vis = filt.permute(1, 2, 0).numpy()
                # Normalize
                filt_vis = (filt_vis - filt_vis.min()) / (filt_vis.max() - filt_vis.min() + 1e-8)
                ax.imshow(filt_vis)
            else:
                # Show first channel
                filt_vis = filt[0].numpy()
                # Normalize
                filt_vis = (filt_vis - filt_vis.min()) / (filt_vis.max() - filt_vis.min() + 1e-8)
                ax.imshow(filt_vis, cmap='gray')
            
            ax.axis('off')
            ax.set_title(f'F{i}', fontsize=8)
        
        # Hide empty subplots
        for i in range(n_filters, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save
        save_path = f'{output_dir}/filters_layer{layer_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {save_path}")
        print(f"\n✅ Filters saved to: {output_dir}/")
    
    # ========================================================================
    # 4. Feature Importance - วิเคราะห์ว่าฟีเจอร์ไหนสำคัญ
    # ========================================================================
    
    def analyze_feature_importance(self, image_path, output_dir='importance'):
        """
        Analyze which features are important for detection
        """
        
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("📊 Analyzing Feature Importance")
        print("="*60)
        
        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Test different image modifications
        modifications = {
            'Original': image_rgb,
            'No Red': self._remove_channel(image_rgb, 0),
            'No Green': self._remove_channel(image_rgb, 1),
            'No Blue': self._remove_channel(image_rgb, 2),
            'Grayscale': cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY),
            'Edges Only': self._get_edges(image_rgb),
            'Blur': cv2.GaussianBlur(image_rgb, (15, 15), 0),
        }
        
        # Run inference on each
        results_dict = {}
        
        for name, mod_image in modifications.items():
            # Prepare
            if len(mod_image.shape) == 2:
                mod_image = cv2.cvtColor(mod_image, cv2.COLOR_GRAY2RGB)
            
            # Predict
            results = self.model.predict(mod_image, verbose=False)
            
            # Count detections
            n_detections = 0
            avg_confidence = 0
            
            if results and results[0].boxes is not None:
                n_detections = len(results[0].boxes)
                if n_detections > 0:
                    avg_confidence = results[0].boxes.conf.mean().item()
            
            results_dict[name] = {
                'detections': n_detections,
                'confidence': avg_confidence
            }
            
            print(f"  {name:15s}: {n_detections} detections, "
                  f"avg conf: {avg_confidence:.3f}")
        
        # Plot results
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        names = list(results_dict.keys())
        detections = [results_dict[n]['detections'] for n in names]
        confidences = [results_dict[n]['confidence'] for n in names]
        
        # Detections bar chart
        axes[0].bar(names, detections, color='steelblue')
        axes[0].set_title('Number of Detections', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count')
        axes[0].grid(axis='y', alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Confidence bar chart
        axes[1].bar(names, confidences, color='coral')
        axes[1].set_title('Average Confidence', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Confidence')
        axes[1].set_ylim(0, 1.0)
        axes[1].grid(axis='y', alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        save_path = f'{output_dir}/feature_importance.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved: {save_path}")
        print(f"\n✅ Analysis complete!")
        
        return results_dict
    
    # Helper functions
    def _remove_channel(self, image, channel):
        """Remove a color channel"""
        img = image.copy()
        img[:, :, channel] = 0
        return img
    
    def _get_edges(self, image):
        """Get edge map"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def main():
    """
    Main function - ใช้งานทั้งหมด
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize YOLO Model Features')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to model (.pt file)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--output', type=str, default='visualization',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = FeatureVisualizer(args.model)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("🎨 YOLO Feature Visualization")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Output: {args.output}")
    
    # Run all visualizations
    viz.visualize_feature_maps(args.image, str(output_dir / 'feature_maps'))
    viz.visualize_gradcam(args.image, str(output_dir / 'gradcam'))
    viz.visualize_filters(layer_idx=0, output_dir=str(output_dir / 'filters'))
    viz.analyze_feature_importance(args.image, str(output_dir / 'importance'))
    
    print("\n" + "="*60)
    print("✅ All visualizations complete!")
    print(f"📁 Check: {output_dir.absolute()}")
    print("="*60)


if __name__ == '__main__':
    main()