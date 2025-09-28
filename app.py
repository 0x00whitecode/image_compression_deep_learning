import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import json
import zlib
import io
import base64
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="Deep Image Compression",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}
.compression-stats {
    background-color: #e8f5e8;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4caf50;
}
.mode-selector {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 2px solid #dee2e6;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained models with caching"""
    try:
        autoencoder = load_model("saved_models/autoencoder.h5", compile=False)
        encoder = load_model("saved_models/encoder.h5", compile=False)
        decoder = load_model("saved_models/decoder.h5", compile=False)
        return autoencoder, encoder, decoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Make sure the model files exist in the 'saved_models' directory")
        return None, None, None

def quantize_latent(latents, levels=256):
    """Quantize latent representations"""
    z_min, z_max = latents.min(), latents.max()
    if z_max == z_min:
        z_max = z_min + 1e-6
    scaled = (latents - z_min) / (z_max - z_min) * (levels - 1)
    q = np.round(scaled).astype(np.uint16)
    meta = {"min": float(z_min), "max": float(z_max), "levels": int(levels)}
    return q, meta

def dequantize_latent(q, meta):
    """Dequantize latent representations"""
    levels, z_min, z_max = meta["levels"], meta["min"], meta["max"]
    scaled = q.astype("float32")
    latents = scaled / (levels - 1) * (z_max - z_min) + z_min
    return latents

def compress_image(image, encoder, levels=256):
    """Compress a single image"""
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Encode to latent space
    latents = encoder.predict(image, verbose=0)
    
    # Quantize and compress
    q, meta = quantize_latent(latents, levels=levels)
    meta_bytes = json.dumps(meta).encode("utf-8")
    raw_bytes = q.tobytes()
    blob = meta_bytes + b"||META_RAW||" + raw_bytes
    compressed = zlib.compress(blob)
    
    return compressed, meta, q.shape

def decompress_image(compressed_blob, shape, decoder):
    """Decompress and reconstruct image"""
    decompressed = zlib.decompress(compressed_blob)
    meta_bytes, raw = decompressed.split(b"||META_RAW||", 1)
    meta = json.loads(meta_bytes.decode("utf-8"))
    q = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    latents = dequantize_latent(q, meta)
    reconstructed = decoder.predict(latents, verbose=0)
    return reconstructed[0], meta

def save_compressed_file(compressed_blob, filename):
    """Create downloadable compressed file"""
    # Encode as base64 for safe storage
    encoded = base64.b64encode(compressed_blob).decode('utf-8')
    return encoded

def load_compressed_file(encoded_data):
    """Load compressed file from base64 encoding"""
    try:
        compressed_blob = base64.b64decode(encoded_data.encode('utf-8'))
        return compressed_blob
    except Exception as e:
        st.error(f"Error loading compressed file: {str(e)}")
        return None

def compute_metrics(original, reconstructed):
    """Compute quality metrics"""
    original = np.expand_dims(original, axis=0) if len(original.shape) == 3 else original
    reconstructed = np.expand_dims(reconstructed, axis=0) if len(reconstructed.shape) == 3 else reconstructed
    
    mse = np.mean((original - reconstructed) ** 2)
    psnr = tf.image.psnr(original, reconstructed, max_val=1.0).numpy().mean()
    ssim = tf.image.ssim(original, reconstructed, max_val=1.0).numpy().mean()
    
    return {"mse": float(mse), "psnr": float(psnr), "ssim": float(ssim)}

def preprocess_image(uploaded_file, target_size=(32, 32)):
    """Preprocess uploaded image"""
    image = Image.open(uploaded_file)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to target size
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    
    return img_array

def create_comparison_plot(original, reconstructed, error_map):
    """Create comparison visualization"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Original", "Reconstructed", "Error Map"],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
    )
    
    # Original image
    fig.add_trace(
        go.Heatmap(z=original[::-1], colorscale='viridis', showscale=False),
        row=1, col=1
    )
    
    # Reconstructed image
    fig.add_trace(
        go.Heatmap(z=reconstructed[::-1], colorscale='viridis', showscale=False),
        row=1, col=2
    )
    
    # Error map
    fig.add_trace(
        go.Heatmap(z=error_map[::-1], colorscale='hot', showscale=True),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig

def get_latent_dimensions(encoder):
    """Safely get latent dimensions from encoder output"""
    try:
        # Handle different types of encoder outputs
        if hasattr(encoder, 'output'):
            output = encoder.output
            
            # If output is a list (multiple outputs), get the first one
            if isinstance(output, list):
                if len(output) > 0:
                    return output[0].shape[-1] if hasattr(output[0], 'shape') else "Unknown"
                else:
                    return "Unknown"
            
            # If output is a single tensor
            elif hasattr(output, 'shape'):
                return output.shape[-1]
            else:
                return "Unknown"
        
        # Alternative: try to get from output layer
        elif hasattr(encoder, 'layers') and len(encoder.layers) > 0:
            last_layer = encoder.layers[-1]
            if hasattr(last_layer, 'output_shape'):
                shape = last_layer.output_shape
                if isinstance(shape, tuple) and len(shape) > 0:
                    return shape[-1]
                elif isinstance(shape, list) and len(shape) > 0:
                    return shape[0][-1] if isinstance(shape[0], tuple) else "Unknown"
        
        return "Unknown"
        
    except Exception as e:
        st.warning(f"Could not determine latent dimensions: {str(e)}")
        return "Unknown"

def compression_mode(encoder, decoder):
    """Handle compression mode interface"""
    st.header("📦 Compression Mode")
    st.markdown("Upload an image to compress it using the trained autoencoder")
    
    # Compression settings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file to compress",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a PNG or JPEG image. It will be resized to 32x32 pixels.",
            key="compression_upload"
        )
    
    with col2:
        quantization_levels = st.selectbox(
            "Quantization Levels",
            [16, 32, 64, 128, 256, 512, 1024],
            index=4,  # Default to 256
            help="Higher levels preserve more quality but reduce compression"
        )
    
    if uploaded_file is not None:
        # Preprocess image
        original_img = preprocess_image(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image (32x32)")
            fig_orig = px.imshow(original_img, aspect='equal')
            fig_orig.update_layout(height=300, coloraxis_showscale=False)
            fig_orig.update_xaxes(showticklabels=False)
            fig_orig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_orig, use_container_width=True)
            
            # Original image stats
            original_size = original_img.nbytes
            st.info(f"**Original Size:** {original_size} bytes ({original_size/1024:.1f} KB)")
        
        with col2:
            # Compress image
            with st.spinner("Compressing image..."):
                compressed_blob, meta, shape = compress_image(original_img, encoder, quantization_levels)
                reconstructed_img, _ = decompress_image(compressed_blob, shape, decoder)
            
            st.subheader("Reconstructed Image")
            fig_recon = px.imshow(reconstructed_img, aspect='equal')
            fig_recon.update_layout(height=300, coloraxis_showscale=False)
            fig_recon.update_xaxes(showticklabels=False)
            fig_recon.update_yaxes(showticklabels=False)
            st.plotly_chart(fig_recon, use_container_width=True)
            
            # Compression stats
            compressed_size = len(compressed_blob)
            compression_ratio = original_size / compressed_size
            st.success(f"**Compressed Size:** {compressed_size} bytes ({compressed_size/1024:.1f} KB)")
            st.success(f"**Compression Ratio:** {compression_ratio:.1f}x")
        
        # Quality metrics
        metrics = compute_metrics(original_img, reconstructed_img)
        
        st.markdown("### 📊 Quality Metrics")
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("PSNR", f"{metrics['psnr']:.1f} dB")
        with metric_cols[1]:
            st.metric("SSIM", f"{metrics['ssim']:.3f}")
        with metric_cols[2]:
            st.metric("MSE", f"{metrics['mse']:.6f}")
        
        # Download compressed file
        st.markdown("### 💾 Download Compressed File")
        encoded_data = save_compressed_file(compressed_blob, uploaded_file.name)
        
        # Create download data with metadata
        download_data = {
            "compressed_data": encoded_data,
            "shape": shape,
            "original_filename": uploaded_file.name,
            "quantization_levels": quantization_levels,
            "compression_ratio": compression_ratio,
            "metrics": metrics
        }
        
        download_json = json.dumps(download_data, indent=2)
        
        st.download_button(
            label="📥 Download Compressed File",
            data=download_json,
            file_name=f"compressed_{uploaded_file.name.split('.')[0]}.json",
            mime="application/json",
            help="Download the compressed image data as a JSON file"
        )
        
        # Store results for advanced analysis
        st.session_state.compression_results = {
            'original_img': original_img,
            'reconstructed_img': reconstructed_img,
            'compressed_blob': compressed_blob,
            'compression_meta': meta,
            'metrics': metrics
        }

def decompression_mode(decoder):
    """Handle decompression mode interface"""
    st.header("📂 Decompression Mode")
    st.markdown("Upload a compressed file to decompress and view the reconstructed image")
    
    uploaded_compressed = st.file_uploader(
        "Choose a compressed file (.json)",
        type=['json'],
        help="Upload a JSON file created by the compression mode",
        key="decompression_upload"
    )
    
    if uploaded_compressed is not None:
        try:
            # Load compressed data
            compressed_data = json.load(uploaded_compressed)
            
            # Extract information
            encoded_data = compressed_data["compressed_data"]
            shape = compressed_data["shape"]
            original_filename = compressed_data.get("original_filename", "unknown.jpg")
            quantization_levels = compressed_data.get("quantization_levels", "Unknown")
            stored_compression_ratio = compressed_data.get("compression_ratio", "Unknown")
            stored_metrics = compressed_data.get("metrics", {})
            
            # Decompress
            compressed_blob = load_compressed_file(encoded_data)
            
            if compressed_blob is not None:
                with st.spinner("Decompressing image..."):
                    reconstructed_img, meta = decompress_image(compressed_blob, shape, decoder)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Reconstructed Image")
                    fig_recon = px.imshow(reconstructed_img, aspect='equal')
                    fig_recon.update_layout(height=400, coloraxis_showscale=False)
                    fig_recon.update_xaxes(showticklabels=False)
                    fig_recon.update_yaxes(showticklabels=False)
                    st.plotly_chart(fig_recon, use_container_width=True)
                
                with col2:
                    st.subheader("File Information")
                    st.json({
                        "Original Filename": original_filename,
                        "Quantization Levels": quantization_levels,
                        "Compression Ratio": f"{stored_compression_ratio}x" if stored_compression_ratio != "Unknown" else "Unknown",
                        "Compressed Size": f"{len(compressed_blob)} bytes",
                        "Shape": shape,
                        "Quantization Range": f"{meta['min']:.3f} - {meta['max']:.3f}"
                    })
                    
                    # Display stored metrics if available
                    if stored_metrics:
                        st.subheader("Quality Metrics")
                        st.json({
                            "PSNR": f"{stored_metrics.get('psnr', 'N/A'):.1f} dB" if 'psnr' in stored_metrics else "N/A",
                            "SSIM": f"{stored_metrics.get('ssim', 'N/A'):.3f}" if 'ssim' in stored_metrics else "N/A",
                            "MSE": f"{stored_metrics.get('mse', 'N/A'):.6f}" if 'mse' in stored_metrics else "N/A"
                        })
                
                # Convert back to PIL Image for download
                img_for_download = (reconstructed_img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_for_download)
                
                # Create download buffer
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.markdown("### 💾 Download Reconstructed Image")
                st.download_button(
                    label="📥 Download Reconstructed Image",
                    data=img_buffer.getvalue(),
                    file_name=f"reconstructed_{original_filename.split('.')[0]}.png",
                    mime="image/png",
                    help="Download the reconstructed image as a PNG file"
                )
                
                # Store for advanced analysis
                st.session_state.decompression_results = {
                    'reconstructed_img': reconstructed_img,
                    'compressed_blob': compressed_blob,
                    'meta': meta,
                    'stored_metrics': stored_metrics
                }
                
        except Exception as e:
            st.error(f"Error loading compressed file: {str(e)}")
            st.info("Please make sure you're uploading a valid compressed file created by this application.")

def advanced_analysis_section():
    """Display advanced analysis for both modes"""
    if 'compression_results' in st.session_state or 'decompression_results' in st.session_state:
        st.markdown("---")
        st.header("🔬 Advanced Analysis")
        
        # Determine which results to use
        if 'compression_results' in st.session_state:
            results = st.session_state.compression_results
            original_img = results.get('original_img')
            reconstructed_img = results['reconstructed_img']
            metrics = results['metrics']
            has_original = True
        else:
            results = st.session_state.decompression_results
            original_img = None
            reconstructed_img = results['reconstructed_img']
            metrics = results.get('stored_metrics', {})
            has_original = False
        
        if has_original and original_img is not None:
            # Full comparison with original
            error_map = np.abs(original_img - reconstructed_img)
            error_map_gray = np.mean(error_map, axis=2)
            
            st.subheader("Visual Comparison")
            original_gray = np.mean(original_img, axis=2)
            reconstructed_gray = np.mean(reconstructed_img, axis=2)
            
            comparison_fig = create_comparison_plot(original_gray, reconstructed_gray, error_map_gray)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Error statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Quality Metrics")
                st.json({
                    "MSE": metrics['mse'],
                    "PSNR": metrics['psnr'],
                    "SSIM": metrics['ssim'],
                    "Max Error": float(np.max(error_map)),
                    "Mean Error": float(np.mean(error_map)),
                    "Error Std": float(np.std(error_map))
                })
            
            with col2:
                st.subheader("Image Statistics")
                st.json({
                    "Original Range": f"{np.min(original_img):.3f} - {np.max(original_img):.3f}",
                    "Reconstructed Range": f"{np.min(reconstructed_img):.3f} - {np.max(reconstructed_img):.3f}",
                    "Original Mean": f"{np.mean(original_img):.3f}",
                    "Reconstructed Mean": f"{np.mean(reconstructed_img):.3f}",
                    "Original Std": f"{np.std(original_img):.3f}",
                    "Reconstructed Std": f"{np.std(reconstructed_img):.3f}"
                })
            
            # Error distribution
            st.subheader("Error Distribution")
            fig_hist = px.histogram(
                x=error_map.flatten(),
                nbins=50,
                title="Distribution of Reconstruction Errors",
                labels={'x': 'Absolute Error', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        else:
            # Decompression mode - show only reconstructed image analysis
            st.subheader("Reconstructed Image Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if metrics:
                    st.subheader("Stored Quality Metrics")
                    st.json(metrics)
                else:
                    st.info("No quality metrics available (original image not provided)")
            
            with col2:
                st.subheader("Image Statistics")
                st.json({
                    "Reconstructed Range": f"{np.min(reconstructed_img):.3f} - {np.max(reconstructed_img):.3f}",
                    "Mean": f"{np.mean(reconstructed_img):.3f}",
                    "Standard Deviation": f"{np.std(reconstructed_img):.3f}",
                    "Shape": list(reconstructed_img.shape)
                })
            
            # Pixel value distribution
            st.subheader("Pixel Value Distribution")
            fig_hist = px.histogram(
                x=reconstructed_img.flatten(),
                nbins=50,
                title="Distribution of Pixel Values",
                labels={'x': 'Pixel Value', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

def main():
    st.title("🖼️ Deep Image Compression")
    st.markdown("Compress and decompress images using a trained convolutional autoencoder")
    
    # Load models
    autoencoder, encoder, decoder = load_models()
    
    if encoder is None or decoder is None:
        st.stop()
    
    # Mode selection
    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    mode = st.radio(
        "Choose Mode:",
        ["🔄 Compression & Decompression", "📦 Compression Only", "📂 Decompression Only"],
        index=0,
        horizontal=True,
        help="Select whether you want to compress images, decompress files, or do both"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content based on mode
    if mode == "🔄 Compression & Decompression":
        # Show both modes in tabs
        tab1, tab2 = st.tabs(["📦 Compress Image", "📂 Decompress File"])
        
        with tab1:
            compression_mode(encoder, decoder)
        
        with tab2:
            decompression_mode(decoder)
    
    elif mode == "📦 Compression Only":
        compression_mode(encoder, decoder)
    
    else:  # Decompression Only
        decompression_mode(decoder)
    
    # Advanced analysis (shown for both modes if data is available)
    if st.sidebar.checkbox("Show Advanced Analysis", value=False):
        advanced_analysis_section()
    
    # Batch processing section (only for compression mode)
    if mode in ["🔄 Compression & Decompression", "📦 Compression Only"]:
        st.markdown("---")
        st.header("📁 Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images for batch compression",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple images to compare compression performance",
            key="batch_upload"
        )
        
        if uploaded_files:
            batch_quantization = st.selectbox(
                "Batch Quantization Levels",
                [16, 32, 64, 128, 256, 512, 1024],
                index=4,
                key="batch_quantization"
            )
            
            if st.button("Process Batch"):
                batch_results = []
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    # Process each image
                    img = preprocess_image(file)
                    compressed, meta, shape = compress_image(img, encoder, batch_quantization)
                    reconstructed, _ = decompress_image(compressed, shape, decoder)
                    metrics = compute_metrics(img, reconstructed)
                    
                    batch_results.append({
                        'filename': file.name,
                        'compression_ratio': img.nbytes / len(compressed),
                        'psnr': metrics['psnr'],
                        'ssim': metrics['ssim'],
                        'mse': metrics['mse'],
                        'original_size': img.nbytes,
                        'compressed_size': len(compressed)
                    })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display batch results
                df_results = pd.DataFrame(batch_results)
                st.subheader("Batch Processing Results")
                st.dataframe(df_results, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Compression Ratio", f"{df_results['compression_ratio'].mean():.1f}x")
                with col2:
                    st.metric("Average PSNR", f"{df_results['psnr'].mean():.1f} dB")
                with col3:
                    st.metric("Average SSIM", f"{df_results['ssim'].mean():.3f}")
                
                # Batch visualization
                fig_batch = px.scatter(
                    df_results, 
                    x='compression_ratio', 
                    y='psnr',
                    size='original_size',
                    hover_data=['filename', 'ssim'],
                    title='Compression Ratio vs Quality',
                    labels={'compression_ratio': 'Compression Ratio', 'psnr': 'PSNR (dB)'}
                )
                st.plotly_chart(fig_batch, use_container_width=True)
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.header("Model Information")
    
    if encoder is not None:
        # Get model info safely
        latent_dim = get_latent_dimensions(encoder)
        st.sidebar.info(f"**Latent Dimensions:** {latent_dim}")
        st.sidebar.info(f"**Input Shape:** 32×32×3")
        st.sidebar.info(f"**Architecture:** Convolutional Autoencoder")
        
        # Performance tips
        with st.sidebar.expander("💡 Performance Tips"):
            st.markdown("""
            **Compression Tips:**
            - **Low quantization** (16-64 levels): Higher compression, lower quality
            - **High quantization** (512-1024 levels): Lower compression, higher quality
            - **256 levels**: Good balance for most images
            - Images with smooth gradients compress better
            - High-detail images may show more artifacts
            
            **File Handling:**
            - Compressed files are saved as JSON with metadata
            - Files can be shared and decompressed later
            - Original filenames are preserved in compressed files
            """)
    
    # About section
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        This app uses a trained convolutional autoencoder for image compression:
        
        **Compression Process:**
        1. **Encoding**: Convert image to latent representation
        2. **Quantization**: Reduce precision of latent values
        3. **Compression**: Apply zlib compression
        4. **Storage**: Save as downloadable JSON file
        
        **Decompression Process:**
        1. **Loading**: Read compressed JSON file
        2. **Decompression**: Reverse zlib compression
        3. **Dequantization**: Restore latent precision
        4. **Decoding**: Reconstruct the image
        
        The model works best with 32×32 pixel images and was trained on CIFAR-10 dataset.
        """)

if __name__ == "__main__":
    main()