import sys
import os
import tempfile
import cv2
import numpy as np
import streamlit as st

# Allow importing from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dehaze_image import dehaze_image
from models.dehaze_video import dehaze_video
from utils.metrics import calculate_psnr, calculate_ssim, resize_to_match

# ---------------- Page config ----------------
st.set_page_config(page_title="AI Dehazing", page_icon="🌫️", layout="wide")

st.title("🌫️ AI-Powered Image & Video Dehazing")
st.write("Remove haze from images and videos using a Dark Channel Prior pipeline "
         "with guided-filter refinement.")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")
omega = st.sidebar.slider("Haze removal strength (omega)", 0.70, 0.95, 0.85, 0.01)
st.sidebar.caption("Lower = gentler · Higher = more aggressive")

def read_image(file):
    """Decode an uploaded file into an OpenCV BGR image."""
    file_bytes = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ---------------- Tabs ----------------
image_tab, video_tab = st.tabs(["🖼️ Image", "🎬 Video"])

# ========== IMAGE TAB ==========
with image_tab:
    uploaded = st.file_uploader("Upload a hazy image", type=["jpg", "jpeg", "png"], key="img")
    gt_uploaded = st.file_uploader(
        "Optional: ground-truth (clear) image for PSNR / SSIM",
        type=["jpg", "jpeg", "png"], key="gt"
    )

    if uploaded is not None:
        hazy = read_image(uploaded)

        with st.spinner("Dehazing image..."):
            result = dehaze_image(hazy, omega=omega)

        hazy_rgb = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before (Hazy)")
            st.image(hazy_rgb, width='stretch')
        with col2:
            st.subheader("After (Dehazed)")
            st.image(result_rgb, width='stretch')

        if gt_uploaded is not None:
            clear = read_image(gt_uploaded)
            hazy_m = resize_to_match(hazy, clear)
            result_m = resize_to_match(result, clear)

            psnr_hazy = calculate_psnr(clear, hazy_m)
            psnr_dehazed = calculate_psnr(clear, result_m)
            ssim_hazy = calculate_ssim(clear, hazy_m)
            ssim_dehazed = calculate_ssim(clear, result_m)

            st.subheader("▪ Quality Metrics (vs. ground truth)")
            m1, m2 = st.columns(2)
            m1.metric("PSNR (dB)", f"{psnr_dehazed:.2f}",
                      delta=f"{psnr_dehazed - psnr_hazy:+.2f} vs hazy")
            m2.metric("SSIM", f"{ssim_dehazed:.4f}",
                      delta=f"{ssim_dehazed - ssim_hazy:+.4f} vs hazy")

        result_encoded = cv2.imencode(".png", result)[1].tobytes()
        st.download_button("⬇️ Download dehazed image", result_encoded,
                           file_name="dehazed.png", mime="image/png")
    else:
        st.info("👆 Upload an image to get started.")

# ========== VIDEO TAB ==========
with video_tab:
    video_file = st.file_uploader("Upload a hazy video", type=["mp4", "mov", "avi"], key="vid")

    if video_file is not None:
        # Save the upload to a temp file (OpenCV needs a path)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        input_path = tfile.name
        output_path = input_path.replace(".mp4", "_dehazed.mp4")

        st.write("Original video preview:")
        st.video(input_path)

        if st.button("→ Dehaze Video"):
            progress_bar = st.progress(0.0, text="Dehazing video...")

            def update(fraction):
                progress_bar.progress(min(fraction, 1.0),
                                      text=f"Dehazing video... {fraction*100:.0f}%")

            dehaze_video(input_path, output_path, omega=omega, progress_callback=update)
            progress_bar.progress(1.0, text="Done!")

            st.success("✓ Video dehazed successfully!")
            st.write("Dehazed result:")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download dehazed video", f.read(),
                                   file_name="dehazed_video.mp4", mime="video/mp4")
    else:
        st.info("👆 Upload a video to get started.")
