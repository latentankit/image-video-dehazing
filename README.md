# 🌫️ AI-Powered Image & Video Dehazing

A production-ready computer vision system that removes haze from images and 
videos using the Dark Channel Prior algorithm with guided filter refinement.

🔗 **Live Demo:** [your-app-url.streamlit.app](https://image-video-dehazing-5xu7gdd2dlxiwwgwe4u5kv.streamlit.app/)

## Results
| Metric | Hazy Input | Dehazed Output | Improvement |
|--------|-----------|----------------|-------------|
| PSNR   | 16.13 dB  | 20.40 dB       | +4.27 dB   |
| SSIM   | 0.8486    | 0.9496         | +0.101     |
*Evaluated on RESIDE SOTS outdoor benchmark dataset*

## Features
- Haze removal from images (JPG, PNG)
- Frame-by-frame video dehazing (MP4, AVI, MOV)
- Before/after comparison viewer
- PSNR & SSIM quality metrics with ground-truth comparison
- Adjustable haze removal strength (omega slider)
- Downloadable dehazed output

## How It Works
Based on the atmospheric scattering model:
**I(x) = J(x)·t(x) + A·(1 − t(x))**

1. **Dark Channel Prior** — estimates haze density per pixel
2. **Atmospheric light estimation** — finds the haze glow color
3. **Transmission map** — calculates how much light survives the haze
4. **Guided filter refinement** — removes halos at object edges
5. **Scene recovery** — solves for the clear image J(x)

## Tech Stack
- Python, PyTorch, OpenCV
- Streamlit (UI + deployment)
- scikit-image (PSNR/SSIM metrics)
- Dataset: RESIDE SOTS benchmark

## Run Locally
```bash
git clone https://github.com/latentankit/image-video-dehazing
cd image-video-dehazing
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```
