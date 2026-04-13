# Download the DINO-v2 semantic encoder
git clone https://github.com/facebookresearch/dinov2.git

# Download the FLUX.1-dev VAE as the detailed encoder
HF_TOKEN=<your_hf_token>
mkdir pretrained && cd pretrained 
huggingface-cli download --resume-download black-forest-labs/FLUX.1-dev  --include "vae/**" --local-dir . --token ${HF_TOKEN}

# Build the GroundingDINO for subject segmentation
cd ..
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e . --no-build-isolation --no-cache-dir
cd ../pretrained && mkdir groundingdino && cd groundingdino
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download the Infinity-2B pretrained weights together with the Flan-T5-XL text encoder
cd .. && mkdir infinity && cd infinity
huggingface-cli download --resume-download FoundationVision/Infinity --local-dir .
cd .. && mkdir flan-t5-xl && cd flan-t5-xl
huggingface-cli download --resume-download google/flan-t5-xl --local-dir .
cd ../..
