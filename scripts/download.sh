mkdir checkpoints && cd checkpoints
huggingface-cli download --resume-download UmiSonoda16/EchoGen --local-dir .
cd ..