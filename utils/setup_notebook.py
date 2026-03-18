import os
import sys
import torch
import subprocess



def run_cmd(cmd):
    """Run shell command safely"""
    print(f"👉 {cmd}")
    subprocess.check_call(cmd, shell=True)


def setup_paddleocr():
    PADDLE_REPO_DIR = "PaddleOCR"

    # -------------------------
    # Clone PaddleOCR
    # -------------------------
    if not os.path.isdir(PADDLE_REPO_DIR):
        print("📦 Cloning PaddleOCR repository...")
        run_cmd("git clone https://github.com/PaddlePaddle/PaddleOCR.git")
    else:
        print("✅ PaddleOCR repository already exists")

    # -------------------------
    # Install dependencies
    # -------------------------
    print("📦 Installing requirements...")
    run_cmd("pip install -r PaddleOCR/requirements.txt")

    print("📦 Installing PaddleOCR & PaddleX...")
    run_cmd('pip install -U colored paddleocr "paddleocr[all]" paddlex "paddlex[base]"')

    print("📦 Installing paddle2onnx...")
    run_cmd("paddlex --install paddle2onnx")

    # -------------------------
    # Install PaddlePaddle
    # -------------------------
    if torch.cuda.is_available():
        print("🚀 CUDA detected → Installing GPU version")
        run_cmd(
            "pip install paddlepaddle-gpu==3.3.0 "
            "-i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
        )
        run_cmd("pip install onnxruntime-gpu")
    else:
        print("⚠️ No CUDA → Installing CPU version")
        run_cmd(
            "pip install paddlepaddle==3.3.0 "
            "-i https://www.paddlepaddle.org.cn/packages/stable/cpu/"
        )
        run_cmd("pip install onnxruntime")

    print("✅ Setup completed!")


def check_system():
    import paddle
    import cpuinfo

    print("\nPaddle version:", paddle.__version__)
    print("Device:", paddle.device.get_available_device())

    cpu_info = cpuinfo.get_cpu_info()
    print("\nCPU:", cpu_info.get("brand_raw"))
    print("Cores:", cpu_info.get("count"))

    if torch.cuda.is_available():
        print("\nGPU:", torch.cuda.get_device_name(0))
    else:
        print("\nNo GPU found")