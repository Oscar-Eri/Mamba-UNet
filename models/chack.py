import torch
import torchvision
import platform
import sys
import os

def print_env_info():
    print("Python 版本:", sys.version)
    print("操作系统:", platform.system(), platform.release())
    print("CPU 架构:", platform.machine())
    print("Python 可执行路径:", sys.executable)
    print("当前工作目录:", os.getcwd())
    
    print("\n--- PyTorch 环境 ---")
    print("PyTorch 版本:", torch.__version__)
    print("Torchvision 版本:", torchvision.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA 版本:", torch.version.cuda)
        print("cuDNN 版本:", torch.backends.cudnn.version())
        print("GPU 数量:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} 名称:", torch.cuda.get_device_name(i))
    else:
        print("未检测到可用的 GPU")

if __name__ == "__main__":
    print_env_info()