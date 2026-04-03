import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup the environment')
    
    parser.add_argument('--cuda_version', type=str, default='11.8', help='CUDA version to use', choices=['11.8', '12.1'])
    args = parser.parse_args()
    
    print(f"[INFO] Installing environment...")
    
    # Install torch
    print(f"[INFO] Installing torch...")
    os.system(f"conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda={args.cuda_version} mkl=2023.1.0 -c pytorch -c nvidia")
    print(f"[INFO] Torch installed.")
    
    # Install requirements
    print(f"[INFO] Installing requirements...")
    os.system(f"pip install -r requirements.txt")
    print(f"[INFO] Requirements installed.")
    
    # Install submodules
    print(f"[INFO] Installing Mini-Splatting2 rasterizer...")
    os.system(f"pip install submodules/diff-gaussian-rasterization_ms")
    print("[INFO] Mini-Splatting2 rasterizer installed.")

    print(f"[INFO] Installing RaDe-GS rasterizer...")
    os.system(f"pip install submodules/diff-gaussian-rasterization")
    print("[INFO] RaDe-GS rasterizer installed.")
    
    print(f"[INFO] Installing GOF rasterizer...")
    os.system(f"pip install submodules/diff-gaussian-rasterization_gof")
    print("[INFO] GOF rasterizer installed.")
    
    print(f"[INFO] Installing Simple KNN...")
    os.system(f"pip install submodules/simple-knn")
    print("[INFO] Simple KNN installed.")
    
    print(f"[INFO] Installing Fused SSIM...")
    os.system(f"pip install submodules/fused-ssim")
    print("[INFO] Fused SSIM installed.")
    
    print(f"[INFO] Installing Triangulation...")
    os.chdir("submodules/tetra_triangulation/")
    os.system(f"conda install -y cmake")
    os.system(f"conda install -y conda-forge::gmp")
    os.system(f"conda install -y conda-forge::cgal")
    # WARNING: CUDA paths must be set before running cmake
    os.system(f"cmake .")
    os.system(f"make")
    os.system(f"pip install -e .")
    os.chdir("../../")
    print("[INFO] Triangulation installed.")
    
    print(f"[INFO] Installing Nvdiffrast...")
    os.chdir("submodules/nvdiffrast/")
    os.system(f"pip install -e .")
    os.chdir("../../")
    print("[INFO] Nvdiffrast installed.")
    
    print(f"[INFO] Installation complete.")
    