"""
下载 SAM 模型权重
"""

import urllib.request
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config

# SAM 模型下载 URL
SAM_MODELS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "size": "2.4GB"
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size": "1.2GB"
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size": "375MB"
    }
}


def download_with_progress(url: str, filepath: Path, desc: str = "Downloading"):
    """带进度条的下载"""
    
    def progress_hook(count, block_size, total_size):
        percent = count * block_size * 100 / total_size
        downloaded = count * block_size / (1024 * 1024)
        total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r{desc}: {percent:.1f}% ({downloaded:.1f}MB / {total:.1f}MB)")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filepath, progress_hook)
    print()  # 换行


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 SAM 模型")
    parser.add_argument(
        "--model", "-m",
        choices=["vit_h", "vit_l", "vit_b"],
        default=config.SAM_MODEL_TYPE,
        help="模型类型 (默认: vit_h)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新下载"
    )
    
    args = parser.parse_args()
    
    model_info = SAM_MODELS[args.model]
    output_path = config.SAM_DIR / model_info["filename"]
    
    print(f"SAM 模型: {args.model}")
    print(f"文件大小: {model_info['size']}")
    print(f"保存路径: {output_path}")
    
    if output_path.exists() and not args.force:
        print(f"\n✓ 模型已存在，跳过下载")
        print(f"  如需重新下载，请使用 --force 参数")
        return
    
    # 创建目录
    config.SAM_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始下载...")
    print(f"URL: {model_info['url']}")
    
    try:
        download_with_progress(model_info["url"], output_path, "下载进度")
        print(f"\n✓ 下载完成: {output_path}")
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n手动下载方式:")
        print(f"  1. 访问 {model_info['url']}")
        print(f"  2. 将文件保存到 {output_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
