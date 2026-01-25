"""
API 连通性测试脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config


def test_qwen_api():
    """测试 Qwen API"""
    print("\n" + "=" * 50)
    print("测试 Qwen API (OpenAI 兼容模式)")
    print("=" * 50)
    
    if not config.QWEN_API_KEY:
        print("✗ 未配置 QWEN_API_KEY")
        return False
    
    print(f"API Key: {config.QWEN_API_KEY[:8]}...{config.QWEN_API_KEY[-4:]}")
    print(f"Base URL: {config.QWEN_BASE_URL}")
    print(f"Model: {config.DEFAULT_MODEL}")
    
    try:
        from src.llm_client import LLMClient
        
        client = LLMClient()
        print("\n发送测试请求...")
        
        if client.test_connection():
            print("✓ API 连接成功!")
            return True
        else:
            print("✗ API 连接失败")
            return False
            
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def test_sam_model():
    """测试 SAM 模型"""
    print("\n" + "=" * 50)
    print("测试 SAM 模型")
    print("=" * 50)
    
    checkpoint_path = config.sam_checkpoint_path
    print(f"模型类型: {config.SAM_MODEL_TYPE}")
    print(f"权重路径: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        print("✗ 模型权重不存在")
        print(f"  请运行: python scripts/download_sam.py")
        return False
    
    print(f"✓ 模型权重存在 ({checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    try:
        import torch
        print(f"\nPyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB")
        
        print("\n加载 SAM 模型...")
        from src.sam_processor import SAMProcessor
        processor = SAMProcessor()
        processor.load_model()
        print("✓ SAM 模型加载成功!")
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def test_directories():
    """测试目录结构"""
    print("\n" + "=" * 50)
    print("测试目录结构")
    print("=" * 50)
    
    dirs = [
        ("数据目录", config.DATA_DIR),
        ("原始图像", config.RAW_DIR),
        ("标注数据", config.ANNOTATED_DIR),
        ("向量索引", config.EMBEDDINGS_DIR),
        ("模型目录", config.SAM_DIR),
        ("输出目录", config.OUTPUT_DIR),
    ]
    
    all_ok = True
    for name, path in dirs:
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (不存在)")
            all_ok = False
    
    return all_ok


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="API 和环境测试")
    parser.add_argument("--qwen", action="store_true", help="测试 Qwen API")
    parser.add_argument("--sam", action="store_true", help="测试 SAM 模型")
    parser.add_argument("--dirs", action="store_true", help="测试目录结构")
    parser.add_argument("--all", "-a", action="store_true", help="运行所有测试")
    
    args = parser.parse_args()
    
    # 如果没有指定任何测试，默认运行所有
    if not any([args.qwen, args.sam, args.dirs, args.all]):
        args.all = True
    
    results = {}
    
    if args.dirs or args.all:
        results["directories"] = test_directories()
    
    if args.qwen or args.all:
        results["qwen_api"] = test_qwen_api()
    
    if args.sam or args.all:
        results["sam_model"] = test_sam_model()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠ 部分测试失败，请检查上述错误信息")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
