from pathlib import Path
import shutil


def main() -> None:
    src = Path(r"C:\Users\shiro\OneDrive\Desktop\Presentation\kaggle (2).json")
    dst_dir = Path.home() / ".kaggle"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "kaggle.json"
    shutil.copyfile(src, dst)
    print(f"Copied Kaggle credentials to: {dst}")


if __name__ == "__main__":
    main()

