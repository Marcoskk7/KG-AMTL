import os
import requests
from tqdm import tqdm

def _safe_extract_path(base_dir: str, member_path: str) -> str:
    base_dir_abs = os.path.abspath(base_dir)
    target_abs = os.path.abspath(os.path.join(base_dir_abs, member_path))
    if os.path.commonpath([base_dir_abs, target_abs]) != base_dir_abs:
        raise RuntimeError(f"检测到不安全的解压路径: {member_path}")
    return target_abs


def extract_archive(archive_path: str, extract_to: str, remove_archive: bool = False) -> str:
    import tarfile
    import zipfile
    import gzip
    import shutil

    os.makedirs(extract_to, exist_ok=True)
    lower = archive_path.lower()

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                _safe_extract_path(extract_to, member.filename)
            zf.extractall(extract_to)
        extracted_to = extract_to
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.name:
                    _safe_extract_path(extract_to, member.name)
            tf.extractall(extract_to)
        extracted_to = extract_to
    elif lower.endswith(".gz"):
        # 兼容单文件 .gz（非 .tar.gz / .tgz）
        out_name = os.path.basename(archive_path[:-3])
        out_path = os.path.join(extract_to, out_name)
        with gzip.open(archive_path, "rb") as fin, open(out_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        extracted_to = out_path
    else:
        raise ValueError(f"不支持的压缩格式: {archive_path}")

    if remove_archive:
        try:
            os.remove(archive_path)
        except OSError:
            pass
    return extracted_to


def download_cwru_dataset(
    url: str = "https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML/releases/download/raw_dataset/CWRU_12k.zip",
    save_dir: str = "./data",
    filename: str = "CWRU_12k.zip",
    extract: bool = True,
    extract_dir=None,
    remove_archive: bool = False,
    timeout: int = 60,
):
    from tqdm import tqdm
    import requests

    os.makedirs(save_dir, exist_ok=True)
    archive_path = os.path.join(save_dir, filename)

    print("正在下载")
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(archive_path, "wb") as f, tqdm(
                total=total_size, unit="iB", unit_scale=True, desc="正在下载"
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        print(f"下载成功: {archive_path}")
    except Exception as e:
        print(f"下载失败: {e}")
        return

    if extract:
        to_dir = extract_dir or save_dir
        print(f"正在解压到: {to_dir}")
        try:
            extracted_to = extract_archive(
                archive_path, extract_to=to_dir, remove_archive=remove_archive
            )
            print(f"解压完成: {extracted_to}")
        except Exception as e:
            print(f"解压失败: {e}")

if __name__ == "__main__":
    download_cwru_dataset()