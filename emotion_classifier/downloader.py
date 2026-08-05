import asyncio
import os
from pathlib import Path
import sys
from typing import Callable, List, Optional
from .models.config import ModelConfig, ModelRepository

class AsyncModelDownloader:
    def __init__(self, model_dir: Path, timeout: float = 60.0):
        self.model_dir = Path(model_dir)
        self.timeout = timeout

    async def _download_file(
        self,
        client,
        url: str,
        dest: Path,
        repo: ModelRepository,
        progress_callback: Optional[Callable] = None
    ):
        print(f"\r  ↓ 下载 {dest} 从 {url}", file=sys.stderr)
        if dest.exists():
            return dest

        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        try:
            headers = {}
            if tmp_path.exists():
                headers["Range"] = f"bytes={tmp_path.stat().st_size}-"
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                #断点续传
                if "Content-Range" in response.headers:
                    total = int(response.headers["Content-Range"].split("/")[1])
                elif tmp_path.exists():
                    total += tmp_path.stat().st_size
                mode = "ab" if tmp_path.exists() else "wb"
                downloaded = tmp_path.stat().st_size if tmp_path.exists() else 0
                chunk_size = 64 * 1024  # 64KB
                with open(tmp_path, mode) as f:
                    async for chunk in response.aiter_bytes(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            bar_len = int(pct / 5)
                            bar = "█" * bar_len + "░" * (20 - bar_len)
                            sys.stderr.write(
                                f"\r  ↓ {dest.name:30s} [{bar}] {pct:5.1f}%"
                            )
                            sys.stderr.flush()
            sys.stderr.write("\n")
            os.replace(tmp_path, dest)
            # SHA256 校验
            if not repo.verify_file(dest.name):
                raise ValueError(
                    f"{dest} 校验失败: "
                    f"{dest.name} 与 {repo.download_url(dest.name)} 不一致"
                )
            return dest
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"下载 {dest} 失败: {e}") from e


    async def download(
        self,
        files: List[str],
        repo: ModelRepository,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[Path]:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        paths = [self.model_dir / f for f in files]
        to_download = []
        for filename in files:
            dest = self.model_dir / filename
            url = repo.download_url(filename)
            # 如果文件已存在且校验通过，跳过
            if dest.exists() and repo.verify_file(filename):
                continue
            to_download.append((url, dest))
        if not to_download:
            return paths
        print(f"  ⏬ 需要下载 {len(to_download)} 个文件:")
        for f in to_download:
            print(f"     • {f}")
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            await asyncio.gather(*[
                self._download_file(client, url, dest, repo, progress_callback)
                for url, dest in to_download
            ])
        print(f"  ✅ 下载完成 → {self.model_dir}")
        return paths

    @staticmethod
    def download_sync(filenames: List[str], model_dir: Path, repo: ModelRepository) -> List[Path]:
        downloader = AsyncModelDownloader(model_dir)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(downloader.download(filenames, repo))
        else:
            future = asyncio.run_coroutine_threadsafe(
                downloader.download(filenames, repo), loop
            )
            return future.result()