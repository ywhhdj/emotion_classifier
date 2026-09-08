import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional

from .models.config import ModelConfig, ModelRepository


class AsyncModelDownloader:
    """模型文件下载器（支持断点续传 + SHA256 校验）。"""

    def __init__(self, model_dir: Path, timeout: float = 60.0, max_concurrency: int = 2):
        self.model_dir = Path(model_dir)
        self.timeout = timeout
        self.max_concurrency = max(1, int(max_concurrency))

    async def _download_file(
        self,
        client,
        url: str,
        dest: Path,
        repo: ModelRepository,
        progress_callback: Optional[Callable] = None,
        sem: Optional[asyncio.Semaphore] = None,
    ):
        if dest.exists():
            return dest

        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        try:
            if sem is not None:
                async with sem:
                    return await self._stream(client, url, dest, tmp_path, repo, progress_callback)
            return await self._stream(client, url, dest, tmp_path, repo, progress_callback)
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"下载 {dest} 失败: {e}") from e

    async def _stream(self, client, url, dest, tmp_path, repo, progress_callback):
        sys.stderr.write(f"\r  ↓ 下载 {dest.name} 从 {url}\n")
        headers = {}
        resume = tmp_path.stat().st_size if tmp_path.exists() else 0
        if resume:
            headers["Range"] = f"bytes={resume}-"

        async with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
            # 206 = 断点续传，200 = 服务端忽略 Range（需从头写）
            if resume and response.status_code != 206:
                resume = 0
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0) or 0)
            if "Content-Range" in response.headers:
                try:
                    total = int(response.headers["Content-Range"].split("/")[1])
                except (IndexError, ValueError):
                    pass
            elif resume:
                total += resume

            mode = "ab" if resume and response.status_code == 206 else "wb"
            downloaded = resume if mode == "ab" else 0
            chunk_size = 256 * 1024
            with open(tmp_path, mode) as f:
                async for chunk in response.aiter_bytes(chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = min(100.0, downloaded / total * 100)
                        bar_len = int(pct / 5)
                        bar = "█" * bar_len + "░" * (20 - bar_len)
                        sys.stderr.write(f"\r  ↓ {dest.name:30s} [{bar}] {pct:5.1f}%")
                        sys.stderr.flush()
                    if progress_callback:
                        progress_callback(dest.name, downloaded, total)
        sys.stderr.write("\n")
        os.replace(tmp_path, dest)

        if not repo.verify_file(dest.name, use_cache=False):
            dest.unlink(missing_ok=True)  # 关键：坏文件必须删除，否则永久残留
            raise ValueError(
                f"{dest} 校验失败: {dest.name} 与 {repo.download_url(dest.name)} 不一致"
            )
        return dest

    async def download(
        self,
        files: List[str],
        repo: ModelRepository,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[Path]:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        paths = [self.model_dir / f for f in files]

        to_download = []
        for filename in files:
            dest = self.model_dir / filename
            if dest.exists() and repo.verify_file(filename):
                continue
            to_download.append((repo.download_url(filename), dest))

        if not to_download:
            return paths

        print(f"  ⏬ 需要下载 {len(to_download)} 个文件:")
        for _, d in to_download:
            print(f"     • {d.name}")

        import httpx
        sem = asyncio.Semaphore(self.max_concurrency)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            await asyncio.gather(*[
                self._download_file(client, url, dest, repo, progress_callback, sem)
                for url, dest in to_download
            ])
        print(f"  ✅ 下载完成 → {self.model_dir}")
        return paths

    @staticmethod
    def download_sync(filenames: List[str], model_dir: Path, repo: ModelRepository) -> List[Path]:
        downloader = AsyncModelDownloader(model_dir)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(downloader.download(filenames, repo))

        # 已在事件循环中：开一个线程跑新的 loop，避免 run_coroutine_threadsafe 自锁
        box: dict = {}

        def runner():
            try:
                box["value"] = asyncio.run(downloader.download(filenames, repo))
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join()
        if "error" in box:
            raise box["error"]
        return box.get("value", [])