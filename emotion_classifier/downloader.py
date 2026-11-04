"""Release 资产下载器（v0.2.0）。

* 断点续传 + SHA256 校验（checksums.json 优先）；
* 支持子路径资产（如 tokenizer/tokenizer.json）与 tokenizer.zip 自动解压；
* 可选资产（calib.json / deploy_manifest.json / checksums.json）失败不阻断；
* 已在事件循环中时开线程跑新 loop，避免 run_coroutine_threadsafe 自锁。
"""

import asyncio
import os
import sys
import threading
import zipfile
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .models.config import ModelRepository

_print_lock = threading.Lock()


class AsyncModelDownloader:
    """模型文件下载器（支持断点续传 + SHA256 校验）。"""

    def __init__(
        self,
        model_dir: Path,
        timeout: float = 60.0,
        max_concurrency: int = 2
    ):
        self.model_dir = Path(model_dir)
        self.timeout = timeout
        self.max_concurrency = max(1, int(max_concurrency))
        self._progress_lock = threading.Lock()

    def _rel(self, dest: Path) -> str:
        return dest.relative_to(self.model_dir).as_posix()

    @staticmethod
    def _final_verify(
        repo: ModelRepository,
        model_dir: Path,
        files: List[str], optional: Iterable[str]
    ) -> None:
        """全部资产落盘后统一复验 SHA256。

        `checksums.json` 是可选资产，与其它文件并发下载；单文件在 `_stream`
        里落盘时清单可能还没到，`verify_file()` 会因找不到期望值直接返回
        True。只在最后统一复验一次，坏文件删除，必需资产失败即报错。
        """
        optional_set = set(optional)
        for filename in dict.fromkeys([*files, *optional_set]):
            dest = model_dir / filename
            if not dest.exists() or dest.is_dir():
                continue
            if repo.verify_file(filename, use_cache=False):
                continue
            dest.unlink(missing_ok=True)
            if filename in optional_set and filename not in files:
                with _print_lock:
                    print(f"  ⚠ 可选资产 {filename} 校验失败（已删除）",
                          file=sys.stderr)
                continue
            raise RuntimeError(
                f"{filename} 校验失败（checksums.json 不匹配），已删除坏文件")

    async def _download_file(
        self,
        client,
        url: str,
        dest: Path,
        repo: ModelRepository,
        progress_callback: Optional[Callable] = None,
        sem: Optional[asyncio.Semaphore] = None,
        optional: bool = False,
    ):
        if dest.exists() and repo.verify_file(self._rel(dest), use_cache=False):
            self._maybe_unpack(dest)
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_name(dest.name + ".tmp")
        try:
            if sem is not None:
                async with sem:
                    await self._stream(client, url, dest, tmp_path, repo,
                                       progress_callback)
            else:
                await self._stream(client, url, dest, tmp_path, repo,
                                   progress_callback)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            if optional:
                with _print_lock:
                    print(f"  ⚠ 可选资产 {dest.name} 获取失败（已忽略）: {exc}",
                          file=sys.stderr)
                return None
            raise RuntimeError(f"下载 {dest} 失败: {exc}") from exc
        self._maybe_unpack(dest)
        return dest

    def _maybe_unpack(self, dest: Path) -> None:
        """tokenizer.zip → 解压到 model_dir（防路径穿越）。"""
        if dest.suffix.lower() != ".zip" or not dest.exists():
            return
        with zipfile.ZipFile(dest) as zf:
            for member in zf.infolist():
                name = member.filename.replace("\\", "/")
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"压缩包含非法路径: {name}")
                target = self.model_dir / name
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as out:
                    out.write(src.read())

    async def _stream(self, client, url, dest, tmp_path, repo, progress_callback):
        with _print_lock:
            sys.stderr.write(f"\r  ↓ 下载 {dest.name} 从 {url}\n")
        headers = {}
        resume = tmp_path.stat().st_size if tmp_path.exists() else 0
        if resume:
            headers["Range"] = f"bytes={resume}-"

        async with client.stream(
            "GET", url,
            headers=headers,
            follow_redirects=True
        ) as response:
            # 206 = 断点续传；200 = 服务端忽略 Range（需从头写）
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
            with open(tmp_path, mode) as f:
                async for chunk in response.aiter_bytes(256 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and progress_callback:
                        progress_callback(dest.name, downloaded, total)
                    elif total and self.max_concurrency == 1:
                        with self._progress_lock:
                            pct = min(100.0, downloaded / total * 100)
                            bar_len = int(pct / 5)
                            bar = "█" * bar_len + "░" * (20 - bar_len)
                            sys.stderr.write(
                                f"\r  ↓ {dest.name:30s} [{bar}] {pct:5.1f}%")
                            sys.stderr.flush()
        with _print_lock:
            sys.stderr.write("\n")
        os.replace(tmp_path, dest)

        # 校验失败必须删除坏文件，否则会永久残留并阻断后续下载
        if not repo.verify_file(self._rel(dest), use_cache=False):
            dest.unlink(missing_ok=True)
            raise ValueError(
                f"{dest.name} 校验失败（checksums.json 不匹配）: {url}"
            )
        return dest

    async def download(
        self,
        files: List[str],
        repo: ModelRepository,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        optional: Iterable[str] = (),
    ) -> List[Path]:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        optional_set = set(optional)

        # 可选资产不与必需资产重复；已存在且校验通过的跳过
        tasks = []
        for filename in dict.fromkeys([*files, *optional_set]):
            dest = self.model_dir / filename
            is_optional = filename in optional_set and filename not in files
            if dest.exists() and repo.verify_file(filename, use_cache=False):
                self._maybe_unpack(dest)
                continue
            tasks.append((filename, dest, is_optional))

        if not tasks:
            return [self.model_dir / f for f in files]

        with _print_lock:
            print(f"  ⏬ 需要下载 {len(tasks)} 个 Release 资产:")
            for name, _, opt in tasks:
                print(f"     • {name}{'（可选）' if opt else ''}")

        import httpx
        sem = asyncio.Semaphore(self.max_concurrency)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            await asyncio.gather(*[
                self._download_file(client, repo.download_url(name), dest, repo,
                                    progress_callback, sem, opt)
                for name, dest, opt in tasks
            ])
        # checksums.json 可能最后才到：清空缓存并统一复验所有资产
        repo._checksums = None
        repo._verify_cache.clear()
        self._final_verify(repo, self.model_dir, files, optional_set)
        with _print_lock:
            print(f"  ✅ 下载完成 → {self.model_dir}")
        return [self.model_dir / f for f in files]

    @staticmethod
    def download_sync(
        filenames: List[str],
        model_dir: Path,
        repo: ModelRepository,
        optional: Optional[Iterable[str]] = None
    ) -> List[Path]:
        downloader = AsyncModelDownloader(model_dir)
        opts = list(optional or [])

        def runner():
            return asyncio.run(downloader.download(filenames, repo,
                                                   optional=opts))

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return runner()

        # 已在事件循环中：开线程跑新 loop
        box: dict = {}

        def threaded():
            try:
                box["value"] = runner()
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc

        t = threading.Thread(target=threaded, daemon=True)
        t.start()
        t.join()
        if "error" in box:
            raise box["error"]
        return box.get("value", [])
