"""Off-thread JPEG writer pool for upscale stage."""
import concurrent.futures
import logging
import threading

import cv2

logger = logging.getLogger("service.worker")


class JpegWriterPool:
    def __init__(self, max_workers: int = 2, max_inflight: int = 16):
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="upscale-jpeg",
        )
        self._pending: list[concurrent.futures.Future] = []
        self._pending_lock = threading.Lock()
        self._inflight_sem = threading.BoundedSemaphore(value=max_inflight)

    def submit(self, out_path: str, img) -> None:
        self._inflight_sem.acquire()
        fut = self._executor.submit(self._write_task, out_path, img)
        with self._pending_lock:
            if len(self._pending) > 32:
                self._pending[:] = [f for f in self._pending if not f.done()]
            self._pending.append(fut)

    def _write_task(self, out_path: str, img) -> None:
        try:
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                logger.warning("JPEG encode failed for %s", out_path)
                return
            with open(out_path, "wb") as f:
                f.write(buf.tobytes())
        except Exception as e:
            logger.warning("JPEG write failed for %s: %s", out_path, e)
        finally:
            self._inflight_sem.release()

    def drain(self, timeout: float | None = None) -> None:
        with self._pending_lock:
            pending = list(self._pending)
            self._pending.clear()
        if not pending:
            return
        logger.info("Draining %d pending JPEG writes", len(pending))
        for f in concurrent.futures.as_completed(pending, timeout=timeout):
            try:
                f.result()
            except Exception:
                pass

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
