"""Small console progress bars used by the pairwise CNN scripts."""

from __future__ import annotations


class ProgressBar:
    """A lightweight progress bar is printed without requiring tqdm."""

    def __init__(self, total: int, label: str, width: int = 30):
        self.total = max(1, int(total))
        self.label = label
        self.width = width
        self._last = -1

    def update(self, current: int, extra: str = "") -> None:
        """The current progress is drawn on one console line."""
        current = min(max(0, int(current)), self.total)
        percent = int(100 * current / self.total)
        if percent == self._last and current != self.total:
            return
        self._last = percent
        filled = int(self.width * current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" {extra}" if extra else ""
        print(f"\r  {self.label} [{bar}] {current}/{self.total}{suffix}", end="", flush=True)

    def finish(self, extra: str = "") -> None:
        """The bar is completed and the console moves to the next line."""
        self.update(self.total, extra=extra)
        print()

