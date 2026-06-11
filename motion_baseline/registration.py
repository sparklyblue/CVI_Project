"""Image registration helpers used to estimate background motion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class ImageRegistrationCache:
    """Downsampled image registrations are cached because many detections share frames."""

    def __init__(self, max_side: int):
        self.max_side = max_side
        self._image_cache: dict[Path, tuple[np.ndarray, float, float]] = {}
        self._registration_cache: dict[tuple[Path, Path], tuple[float, float, float, float]] = {}
        self._local_registration_cache: dict[tuple, tuple[float, float, float, float]] = {}

    def load_small_gray(self, path: Path) -> tuple[np.ndarray, float, float]:
        """
        Returns a normalized grayscale image and x/y scale factors.

        The scale factors convert a shift measured on the small image back to
        pixels in the original image.
        """
        if path in self._image_cache:
            return self._image_cache[path]

        with Image.open(path) as img:
            img = img.convert("L")
            original_w, original_h = img.size
            scale = min(1.0, self.max_side / max(original_w, original_h))
            small_w = max(16, int(round(original_w * scale)))
            small_h = max(16, int(round(original_h * scale)))
            if (small_w, small_h) != img.size:
                img = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0

        sx = original_w / arr.shape[1]
        sy = original_h / arr.shape[0]
        self._image_cache[path] = (arr, sx, sy)
        return arr, sx, sy

    def register(self, path_a: Path, path_b: Path) -> tuple[float, float, float, float]:
        """
        Estimates how the background moved from image A to image B.

        Returns:
            dx_full, dy_full, ncc, peak_ratio

        The alignment quality is kept as a feature, so visually weak pairs are
        represented by the model instead of being removed by a fixed frame-gap rule.
        """
        key = (path_a, path_b)
        if key in self._registration_cache:
            return self._registration_cache[key]

        a, sx, sy = self.load_small_gray(path_a)
        b, _, _ = self.load_small_gray(path_b)
        if a.shape != b.shape:
            # This branch is included for safety if a future dataset mixes sizes.
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a = a[:h, :w]
            b = b[:h, :w]

        dx_small, dy_small, peak_ratio = phase_correlation_shift(a, b)

        # The sign with better normalized correlation is used as the final sign.
        ncc_pos = shifted_ncc(a, b, dx_small, dy_small)
        ncc_neg = shifted_ncc(a, b, -dx_small, -dy_small)
        if ncc_neg > ncc_pos:
            dx_small = -dx_small
            dy_small = -dy_small
            ncc = ncc_neg
        else:
            ncc = ncc_pos

        result = (dx_small * sx, dy_small * sy, float(ncc), float(peak_ratio))
        self._registration_cache[key] = result
        return result

    def register_local(
        self,
        path_a: Path,
        path_b: Path,
        crop_box: tuple[int, int, int, int],
        mask_box_a: tuple[int, int, int, int],
        mask_box_b: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float]:
        """
        Estimates local background motion around an animal.

        The same crop area is read from both images. The animal boxes are masked
        with the local median so the phase correlation is encouraged to use
        nearby background instead of the animal itself.
        """
        key = (path_a, path_b, crop_box, mask_box_a, mask_box_b)
        if key in self._local_registration_cache:
            return self._local_registration_cache[key]

        crop_a = load_crop_gray(path_a, crop_box)
        crop_b = load_crop_gray(path_b, crop_box)
        crop_a = mask_crop_box(crop_a, crop_box, mask_box_a)
        crop_b = mask_crop_box(crop_b, crop_box, mask_box_b)

        crop_a, sx, sy = resize_for_registration(crop_a, self.max_side)
        crop_b, _, _ = resize_for_registration(crop_b, self.max_side)
        dx_small, dy_small, peak_ratio = phase_correlation_shift(crop_a, crop_b)

        ncc_pos = shifted_ncc(crop_a, crop_b, dx_small, dy_small)
        ncc_neg = shifted_ncc(crop_a, crop_b, -dx_small, -dy_small)
        if ncc_neg > ncc_pos:
            dx_small = -dx_small
            dy_small = -dy_small
            ncc = ncc_neg
        else:
            ncc = ncc_pos

        result = (dx_small * sx, dy_small * sy, float(ncc), float(peak_ratio))
        self._local_registration_cache[key] = result
        return result


def load_crop_gray(path: Path, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    """A grayscale crop is read from disk and normalized to 0-1."""
    with Image.open(path) as img:
        img = img.convert("L")
        crop = img.crop(crop_box)
    return np.asarray(crop, dtype=np.float32) / 255.0


def mask_crop_box(
    crop: np.ndarray,
    crop_box: tuple[int, int, int, int],
    mask_box: tuple[int, int, int, int],
) -> np.ndarray:
    """The animal area is filled with local median intensity."""
    x1 = max(mask_box[0] - crop_box[0], 0)
    y1 = max(mask_box[1] - crop_box[1], 0)
    x2 = min(mask_box[2] - crop_box[0], crop.shape[1])
    y2 = min(mask_box[3] - crop_box[1], crop.shape[0])
    if x2 <= x1 or y2 <= y1:
        return crop
    masked = crop.copy()
    masked[y1:y2, x1:x2] = float(np.median(crop))
    return masked


def resize_for_registration(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    """A local crop is resized and scale factors back to crop pixels are returned."""
    h, w = image.shape
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return image, 1.0, 1.0
    new_w = max(16, int(round(w * scale)))
    new_h = max(16, int(round(h * scale)))
    pil = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255), mode="L")
    resized = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return arr, w / arr.shape[1], h / arr.shape[0]


def phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """
    Estimates translation between two small grayscale images.

    The returned shift is measured in the small image coordinate system. A
    simple peak ratio is returned as a confidence-like value.
    """
    eps = 1e-8
    a_norm = normalize_image(a)
    b_norm = normalize_image(b)

    window_y = np.hanning(a_norm.shape[0]).astype(np.float32)
    window_x = np.hanning(a_norm.shape[1]).astype(np.float32)
    window = np.outer(window_y, window_x)
    a_win = a_norm * window
    b_win = b_norm * window

    fa = np.fft.fft2(a_win)
    fb = np.fft.fft2(b_win)
    cross_power = fa * np.conj(fb)
    cross_power /= np.abs(cross_power) + eps
    corr = np.abs(np.fft.ifft2(cross_power))

    peak_y, peak_x = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak_value = float(corr[peak_y, peak_x])

    h, w = corr.shape
    if peak_y > h // 2:
        peak_y -= h
    if peak_x > w // 2:
        peak_x -= w

    # A robust scale is used so the value is not tied to image size.
    background_level = float(np.median(corr) + eps)
    peak_ratio = peak_value / background_level
    return float(peak_x), float(peak_y), float(peak_ratio)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Image contrast is normalized before registration."""
    arr = image.astype(np.float32)
    arr = arr - float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        return arr
    return arr / std


def shifted_ncc(a: np.ndarray, b: np.ndarray, dx: float, dy: float) -> float:
    """
    Returns normalized cross-correlation after applying an integer shift.

    Only the overlapping area is used, so wrapped image edges are not compared.
    """
    sx = int(round(dx))
    sy = int(round(dy))
    h, w = a.shape

    a_y0 = max(0, -sy)
    a_y1 = min(h, h - sy)
    b_y0 = a_y0 + sy
    b_y1 = a_y1 + sy

    a_x0 = max(0, -sx)
    a_x1 = min(w, w - sx)
    b_x0 = a_x0 + sx
    b_x1 = a_x1 + sx

    if a_y1 <= a_y0 or a_x1 <= a_x0:
        return 0.0

    patch_a = normalize_image(a[a_y0:a_y1, a_x0:a_x1]).ravel()
    patch_b = normalize_image(b[b_y0:b_y1, b_x0:b_x1]).ravel()
    if patch_a.size < 32 or patch_b.size < 32:
        return 0.0
    denom = float(np.linalg.norm(patch_a) * np.linalg.norm(patch_b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(patch_a, patch_b) / denom)
