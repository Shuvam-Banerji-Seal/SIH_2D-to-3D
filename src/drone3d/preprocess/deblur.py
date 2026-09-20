"""Motion-blur mitigation: focus-aware unsharp masking and Wiener deconvolution."""

from __future__ import annotations

import cv2
import numpy as np

from drone3d.preprocess.quality import laplacian_variance

__all__ = ["auto_deblur", "gaussian_psf", "unsharp_mask", "wiener_deconvolution"]


def unsharp_mask(
    image: np.ndarray,
    *,
    sigma: float = 1.4,
    amount: float = 0.8,
    threshold: int = 0,
) -> np.ndarray:
    """Classic unsharp masking on a BGR uint8 image."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    if threshold > 0:
        difference = cv2.absdiff(image, blurred)
        mask = (difference < threshold).astype(np.uint8) * 255
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return np.where(mask[..., None] > 0, image, sharpened).astype(np.uint8)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def gaussian_psf(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Centered Gaussian point-spread function, normalized to unit sum."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    psf = np.exp(
        -((yy - height // 2) ** 2 + (xx - width // 2) ** 2) / (2.0 * max(sigma, 1e-6) ** 2)
    )
    psf /= psf.sum()
    return np.fft.ifftshift(psf)


def wiener_deconvolution(
    image: np.ndarray,
    *,
    psf_sigma: float = 1.0,
    snr: float = 0.01,
) -> np.ndarray:
    """Frequency-domain Wiener deconvolution assuming a Gaussian blur kernel.

    Args:
        image: BGR uint8 image.
        psf_sigma: Standard deviation of the assumed blur PSF (pixels).
        snr: Signal-to-noise ratio used to regularize the inverse filter.
    """
    if snr <= 0:
        raise ValueError("snr must be positive")
    source = image.astype(np.float64) / 255.0
    height, width = source.shape[:2]
    psf = gaussian_psf((height, width), psf_sigma)
    transfer = np.fft.fft2(psf)

    channels = source.shape[2] if source.ndim == 3 else 1
    restored = np.empty_like(source)
    for channel in range(channels):
        plane = source[..., channel] if source.ndim == 3 else source
        spectrum = np.fft.fft2(plane)
        inverse = np.conj(transfer) / (np.abs(transfer) ** 2 + 1.0 / snr)
        filtered = np.real(np.fft.ifft2(spectrum * inverse))
        restored[..., channel] = filtered

    restored = np.clip(restored * 255.0, 0, 255)
    return restored.astype(np.uint8)


def auto_deblur(
    image: np.ndarray,
    *,
    threshold: float = 60.0,
    sigma: float = 1.4,
    amount: float = 0.8,
    psf_sigma: float = 1.0,
    snr: float = 0.01,
) -> np.ndarray:
    """Deblur only when the frame is softer than ``threshold``; guard against regression."""
    if laplacian_variance(image) >= threshold:
        return image
    candidate = wiener_deconvolution(image, psf_sigma=psf_sigma, snr=snr)
    candidate = unsharp_mask(candidate, sigma=sigma, amount=amount)
    if laplacian_variance(candidate) <= laplacian_variance(image):
        return image
    return candidate
