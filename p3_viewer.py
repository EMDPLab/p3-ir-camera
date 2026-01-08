#!/usr/bin/env python3
"""P3 Thermal Camera Viewer with ROI + Real-time Plot.

Controls:
  q - Quit           +/- - Zoom
  r - Rotate 90°     c - Colormap
  s - Shutter/NUC    g - Gain mode
  m - Mirror         h - Help
  space - Screenshot D - Dump raw data
  e - Emissivity cycle     1-9 - Set emissivity (0.1-0.9)
  x - Scale mode     p - Enhanced (CLAHE+DDE)
  a - AGC mode       t - Toggle reticule
  d - Toggle DDE

ROI:
  U - ROI mode: POINT
  I - ROI mode: LINE
  O - ROI mode: RECT
  Left click/drag: create ROI (POINT: click, LINE/RECT: drag)
  Right click: delete nearest ROI
  Backspace: delete last ROI
  Delete: clear all ROI
  v - Toggle plot
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import Any, cast
from dataclasses import dataclass, field
from collections import deque

import cv2
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt

from p3_camera import GainMode, P3Camera, raw_to_celsius


# =============================================================================
# ROI
# =============================================================================

class ROITag(IntEnum):
    POINT = 0
    LINE = 1
    RECT = 2


@dataclass
class ROI:
    rid: int
    tag: ROITag
    # Points stored in thermal coordinates: (x, y)
    p0: tuple[int, int]
    p1: tuple[int, int] | None = None  # LINE/RECT uses p1
    enabled: bool = True
    # time series (Celsius): mean, min, max
    t_mean: deque[float] = field(default_factory=lambda: deque(maxlen=600))
    t_min: deque[float] = field(default_factory=lambda: deque(maxlen=600))
    t_max: deque[float] = field(default_factory=lambda: deque(maxlen=600))


# =============================================================================
# Colormaps
# =============================================================================

class ColormapID(IntEnum):
    WHITE_HOT = 0
    BLACK_HOT = 1
    RAINBOW = 2
    IRONBOW = 3
    MILITARY = 4
    SEPIA = 5


class ScaleMode(IntEnum):
    OFF = 0
    NEAREST = 1
    BILINEAR = 2
    BICUBIC = 3
    LANCZOS = 4


class AGCMode(IntEnum):
    FACTORY = 0
    TEMPORAL_1 = 1
    FIXED_RANGE = 2


AGC_PERCENTILES = {AGCMode.TEMPORAL_1: 1.0}

SCALE_INTERP = {
    ScaleMode.NEAREST: cv2.INTER_NEAREST,
    ScaleMode.BILINEAR: cv2.INTER_LINEAR,
    ScaleMode.BICUBIC: cv2.INTER_CUBIC,
    ScaleMode.LANCZOS: cv2.INTER_LANCZOS4,
}

COLORMAPS: dict[ColormapID, NDArray[np.uint8]] = {}


def _cv_lut(colormap: int) -> NDArray[np.uint8]:
    gray = np.arange(256, dtype=np.uint8).reshape(1, 256)
    colored = cv2.applyColorMap(gray, colormap)
    return cast(NDArray[np.uint8], colored.reshape(256, 3))


def _init_colormaps() -> None:
    global COLORMAPS
    ramp = np.arange(256, dtype=np.uint8)

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = ramp
    COLORMAPS[ColormapID.WHITE_HOT] = lut

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = 255 - ramp
    COLORMAPS[ColormapID.BLACK_HOT] = lut

    COLORMAPS[ColormapID.RAINBOW] = _cv_lut(cv2.COLORMAP_JET)
    COLORMAPS[ColormapID.IRONBOW] = _cv_lut(cv2.COLORMAP_INFERNO)

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = (ramp * 0.2).astype(np.uint8)
    lut[:, 1] = ramp
    lut[:, 2] = (ramp * 0.3).astype(np.uint8)
    COLORMAPS[ColormapID.MILITARY] = lut

    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = (ramp * 0.4).astype(np.uint8)
    lut[:, 1] = (ramp * 0.7).astype(np.uint8)
    lut[:, 2] = ramp
    COLORMAPS[ColormapID.SEPIA] = lut


_init_colormaps()


def get_colormap(cmap_id: ColormapID | int) -> NDArray[np.uint8]:
    return COLORMAPS[ColormapID(cmap_id)]


def apply_colormap(img_u8: NDArray[np.uint8], cmap_id: ColormapID | int) -> NDArray[np.uint8]:
    lut = get_colormap(cmap_id)
    return lut[img_u8]


# =============================================================================
# ISP
# =============================================================================

_agc_ema_low: float | None = None
_agc_ema_high: float | None = None


def agc_temporal(img: NDArray[np.uint16], pct: float = 1.0, ema_alpha: float = 0.1) -> NDArray[np.uint8]:
    global _agc_ema_low, _agc_ema_high
    low = float(np.percentile(img, pct))
    high = float(np.percentile(img, 100.0 - pct))

    if _agc_ema_low is None or _agc_ema_high is None:
        _agc_ema_low, _agc_ema_high = low, high
    else:
        _agc_ema_low = ema_alpha * low + (1 - ema_alpha) * _agc_ema_low
        _agc_ema_high = ema_alpha * high + (1 - ema_alpha) * _agc_ema_high

    if _agc_ema_high <= _agc_ema_low:
        return np.zeros(img.shape, dtype=np.uint8)

    normalized = (img.astype(np.float32) - _agc_ema_low) / (_agc_ema_high - _agc_ema_low)
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


def agc_fixed(img: NDArray[np.uint16], temp_min: float = 18.0, temp_max: float = 35.0) -> NDArray[np.uint8]:
    raw_min = (temp_min + 273.15) * 64
    raw_max = (temp_max + 273.15) * 64
    normalized = (img.astype(np.float32) - raw_min) / (raw_max - raw_min)
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


def dde(img_u8: NDArray[np.uint8], strength: float = 0.5, kernel_size: int = 3) -> NDArray[np.uint8]:
    if strength <= 0:
        return img_u8
    ksize = kernel_size | 1
    blurred = cv2.GaussianBlur(img_u8, (ksize, ksize), 0)
    img_f = img_u8.astype(np.float32)
    blurred_f = blurred.astype(np.float32)
    enhanced = img_f + strength * (img_f - blurred_f)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def tnr(img: NDArray[np.uint16], prev_img: NDArray[np.uint16] | None, alpha: float = 0.3) -> NDArray[np.uint16]:
    if prev_img is None:
        return img
    result = alpha * img.astype(np.float32) + (1 - alpha) * prev_img.astype(np.float32)
    return result.astype(np.uint16)


# =============================================================================
# Viewer
# =============================================================================

class P3Viewer:
    def __init__(self) -> None:
        self.camera = P3Camera()

        self.rotation: int = 0
        self.colormap_idx: int = int(ColormapID.IRONBOW)
        self.mirror: bool = False
        self.show_help: bool = False
        self.show_reticule: bool = True
        self.zoom: int = 3
        self.fps: float = 0.0

        self.enhanced: bool = True
        self.use_clahe: bool = True
        self.scale_mode: ScaleMode = ScaleMode.BICUBIC
        self.agc_mode: AGCMode = AGCMode.FACTORY
        self.dde_strength: float = 0.3
        self.tnr_alpha: float = 0.5

        self._fps_count: int = 0
        self._fps_time: float = time.time()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._last_display: NDArray[np.uint8] | None = None
        self._prev_frame: NDArray[np.uint16] | None = None
        self._ir_brightness: NDArray[np.uint8] | None = None

        # ROI system
        self.rois: list[ROI] = []
        self._roi_next_id: int = 1
        self.roi_mode: ROITag = ROITag.RECT
        self._mouse_down: bool = False
        self._temp_p0_disp: tuple[int, int] | None = None
        self._temp_p1_disp: tuple[int, int] | None = None
        self._last_thermal: NDArray[np.uint16] | None = None

        # Plot
        self.plot_enabled: bool = True
        self._plot_inited: bool = False
        self._plot_lines: dict[int, Any] = {}
        self._fig = None
        self._ax = None

        # Plot throttling (prevents frame stutter/jitter)
        self.plot_interval_s: float = 0.10   # 10 Hz plot refresh
        self._plot_last_t: float = 0.0


    def run(self) -> None:
        print("P3 Thermal Viewer")
        self.camera.connect()
        name, version = self.camera.init()
        print(f"Device: {name}, Firmware: {version}")
        self.camera.start_streaming()
        print("Press 'h' for help")

        cv2.namedWindow("P3 Thermal", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("P3 Thermal", 640, 480)
        # CRITICAL FIX: mouse callback must be set here (window exists, self exists)
        cv2.setMouseCallback("P3 Thermal", self._on_mouse)  # <-- delete (self not defined here)


        try:
            while True:
                ir_brightness, thermal = self.camera.read_frame_both()
                if thermal is None:
                    continue
                self._ir_brightness = ir_brightness

                thermal = tnr(thermal, self._prev_frame, alpha=self.tnr_alpha)
                self._prev_frame = thermal.copy()

                self._last_display = self._render(thermal)
                cv2.imshow("P3 Thermal", self._last_display)
                self._update_fps()

                if not self._handle_key(thermal):
                    break
                if cv2.getWindowProperty("P3 Thermal", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.camera.stop_streaming()
            cv2.destroyAllWindows()

    def _update_fps(self) -> None:
        self._fps_count += 1
        now = time.time()
        if now - self._fps_time >= 1.0:
            self.fps = self._fps_count / (now - self._fps_time)
            self._fps_count = 0
            self._fps_time = now

    def _get_spot_coords(self, thermal: NDArray[np.uint16]) -> tuple[int, int]:
        th, tw = thermal.shape
        cy, cx = th // 2, tw // 2
        if self.mirror:
            cx = tw - 1 - cx
        return cy, cx

    def _render(self, thermal: NDArray[np.uint16]) -> NDArray[np.uint8]:
        # AGC
        if self.agc_mode == AGCMode.FACTORY:
            if self._ir_brightness is not None:
                img_u8 = self._ir_brightness[2:, :].copy()
            else:
                img_u8 = agc_temporal(thermal, pct=1.0)
        elif self.agc_mode == AGCMode.FIXED_RANGE:
            img_u8 = agc_fixed(thermal)
        else:
            pct = AGC_PERCENTILES.get(self.agc_mode, 1.0)
            img_u8 = agc_temporal(thermal, pct=pct)

        # Optional 2x upscaling (applied to 8-bit before color)
        if self.scale_mode != ScaleMode.OFF:
            h, w = img_u8.shape[:2]
            img_u8 = cv2.resize(img_u8, (w * 2, h * 2), interpolation=SCALE_INTERP[self.scale_mode])

        if self.use_clahe:
            img_u8 = cast(NDArray[np.uint8], self._clahe.apply(img_u8))

        img_u8 = dde(np.asarray(img_u8, dtype=np.uint8), strength=self.dde_strength)

        img = apply_colormap(img_u8, self.colormap_idx)

        if self.mirror:
            img = cv2.flip(img, 1)

        if self.rotation == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif self.rotation == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Zoom (final display image)
        h, w = img.shape[:2]
        result = cast(NDArray[np.uint8], cv2.resize(img, (w * self.zoom, h * self.zoom), interpolation=cv2.INTER_LINEAR))

        # Overlays must be drawn on the displayed frame
        self._draw_overlays(result, thermal)
        self._last_thermal = thermal

        # Draw ROIs on the *final* displayed image (after zoom/rotate)
        self._draw_rois(result, thermal)

        # Throttle plot updates to avoid OpenCV frame jitter
        if self.plot_enabled:
            now = time.time()
            if now - self._plot_last_t >= self.plot_interval_s:
                self._plot_last_t = now
                self._update_plot()

        return result

    def _draw_overlays(self, img: NDArray[np.uint8], thermal: NDArray[np.uint16]) -> None:
        h, w = img.shape[:2]
        cy, cx = self._get_spot_coords(thermal)

        spot = raw_to_celsius(thermal[cy, cx])
        tmin = raw_to_celsius(thermal.min())
        tmax = raw_to_celsius(thermal.max())

        cv2.putText(img, f"Spot: {spot:.1f}C | Range: {tmin:.1f}-{tmax:.1f}C",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cmap_name = ColormapID(self.colormap_idx).name
        gain_name = self.camera.gain_mode.name
        emissivity = self.camera.env_params.emissivity
        scale = self.scale_mode.name if self.scale_mode != ScaleMode.OFF else ""
        roi_mode = self.roi_mode.name
        status = f"{self.fps:.1f} FPS | {cmap_name} | {gain_name} | e={emissivity:.2f} | ROI={roi_mode} {scale}"
        cv2.putText(img, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.show_reticule:
            cx_d, cy_d = w // 2, h // 2
            cv2.line(img, (cx_d - 15, cy_d), (cx_d + 15, cy_d), (0, 255, 0), 1)
            cv2.line(img, (cx_d, cy_d - 15), (cx_d, cy_d + 15), (0, 255, 0), 1)
            cv2.putText(img, f"{spot:.1f}C", (cx_d + 20, cy_d - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.show_help:
            self._draw_help(img)

    def _draw_help(self, img: NDArray[np.uint8]) -> None:
        lines = [
            "q Quit   +/- Zoom  r Rotate  c Color",
            "s Shutter  g Gain  m Mirror  h Help",
            "e Emiss cycle  1-9 Emiss set",
            "x Scale  p Enhanced  a AGC  d DDE  t Reticule",
            "U POINT  I LINE  O RECT | L-drag create | R-click delete",
            "Backspace delete last ROI | Delete clear all | v Plot toggle",
        ]
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 30), (520, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (10, 50 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _handle_key(self, thermal: NDArray[np.uint16]) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return True

        if key == ord("q"):
            return False

        if key == ord("r"):
            self.rotation = (self.rotation + 90) % 360
        elif key == ord("c"):
            self.colormap_idx = (self.colormap_idx + 1) % len(ColormapID)
        elif key == ord("s"):
            self.camera.trigger_shutter()
            print("Shutter triggered")
        elif key == ord("g"):
            new_mode = GainMode.LOW if self.camera.gain_mode == GainMode.HIGH else GainMode.HIGH
            self.camera.set_gain_mode(new_mode)
            print(f"Gain mode: {new_mode.name}")
        elif key == ord("m"):
            self.mirror = not self.mirror
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("e"):
            self._cycle_emissivity()
        elif ord("1") <= key <= ord("9"):
            self._set_emissivity((key - ord("0")) / 10.0)
        elif key == ord("d"):
            self._toggle_dde()
        elif key == ord("D"):
            self._dump(thermal)
        elif key == ord(" "):
            self._screenshot()
        elif key in (ord("+"), ord("=")):
            self.zoom = min(6, self.zoom + 1)
        elif key in (ord("-"), ord("_")):
            self.zoom = max(1, self.zoom - 1)
        elif key == ord("x"):
            self.scale_mode = ScaleMode((self.scale_mode + 1) % len(ScaleMode))
            print(f"Scale: {self.scale_mode.name}")
        elif key == ord("p"):
            self._toggle_enhanced()
        elif key == ord("a"):
            self.agc_mode = AGCMode((self.agc_mode + 1) % len(AGCMode))
            print(f"AGC: {self.agc_mode.name}")
        elif key == ord("t"):
            self.show_reticule = not self.show_reticule

        # CRITICAL FIX: ROI mode uses U/I/O (no conflict with emissivity 1-9)
        elif key in (ord("u"), ord("U")):
            self.roi_mode = ROITag.POINT
            print("ROI mode: POINT")
        elif key in (ord("i"), ord("I")):
            self.roi_mode = ROITag.LINE
            print("ROI mode: LINE")
        elif key in (ord("o"), ord("O")):
            self.roi_mode = ROITag.RECT
            print("ROI mode: RECT")

        elif key in (ord("v"), ord("V")):
            self.plot_enabled = not self.plot_enabled
            print(f"Plot: {'ON' if self.plot_enabled else 'OFF'}")

        # ROI deletion keys (platform differences exist; keep both)
        elif key in (8, 127):  # Backspace on some systems returns 8 or 127
            if self.rois:
                rid = self.rois[-1].rid
                self.rois.pop()
                self._plot_lines.pop(rid, None)
                print("Deleted last ROI")
        elif key == 46:  # Delete often maps to 46 in OpenCV
            self.rois.clear()
            self._plot_lines.clear()
            print("Cleared all ROI")

        return True

    def _toggle_enhanced(self) -> None:
        self.enhanced = not self.enhanced
        if self.enhanced:
            self.use_clahe = True
            self.dde_strength = 0.3
        else:
            self.use_clahe = False
            self.dde_strength = 0.0
        print(f"Enhanced: {'ON' if self.enhanced else 'OFF'}")

    def _toggle_dde(self) -> None:
        self.dde_strength = 0.0 if self.dde_strength > 0 else 0.3
        print(f"DDE: {'ON' if self.dde_strength > 0 else 'OFF'}")

    def _dump(self, thermal: NDArray[np.uint16]) -> None:
        ts = time.strftime("%H%M%S")
        cy, cx = self._get_spot_coords(thermal)
        print(f"\n--- Dump {ts} ---")
        print(f"Shape: {thermal.shape}, Range: {thermal.min()}-{thermal.max()}")
        print(f"Center raw: {thermal[cy, cx]}, Center C: {raw_to_celsius(thermal[cy, cx]):.1f}")
        np.save(f"p3_raw_{ts}.npy", thermal)
        print(f"Saved: p3_raw_{ts}.npy\n")

    def _screenshot(self) -> None:
        if self._last_display is None:
            print("No frame to save")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"p3_{ts}.png"
        cv2.imwrite(filename, self._last_display)
        print(f"Saved: {filename}")

    def _set_emissivity(self, value: float) -> None:
        self.camera.env_params.emissivity = value
        print(f"Emissivity: {value:.2f}")

    def _cycle_emissivity(self) -> None:
        values = [0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30, 0.10]
        current = self.camera.env_params.emissivity
        idx = 0
        for i, v in enumerate(values):
            if abs(current - v) < 0.01:
                idx = (i + 1) % len(values)
                break
        self._set_emissivity(values[idx])

    # ----------------------------
    # Coordinate mapping
    # ----------------------------

    def _display_shape(self, thermal: NDArray[np.uint16]) -> tuple[int, int]:
        th, tw = thermal.shape
        h, w = th, tw
        if self.scale_mode != ScaleMode.OFF:
            h *= 2
            w *= 2
        if self.rotation in (90, 270):
            h, w = w, h
        return h, w

    def _disp_to_prezoom(self, xd: int, yd: int, thermal: NDArray[np.uint16]) -> tuple[int, int]:
        pre_h, pre_w = self._display_shape(thermal)
        x = int(xd / self.zoom)
        y = int(yd / self.zoom)
        x = max(0, min(pre_w - 1, x))
        y = max(0, min(pre_h - 1, y))
        return x, y

    def _prezoom_to_thermal(self, x: int, y: int, thermal: NDArray[np.uint16]) -> tuple[int, int]:
        th, tw = thermal.shape
        scale2 = 2 if self.scale_mode != ScaleMode.OFF else 1
        h0 = th * scale2
        w0 = tw * scale2

        # undo rotation
        if self.rotation == 90:
            x0 = y
            y0 = h0 - 1 - x
        elif self.rotation == 180:
            x0 = w0 - 1 - x
            y0 = h0 - 1 - y
        elif self.rotation == 270:
            x0 = w0 - 1 - y
            y0 = x
        else:
            x0, y0 = x, y

        # undo mirror (mirror before rotate in pipeline, so undo after undo-rotate)
        if self.mirror:
            x0 = w0 - 1 - x0

        # undo 2x
        if scale2 == 2:
            x0 //= 2
            y0 //= 2

        x0 = max(0, min(tw - 1, int(x0)))
        y0 = max(0, min(th - 1, int(y0)))
        return x0, y0

    def disp_to_thermal(self, xd: int, yd: int, thermal: NDArray[np.uint16]) -> tuple[int, int]:
        x_pre, y_pre = self._disp_to_prezoom(xd, yd, thermal)
        return self._prezoom_to_thermal(x_pre, y_pre, thermal)

    def thermal_to_disp_prezoom(self, xt: int, yt: int, thermal: NDArray[np.uint16]) -> tuple[int, int]:
        th, tw = thermal.shape
        x, y = xt, yt
        scale2 = 2 if self.scale_mode != ScaleMode.OFF else 1

        x *= scale2
        y *= scale2

        if self.mirror:
            w0 = tw * scale2
            x = w0 - 1 - x

        if self.rotation == 90:
            h0 = th * scale2
            x, y = h0 - 1 - y, x
        elif self.rotation == 180:
            w0 = tw * scale2
            h0 = th * scale2
            x, y = w0 - 1 - x, h0 - 1 - y
        elif self.rotation == 270:
            w0 = tw * scale2
            x, y = y, w0 - 1 - x

        return int(x), int(y)

    def thermal_to_disp(self, xt: int, yt: int, thermal: NDArray[np.uint16]) -> tuple[int, int]:
        x, y = self.thermal_to_disp_prezoom(xt, yt, thermal)
        return x * self.zoom, y * self.zoom

    # ----------------------------
    # Mouse / ROI
    # ----------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: Any = None) -> None:
        if self._last_thermal is None:
            return
        thermal = self._last_thermal

        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
            self._temp_p0_disp = (x, y)
            self._temp_p1_disp = (x, y)

            if self.roi_mode == ROITag.POINT:
                xt, yt = self.disp_to_thermal(x, y, thermal)
                self._add_roi_point(xt, yt)
                self._mouse_down = False
                self._temp_p0_disp = None
                self._temp_p1_disp = None

        elif event == cv2.EVENT_MOUSEMOVE and self._mouse_down:
            self._temp_p1_disp = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self._mouse_down:
            self._mouse_down = False
            if self._temp_p0_disp is None or self._temp_p1_disp is None:
                return
            x0, y0 = self._temp_p0_disp
            x1, y1 = self._temp_p1_disp
            self._temp_p0_disp = None
            self._temp_p1_disp = None

            xt0, yt0 = self.disp_to_thermal(x0, y0, thermal)
            xt1, yt1 = self.disp_to_thermal(x1, y1, thermal)

            if self.roi_mode == ROITag.LINE:
                self._add_roi_line(xt0, yt0, xt1, yt1)
            elif self.roi_mode == ROITag.RECT:
                self._add_roi_rect(xt0, yt0, xt1, yt1)

        elif event == cv2.EVENT_RBUTTONDOWN:
            xt, yt = self.disp_to_thermal(x, y, thermal)
            self._delete_nearest_roi(xt, yt)

    def _add_roi_point(self, x: int, y: int) -> None:
        self.rois.append(ROI(self._roi_next_id, ROITag.POINT, (x, y)))
        self._roi_next_id += 1
        print(f"Added ROI#{self._roi_next_id-1} POINT")

    def _add_roi_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.rois.append(ROI(self._roi_next_id, ROITag.LINE, (x0, y0), (x1, y1)))
        self._roi_next_id += 1
        print(f"Added ROI#{self._roi_next_id-1} LINE")

    def _add_roi_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.rois.append(ROI(self._roi_next_id, ROITag.RECT, (x0, y0), (x1, y1)))
        self._roi_next_id += 1
        print(f"Added ROI#{self._roi_next_id-1} RECT")

    def _delete_nearest_roi(self, x: int, y: int) -> None:
        if not self.rois:
            return
        best_i = -1
        best_d2 = 1e18
        for i, r in enumerate(self.rois):
            if r.tag == ROITag.POINT:
                cx, cy = r.p0
            else:
                assert r.p1 is not None
                cx = (r.p0[0] + r.p1[0]) // 2
                cy = (r.p0[1] + r.p1[1]) // 2
            d2 = (cx - x) ** 2 + (cy - y) ** 2
            if d2 < best_d2:
                best_d2, best_i = d2, i

        if best_d2 <= (12 ** 2):
            rid = self.rois[best_i].rid
            self.rois.pop(best_i)
            self._plot_lines.pop(rid, None)
            print(f"Deleted ROI#{rid}")

    def _line_samples(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        pts: list[tuple[int, int]] = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            pts.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return pts

    def _roi_values(self, r: ROI, thermal: NDArray[np.uint16]) -> tuple[np.ndarray, list[tuple[int, int]]]:
        th, tw = thermal.shape
        if r.tag == ROITag.POINT:
            x, y = r.p0
            return np.array([thermal[y, x]], dtype=np.uint16), [(x, y)]

        assert r.p1 is not None
        if r.tag == ROITag.LINE:
            pts = self._line_samples(r.p0[0], r.p0[1], r.p1[0], r.p1[1])
            pts = [(max(0, min(tw - 1, x)), max(0, min(th - 1, y))) for (x, y) in pts]
            vals = np.array([thermal[y, x] for (x, y) in pts], dtype=np.uint16)
            return vals, pts

        x0, y0 = r.p0
        x1, y1 = r.p1
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        xa = max(0, min(tw - 1, xa))
        xb = max(0, min(tw - 1, xb))
        ya = max(0, min(th - 1, ya))
        yb = max(0, min(th - 1, yb))
        patch = thermal[ya:yb + 1, xa:xb + 1].copy()
        pts = [(x, y) for y in range(ya, yb + 1) for x in range(xa, xb + 1)]
        return patch.reshape(-1), pts

    def _draw_rois(self, img: NDArray[np.uint8], thermal: NDArray[np.uint16]) -> None:
        y0_text = 40
        for k, r in enumerate(self.rois):
            vals, pts = self._roi_values(r, thermal)
            if vals.size == 0:
                continue

            i_min = int(vals.argmin())
            i_max = int(vals.argmax())
            v_min = raw_to_celsius(int(vals[i_min]))
            v_max = raw_to_celsius(int(vals[i_max]))
            v_mean = float(np.mean([raw_to_celsius(int(v)) for v in vals]))

            r.t_min.append(v_min)
            r.t_max.append(v_max)
            r.t_mean.append(v_mean)

            x_min, y_min = pts[i_min]
            x_max, y_max = pts[i_max]

            # geometry
            if r.tag == ROITag.POINT:
                xd, yd = self.thermal_to_disp(r.p0[0], r.p0[1], thermal)
                cv2.circle(img, (xd, yd), 6, (255, 255, 255), 1)
            elif r.tag == ROITag.LINE:
                assert r.p1 is not None
                x0d, y0d = self.thermal_to_disp(r.p0[0], r.p0[1], thermal)
                x1d, y1d = self.thermal_to_disp(r.p1[0], r.p1[1], thermal)
                cv2.line(img, (x0d, y0d), (x1d, y1d), (255, 255, 255), 1)
            else:
                assert r.p1 is not None
                x0, y0 = r.p0
                x1, y1 = r.p1
                xa, xb = sorted((x0, x1))
                ya, yb = sorted((y0, y1))
                pA = self.thermal_to_disp(xa, ya, thermal)
                pB = self.thermal_to_disp(xb, yb, thermal)
                cv2.rectangle(img, pA, pB, (255, 255, 255), 1)

            # hot/cold markers
            xhd, yhd = self.thermal_to_disp(x_max, y_max, thermal)
            xcd, ycd = self.thermal_to_disp(x_min, y_min, thermal)
            cv2.circle(img, (xhd, yhd), 5, (0, 0, 255), 2)
            cv2.circle(img, (xcd, ycd), 5, (255, 0, 0), 2)

            cv2.putText(
                img,
                f"ROI#{r.rid} {r.tag.name}: min {v_min:.1f}C  mean {v_mean:.1f}C  max {v_max:.1f}C",
                (10, y0_text + 18 * k),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        # rubber-band preview while dragging (drawn in display coords)
        if self._temp_p0_disp is not None and self._temp_p1_disp is not None and self._mouse_down:
            x0, y0 = self._temp_p0_disp
            x1, y1 = self._temp_p1_disp
            if self.roi_mode == ROITag.LINE:
                cv2.line(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
            elif self.roi_mode == ROITag.RECT:
                cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)

    # ----------------------------
    # Plot
    # ----------------------------

    def _init_plot(self) -> None:
        plt.ion()
        self._fig, self._ax = plt.subplots()
        self._ax.set_title("ROI Mean Temperature (C)")
        self._ax.set_xlabel("Samples")
        self._ax.set_ylabel("Celsius")
        self._plot_inited = True

    def _update_plot(self) -> None:
        if not self.rois:
            return
        if not self._plot_inited:
            self._init_plot()
        assert self._ax is not None and self._fig is not None

        for r in self.rois:
            if len(r.t_mean) < 2:
                continue
            y = np.array(r.t_mean, dtype=np.float32)
            x = np.arange(len(y), dtype=np.float32)
            line = self._plot_lines.get(r.rid, None)
            if line is None:
                (line,) = self._ax.plot(x, y, label=f"ROI#{r.rid}")
                self._plot_lines[r.rid] = line
                self._ax.legend(loc="upper right")
            else:
                line.set_data(x, y)

        # Keep this light; full autoscale every refresh is expensive.
        self._ax.relim()
        self._ax.autoscale_view()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()



def main() -> None:
    try:
        P3Viewer().run()
    except RuntimeError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()
