import math
import numpy as np
import random
from colorsys import rgb_to_hsv, hsv_to_rgb, rgb_to_hls, hls_to_rgb


def hex_to_rgb(hex_code: str):
    """Convert hex color (#RRGGBB) to RGB tuple (0-255)."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple):
    """Convert RGB tuple (0-255) back to hex color (#RRGGBB)."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def rand_hex() -> str:
    """Produce a random Hex value for creating dummy tests

    :return str: Hex value with random red, green and blue values.
    """
    return rgb_to_hex((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))


def hex_grid_to_rgb_array(hex_grid: np.ndarray) -> np.ndarray:
    """
    Convert a 2D sequence of Cells with hex strings into a numpy uint8 array of shape (H, W, 3).
    """
    # Basic validation & dimensions
    if (hex_grid == None).all():
        return np.zeros((0, 0, 3), dtype=np.uint8)

    n, m = hex_grid.shape
    arr = np.zeros((n, m, 3), dtype=np.uint8)

    # May be a faster way to do this - np.vectorize or to flatten it and then reshape TODO performative code
    for i in range(n):
        for j in range(m):
            cell = hex_grid[i, j]
            arr[i, j] = hex_to_rgb(cell.colour) if cell is not None else 0

    return arr


def color_properties(hex_code: str):
    """
    Get relative saturation and brightness of a color.
    Returns values in range [0,1].
    """
    r, g, b = [c / 255.0 for c in hex_to_rgb(hex_code)]
    h, s, v = rgb_to_hsv(r, g, b)
    return {"saturation": round(s, 3), "brightness": round(v, 3)}


def mix_colors(hex1: str, hex2: str, ratio: float = 0.5):
    """
    Mix two hex colors together.
    ratio=0.5 gives an even blend, 
    ratio=0.0 gives hex1, ratio=1.0 gives hex2.
    """
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)

    r = int(r1 * (1 - ratio) + r2 * ratio)
    g = int(g1 * (1 - ratio) + g2 * ratio)
    b = int(b1 * (1 - ratio) + b2 * ratio)

    return rgb_to_hex((r, g, b))


def mix_multiple_colors(hex_colours: list, weights: list = None, bias: float = 1):
    """
    Mix multiple hex colors together.
    - hex_colors: list of hex strings (#RRGGBB).
    - weights: list of relative weights (same length as hex_colors).
               If None, equal weights are assumed.
    """
    if bias <= 0 or bias > 1:
        raise ValueError(f"Bias value given was incorrect: {bias}")

    n = len(hex_colours)
    if n == 0:
        raise ValueError(f"At least one color required. Hex colours has length 0")

    if weights is None:
        weights = [1] * n
    if len(weights) != n:
        raise ValueError(f"weights must be the same length as hex_colors. Weights: {weights}. Hex colours: {hex_colours}")

    total_weight = sum(weights) * bias  # Here add in bias so that the system slowly decays over time without input (or lots of initial input)
    r_total = g_total = b_total = 0

    for hex_code, w in zip(hex_colours, weights):
        r, g, b = hex_to_rgb(hex_code)
        r_total += r * w
        g_total += g * w
        b_total += b * w

    r = int(r_total / total_weight)
    g = int(g_total / total_weight)
    b = int(b_total / total_weight)

    return rgb_to_hex((r, g, b))

def mix_colors_hsl(hex_colors: list, weights: list = None):
    """
    Blend multiple colors in HSL space.
    """
    n = len(hex_colors)
    if n == 0:
        raise ValueError("At least one color required")
    if weights is None:
        weights = [1] * n
    if len(weights) != n:
        raise ValueError("weights must match colors length")

    total_weight = sum(weights)
    h_total = s_total = l_total = 0
    h_cos = h_sin = 0  # for circular hue averaging

    for hex_code, w in zip(hex_colors, weights):
        r, g, b = [c / 255.0 for c in hex_to_rgb(hex_code)]
        h, l, s = rgb_to_hls(r, g, b)  # Note: colorsys uses HLS not HSL
        h_cos += w * (math.cos(2 * math.pi * h))
        h_sin += w * (math.sin(2 * math.pi * h))
        s_total += s * w
        l_total += l * w

    # Average hue via atan2 to handle circular wrap-around
    h_avg = (math.atan2(h_sin, h_cos) / (2 * math.pi)) % 1.0
    s_avg = s_total / total_weight
    l_avg = l_total / total_weight

    r, g, b = hls_to_rgb(h_avg, l_avg, s_avg)
    return rgb_to_hex((int(r*255), int(g*255), int(b*255)))