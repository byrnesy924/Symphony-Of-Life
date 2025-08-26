import random
import numpy as np
import imageio.v2 as imageio

from pathlib import Path
from dataclasses import dataclass
from utils_gol import mix_multiple_colors, hex_grid_to_rgb_array, rand_hex

from time import perf_counter


@dataclass
class Cell:
    """Each cell in the game of life. Houses the x-y coords and the colour (This is the advanced replacement for the status) of the cell
    """
    x: int
    y: int
    colour: str  # Hex value


class ColourGrid:
    def __init__(self, pattern: np.ndarray[Cell], bias: float):
        # Pattern is a matrix of x, y positions that contain a cell
        self.pattern: np.ndarray[Cell] = pattern
        self.n, self.m = pattern.shape
        self.bias = bias

    def _get_grid(self) -> np.ndarray[Cell]:
        return self.pattern

    def evolve(self, inputs: np.ndarray[Cell]) -> None:
        """Evolves the grade based on the rules in evaluate cell.

        Coupled with self - all cells are stored in self.pattern and bias is stored in self.bias
        """
        for i in range(self.n):
            for j in range(self.m):

                # Allow the agent to input in new cells - override the previous cell
                if inputs is not None and inputs[i, j] is not None:
                    # TODO - consider blending with current cell
                    self.pattern[i, j] = inputs[i, j]
                    continue

                # TODO consider including the corners!

                self.pattern[i, j] = evaluate_cell(
                    cell=self.pattern[i, j],
                    left=self.pattern[i, j-1] if j > 0 else None,
                    right=self.pattern[i, j+1] if j < self.m - 1 else None,
                    up=self.pattern[i - 1, j] if i > 0 else None,
                    down=self.pattern[i + 1, j] if i < self.n - 1 else None,
                    bias=self.bias
                )
        return


def evaluate_cell(
        cell: Cell,
        left: Cell | None,
        right: Cell | None,
        up: Cell | None,
        down: Cell | None,
        bias: float,
) -> Cell:
    """evaluate if a cell should stay alive and what colour it should be

    :param Cell cell: _description_
    :param Cell left: _description_
    :param Cell right: _description_
    :param Cell up: _description_
    :param Cell down: _description_
    :return Cell: updated cell with a new colour
    """
    if bias <= 0 or bias > 1:
        raise ValueError(f"Bias value given was incorrect: {bias}")

    cells: list[Cell] = [cell, left, right, up, down]
    colours: list[str] = [item.colour for item in cells if item is not None]

    weights: list[int | float] = [1]*len(colours)  # TODO consider a way to get agents to interract with this if possible

    # TODO consider HSL blending instead of RGB
    new_colour = mix_multiple_colors(
        hex_colours=colours,
        weights=weights,
        bias=bias
    )

    return Cell(x=cell.x, y=cell.y, colour=new_colour)


def render_frames_from_lifegrid(
    life_grid: ColourGrid,
    steps: int,
    extra_input: dict[np.ndarray[Cell]],
    start_step: int = 0,
) -> list[np.ndarray]:
    """
    Produce a list of numpy RGB frames (H, W, 3) by evolving `life` for `steps`
    iterations. Does not display anything on-screen.

    :param dict extra_input: A dictionary of additional inputs to give. The keys of the dict need to be integers which are the timesteps to
    input in. The value is an array grid of Cells to input

    """
    if steps <= 0:
        return []

    frames: list[np.ndarray] = []
    # capture initial state before any evolve (optional)
    initial_grid = life_grid._get_grid()
    frames.append(hex_grid_to_rgb_array(initial_grid))

    for step in range(start_step + 1, start_step + steps):
        extra_input_this_turn = None
        if step in extra_input or str(step) in extra_input:
            extra_input_this_turn: np.ndarray[Cell] = extra_input.get(step) if step in extra_input else extra_input.get(str(step))

            # TODO validate the shape of the input numpy array

        life_grid.evolve(inputs=extra_input_this_turn)
        grid = life_grid._get_grid()
        frames.append(hex_grid_to_rgb_array(grid))

    return frames


def random_board(x, y, threshold: float = 0.8) -> np.ndarray:
    """Create a random board"""
    arr = np.empty(shape=(x, y), dtype=object)

    for row in range(x):
        for col in range(y):
            if random.random() > threshold:
                arr[row, col] = Cell(x=row, y=col, colour=rand_hex())
            else:
                # Put in a black cell
                arr[row, col] = Cell(x=row, y=row, colour="#000000")
    return arr


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 12, loop: int = 0) -> None:
    """
    Save a list of RGB frames to an animated GIF.

    Parameters
    ----------
    frames : list of np.ndarray
        Each frame should be a numpy array of shape (H, W, 3) with dtype=uint8,
        representing an RGB image.
    path : str or Path
        File path to save the .gif.
    fps : int, default=12
        Frames per second (controls playback speed).
    loop : int, default=0
        Number of times the GIF should loop. 0 means infinite.
    """
    if not frames:
        raise ValueError("No frames to save.")

    # Verify frames are in the right shape & dtype
    checked_frames = []
    for idx, frame in enumerate(frames):
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Frame {idx} must have shape (H, W, 3), got {arr.shape}.")

        # check frame colours
        if arr.dtype != np.uint8:
            # clip i.e. limit the value of each item in the array to int from 0 to 255
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        checked_frames.append(arr)

    # Duration per frame (seconds)
    duration = 1.0 / float(fps)

    # Save GIF
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, checked_frames, format="GIF", duration=duration, loop=loop)


if __name__ == "__main__":

    sizes_to_do = [
        (64, 128),
        (128, 256),
        (360, 640),
        (480, 640),
        (720, 1080),
        (1080, 1920),
        (1440, 2560)
    ]

    number_of_steps = 144
    bias = 0.4

    for index, size in enumerate(sizes_to_do):
        start = perf_counter()
        rows, cols = size
        # Create starting board
        starting_board = random_board(rows, cols)
        input_boards = dict()

        # Create some random inputs
        for i in range(number_of_steps):
            if random.random() > 0.9:
                input_board = random_board(rows, cols, threshold=0.95)
                input_boards[i] = input_board

        frames = render_frames_from_lifegrid(
            life_grid=ColourGrid(pattern=starting_board, bias=bias),
            steps=number_of_steps,
            extra_input=input_boards
        )

        save_gif(frames=frames, path=f"random-{index}-{rows}x{cols}.gif", fps=12, loop=4)
        end = perf_counter()
        print(f"Took {end - start} seconds ({(end - start)/60} mins) to do size {rows}x{cols} (Total cells {rows*cols})")
