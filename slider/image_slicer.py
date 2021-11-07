import os
from pathlib import Path
from PIL import Image

DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 600
DEFAULT_ROWS = 4
DEFAULT_COLS = 4
DEFAULT_BLANK_COLOR = "midnightblue"


def partition_image(image: Image, n_rows: int = None, n_cols: int = None):
    """ Slide up an image into a grid of n_rows by n_cols"""
    if n_rows is None:
        n_rows = DEFAULT_ROWS
    if n_cols is None:
        n_cols = n_rows

    dx = image.width // n_cols
    dy = image.height // n_rows

    parts = [image.crop((dx * col, dy * row, dx * (col + 1), dy * (row + 1)))
             for row in range(n_rows) for col in range(n_cols)
             ]
    return parts


def prepare_slider_images(filename, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                          n_rows=DEFAULT_ROWS, n_cols=DEFAULT_COLS,
                          blank_color=DEFAULT_BLANK_COLOR,
                          ):
    image_parts = {}

    with Image.open(filename) as full_image:
        resized_image = full_image.resize((width, height))
        image_parts['images'] = partition_image(resized_image, n_rows, n_cols)
        image_parts['blank'] = Image.new('RGBA', image_parts['images'][-1].size, blank_color)
    return image_parts


if __name__ == '__main__':
    filepath = Path('..', 'resources', 'images') / 'irish_red_setter.jpg'
    parts = prepare_slider_images(filepath)
