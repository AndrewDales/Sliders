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

    parts = {str((i, j)): image.crop((dx * i, dy * j, dx * (i + 1), dy * (j + 1)))
             for i in range(n_rows) for j in range(n_cols)
             }
    return parts


def prepare_slider_images(filename, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                          n_rows=DEFAULT_ROWS, n_cols=DEFAULT_COLS,
                          blank_color=DEFAULT_BLANK_COLOR,
                          ):
    blank_pos = (n_rows - 1, n_cols - 1)
    image_parts = {}

    with Image.open(filename) as full_image:
        resized_image = full_image.resize((width, height))
        image_parts['images'] = partition_image(resized_image, n_rows, n_cols)
        image_parts['blank'] = {'pos': blank_pos,
                                'image': Image.new('RGBA', image_parts['images'][str(blank_pos)].size, blank_color)}
    return image_parts


if __name__ == '__main__':
    filepath = Path('..', 'resources', 'images') / 'irish_red_setter.jpg'
    parts = prepare_slider_images(filepath)
