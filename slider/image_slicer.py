import os
from pathlib import Path
from PIL import Image


class SliderImage:
    """ SlicerImage stores the location of an image file that can be resized and partitioned"""
    blank_color: str = "midnightblue"
    width: int = 600
    height: int = 600

    def __init__(self, filename: Path):
        self.filename = filename
        self.parts = None
        self.n_rows = None
        self.n_cols = None

    def resize_image(self, width=None, height=None):
        if width is None:
            width = self.width
        if height is None:
            height = width

        with Image.open(self.filename) as full_image:
            resized_image = full_image.resize((width, height))
        self.width = width
        self.height = height
        return resized_image

    def partition_image(self, n_rows=None, n_cols=None):
        if n_rows is None:
            n_rows = 3
        if n_cols is None:
            n_cols = n_rows

        self.n_rows = n_rows
        self.n_cols = n_cols

        with Image.open(self.filename) as image:
            dx = self.width // n_cols
            dy = self.height // n_rows

            self.parts = {(i, j): image.crop((dx * i, dy * j, dx * (i + 1), dy * (j + 1)))
                          for i in range(n_rows) for j in range(n_cols)
                          }
        return self.parts

    @classmethod
    def add_blank(cls, parts, pos=None, blank_color=None):
        # Find the last target_positions in the dictionary to make blank
        if pos is None:
            pos = list(parts.keys())[-1]
        if blank_color is None:
            blank_color = cls.blank_color

        parts[pos] = Image.new('RGBA', parts[pos].size, blank_color)
        return parts


def partition_image(filename: Path, n_x=None, n_y=None, ):
    BLANK_COLOR = "midnightblue"

    # Set default sizes for horizontal and vertical slices
    if n_x is None:
        n_x = 3
    if n_y is None:
        n_y = n_x

    # Create a directory path for the image parts from the image path and the image name
    parts_dir = filename.parent / (filename.stem + '_parts')
    filetype = filename.suffix
    if not os.path.isdir(parts_dir):
        os.mkdir(parts_dir)

    # Crop the image into slices so there are n_rows slices across and n_cols slices down and save them in the directory
    with Image.open(filename) as image_file:
        width, height = image_file.size
        width_d = width // n_x
        height_d = height // n_y
        for i in range(n_x):
            for j in range(n_y):
                part_image = image_file.crop((i * width_d, j * height_d, (i + 1) * width_d, (j + 1) * height_d))
                part_image_name = parts_dir / f'part_{j}_{i}{filetype}'
                part_image.save(part_image_name)

    # Create a blank tile
    blank_tile = Image.new('RGBA', (width_d, height_d), BLANK_COLOR)
    blank_tile.save(parts_dir / f"blank_tile{filetype}")


if __name__ == '__main__':
    cat_file = Path('../resources/images', 'cat_with_whiskers.png')
    partition_image(cat_file, 4)
