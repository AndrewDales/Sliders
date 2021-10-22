import os
from pathlib import Path
from PIL import Image


class SliderImages:
    blank_color: str = "midnightblue"
    num_pixels: int = 600

    def __init__(self, filename: Path):
        self.filename = filename
        self.parts = None
        self.n_x = None
        self.n_y = None

    def resize_image(self, pixels=None):
        if pixels is None:
            pixels = self.num_pixels

        with Image.open(self.filename) as full_image:
            resized_image = full_image.resize((pixels, pixels))
        self.num_pixels = pixels
        return resized_image

    def partition_image(self, n_x=None, n_y=None):
        if n_x is None:
            n_x = 3
        if n_y is None:
            n_y = n_x

        self.n_x = n_x
        self.n_y = n_y

        with Image.open(self.filename) as image:
            if image.size != (self.num_pixels, self.num_pixels):
                image = self.resize_image()

            dx = self.num_pixels // n_x
            dy = self.num_pixels // n_y

            self.parts = {(j, i): image.crop((dx * i, dy * j, dx * (i + 1), dy * (j + 1)))
                          for i in range(n_x) for j in range(n_y)
                          }
        return self.parts

    @staticmethod
    def add_blank(parts, pos=None, blank_color="midnightblue"):
        # Find the last target_positions in the dictionary to make blank
        if pos is None:
            pos = list(parts.keys())[-1]

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

    # Crop the image into slices so there are n_x slices across and n_y slices down and save them in the directory
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
    cat_file = Path('images', 'cat_with_whiskers.png')
    partition_image(cat_file, 4)
