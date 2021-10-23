from pathlib import Path

import pytest
from PIL import Image
from slider.image_slicer import SliderImage


@pytest.fixture
def get_files():
    files = {'cat': Path('resources', 'images', 'cat_with_whiskers.jpg'),
             'dog': Path('resources', 'images', 'irish_red_setter.jpg'),
             'tiger': Path('resources', 'images', 'tiger_landscape.jpg')}
    return files


def test_resize_image(get_files):
    # with Image.open(get_files['tiger']) as tiger_picture:
    #     tiger_picture.show()
    tiger_rs = SliderImage(get_files['tiger']).resize_image(200, 100)
    tiger_rs.show()
    assert tiger_rs.width == 200
    assert tiger_rs.height == 100


def test_partition_image():
    dog_rs = SliderImage(get_files['dog']).resize_image(800, 400)
    dog_parts = dog_rs


def test_add_blank():
    assert False
