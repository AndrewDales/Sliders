from pathlib import Path

import pytest
from PIL import Image
from slider.image_slicer import prepare_slider_images, partition_image


@pytest.fixture
def filenames():
    files = {'cat': Path('resources', 'images', 'cat_with_whiskers.jpg'),
             'dog': Path('resources', 'images', 'irish_red_setter.jpg'),
             'tiger': Path('resources', 'images', 'tiger_landscape.jpg')}
    return files


def test_partition_image(filenames):
    with Image.open(filenames['dog']) as dog_im:
        dog_rs = dog_im.resize((400, 200))
        dog_parts = partition_image(dog_rs, 2, 4)
    dog_parts[str((1, 1))].show()
    assert all((part.width == 100 and part.height == 100 for part in dog_parts.values()))


def test_prepare_slider_images(filenames):
    parts = prepare_slider_images(filenames['tiger'])
    assert parts.keys() == {'images', 'blank'}
    assert len(parts['images']) == 16
