import tkinter as tk
import os
from pathlib import Path
from PIL import ImageTk
from functools import partial
from image_slicer import SliderImage
from slider import Slider

PICTURE_FILE = "irish_red_setter.jpg"


class SliderDisplay(tk.Frame):

    def __init__(self, parent, filename: Path, control_slider=None):
        super().__init__(parent)

        if control_slider is None:
            control_slider = Slider(n_rows=4)
            control_slider.shuffle()

        self.slider = control_slider
        self.images = SliderImage.add_blank(
            SliderImage(filename).partition_image(self.slider.n_rows, self.slider.n_cols))
        self.tk_images = {k: ImageTk.PhotoImage(v) for k, v in self.images.items()}
        self.coords = tuple((i, j) for i in range(self.slider.n_rows) for j in range(self.slider.n_cols))

        self.picture_buttons = [tk.Button(self,
                                          image=self.tk_images[pos],
                                          border=4,
                                          command=partial(self.img_button_clicked, pos))
                                for pos in self.coords]
        self.display_widgets()
        self.reorder_pictures(self.slider.data)

    def display_widgets(self):
        for i, pos in enumerate(self.coords):
            self.picture_buttons[i].grid(row=pos[0], column=pos[1])

    def reorder_pictures(self, order):
        sorted_coords = [self.coords[i] for i in order]
        for i, pos in enumerate(sorted_coords):
            self.picture_buttons[i].config(image=self.tk_images[pos])

    def img_button_clicked(self, pos):
        neighbours = self.slider.find_blank_neighbours()
        neighbours_swap = {tuple(value): key for key, value in neighbours.items()}
        slide_dir = neighbours_swap.get(pos)
        if slide_dir:
            self.slider.slide(slide_dir)
        self.reorder_pictures(self.slider.data)


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Slider')
    # root.resizable(False, False)
    img_path = Path('../resources/images', PICTURE_FILE)
    s_root = Slider(2, 3, [1, 3, 4, 5, 0, 2])
    # s_root.shuffle()
    main_frame = SliderDisplay(root, img_path, s_root)
    main_frame.pack()
    root.mainloop()
