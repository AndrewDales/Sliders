import tkinter as tk
from pathlib import Path
from PIL import ImageTk
from functools import partial
from image_slicer import SliderImages
from slider import Slider


PICTURE_FILE = "irish_red_setter.jpg"


class SliderDisplay(tk.Frame):
    n_dims = 3

    def __init__(self, parent, filename: Path, control_slider=None):
        super().__init__(parent)

        if control_slider is None:
            control_slider = Slider(n_dims=4)
            control_slider.shuffle()

        self.slider = control_slider
        self.images = SliderImages.add_blank(SliderImages(filename).partition_image(self.n_dims))
        self.tk_images = {k: ImageTk.PhotoImage(v) for k, v in self.images.items()}
        self.coords = tuple((i, j) for i in range(self.n_dims) for j in range(self.n_dims))

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
    img_path = Path('images', PICTURE_FILE)
    initial_order = [2, 8, 1, 0, 7, 4, 6, 5, 3]
    s_root = Slider(initial_order)
    main_frame = SliderDisplay(root, img_path, s_root)
    main_frame.pack()
    root.mainloop()
