import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import showinfo
from pathlib import Path
from PIL import ImageTk
from functools import partial
import imghdr
from slider import SliderNode, find_full_route
from image_slicer import prepare_slider_images
from pprint import pprint

PICTURE_FILE = "irish_red_setter.jpg"
DEFAULT_SIZE = (6, 6)


class SliderApplication(tk.Frame):
    """ SliderApplication is a container frame that includes the SliderDisplay and the SliderControls"""

    def __init__(self, master=None, image_filename: Path = None):
        super().__init__(master)
        if image_filename is None:
            image_filename = Path('..', 'resources', 'images') / 'irish_red_setter.jpg'
        self.image_filename = image_filename
        self.slider = SliderNode(*DEFAULT_SIZE)

        # self.slider = SliderNode(3, 4, [0, 1, 2, 3, 7, 6, 5, 4, 11, 10, 9, 8])
        self.callbacks = {'change_size': self.change_size,
                          'shuffle': self.shuffle,
                          'tile_press': [partial(self.tile_press, i) for i in sorted(self.slider.data)],
                          'solve': self.solve_puzzle,
                          'openfile': self.open_image_file,
                          }

        self.title = ttk.Label(self, text="Slider Puzzle and Solver", style='Title.TLabel')
        self.frames = {'display': SliderDisplay(self),
                       'controls': SliderControls(self),
                       }
        self.solution_label = ttk.Label(self, text="")
        self.solution_path = None

        self.title.grid(row=0, column=0, columnspan=2, padx=20, pady=20)
        self.frames['display'].grid(row=1, column=0, padx=20, pady=10)
        self.frames['controls'].grid(row=1, column=1, padx=20, pady=20)
        self.solution_label.grid(row=2, column=0, columnspan=2, padx=20, pady=(10, 20))
        self.shuffle()

        # style
        self.style = ttk.Style(self)
        self.set_styles()

    def set_styles(self):
        #  self.style.theme_use("classic")

        # Button style
        self.style.configure('TButton', font=('Helvetica', 12))

        # Label style
        self.style.configure('TLabel', font=('Helvetica', 12))

        # Option style
        self.style.configure('TMenubutton', font=('Helvetica', 12))

        # Slider button style
        self.style.configure('Slider.TButton',
                             background="midnightblue",
                             padding=0
                             )

        # heading style
        self.style.configure('Title.TLabel', font=('Helvetica', 20))

    def shuffle(self):
        self.slider.shuffle()
        self.frames['display'].display_widgets()
        self.solution_label.config(text="")

    def tile_press(self, button):
        # button is the index of the button in the original order
        # use index to find the location of this button in the current slider
        button_press_loc = self.slider.data.index(button)
        if button_press_loc in self.slider.find_blank_neighbours().values():
            # swap the location of the button in the current slider with the location of the blank
            self.slider._swap(button_press_loc, self.slider.blank_pos)
            self.frames['display'].display_widgets()
        if self.slider.is_solved:
            self.solution_label.config(text="Congratulations - Slider Puzzle is solved")
        else:
            self.solution_label.config(text="")

    def solve_puzzle(self):
        # self.disable_widgets(self)
        self.slider.path = ""
        s_solved, timings = find_full_route(self.slider)
        self.solution_path = iter(s_solved.path)
        self.show_path()

    def show_path(self):
        try:
            letter = next(self.solution_path)
            self.slider.slide(letter)
            self.frames['display'].display_widgets()
            self.after(100, self.show_path)
        except StopIteration:
            pass
            # self.draw_puzzle()

    def open_image_file(self):
        filetypes = (
            ('image files', '*.png *.jpg'),
            ('All files', '*.*'),
        )

        filename = filedialog.askopenfilename(
            title="Open an image file",
            initialdir=Path('..', 'resources', 'images'),
            filetypes=filetypes,
        )

        if filename and imghdr.what(filename):
            self.image_filename = filename
            self.shuffle()
            self.draw_puzzle()

    def change_size(self, event):
        size_strings = self.frames['controls'].size_var.get()
        size_strings = size_strings.split()
        n_rows = int(size_strings[0])
        n_cols = int(size_strings[-1])
        self.slider = SliderNode(n_rows, n_cols)
        self.callbacks['tile_press'] = [partial(self.tile_press, i) for i in sorted(self.slider.data)]
        self.draw_puzzle()
        self.shuffle()

    def draw_puzzle(self):
        self.frames['display'].grid_forget()
        self.frames['display'] = SliderDisplay(self)
        self.frames['display'].grid(row=1, column=0, padx=20, pady=10)
        # self.shuffle()

    # Todo - widget commmands are turned off on solve, but are not turned back on again.
    def disable_widgets(self, widget):
        for frm in self.frames.values():
            for child in frm.winfo_children():
                if child.winfo_class() == "TButton":
                    child.configure(command="")

    def print_hello(self):
        print("hello")


class SliderDisplay(tk.Frame):
    def __init__(self, parent):
        # super().__init__(parent, highlightthickness=10, highlightcolor="black")
        super().__init__(parent, borderwidth=15, relief=tk.RIDGE)

        self.image_filepath = parent.image_filename
        self.parent = parent
        self.tk_images = []
        self.blank_image = []

        self.coords = tuple((row, column) for row in range(self.parent.slider.n_rows)
                            for column in range(self.parent.slider.n_cols))
        self.picture_buttons = self.create_buttons()
        self.display_widgets()

    def create_buttons(self):
        pil_part = prepare_slider_images(self.image_filepath,
                                         n_rows=self.parent.slider.n_rows,
                                         n_cols=self.parent.slider.n_cols)
        self.tk_images = [ImageTk.PhotoImage(img) for img in pil_part["images"]]
        self.blank_image = ImageTk.PhotoImage(pil_part['blank'])
        picture_buttons = [ttk.Button(self,
                                      image=img,
                                      style='Slider.TButton',
                                      command=self.parent.callbacks['tile_press'][i],
                                      )
                           for i, img in enumerate(self.tk_images)]
        picture_buttons[self.parent.slider.blank].config(image=self.blank_image
                                                         )
        return picture_buttons

    def display_widgets(self):
        """ Show the buttons in the correct locations"""
        for i, button in enumerate(self.picture_buttons):
            # button i goes in the place where i sits in slider.data
            button_pos_index = self.parent.slider.data.index(i)
            button_pos = self.coords[button_pos_index]
            button.grid(row=button_pos[0], column=button_pos[1])


class SliderControls(tk.Frame):
    BUTTON_WIDTH = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.puzzle_sizes = ('3 by 3', '4 by 4', '5 by 5', '6 by 6', '7 by 7', '8 by 8')
        self.size_var = tk.StringVar(self)
        self.widgets = self.create_widgets()
        self.place_widgets()

    def create_widgets(self):
        widgets = {
            'size_label': ttk.Label(self, text="Choose Puzzle Size:"),
            "size": ttk.OptionMenu(self, self.size_var, self.puzzle_sizes[3], *self.puzzle_sizes,
                                   command=self.parent.callbacks['change_size']),
            "shuffle": ttk.Button(self, text="Shuffle", command=self.parent.callbacks['shuffle']),
            "solve": ttk.Button(self, text="Solve", command=self.parent.callbacks['solve']),

            "picture": ttk.Button(self, text="Select Picture", command=self.parent.callbacks['openfile']),
            "close": ttk.Button(self, text="Quit", command=root.destroy),
        }
        for widget in widgets.values():
            if widget.winfo_class() == 'Button':
                widget.config(width=self.BUTTON_WIDTH)
        return widgets

    def place_widgets(self):
        for i, widget in enumerate(self.widgets.values()):
            widget.grid(row=i, column=0, padx=10, pady=10)


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Slider App')
    # root.resizable(False, False)
    main_frame = SliderApplication(root)
    main_frame.pack()
    root.mainloop()
