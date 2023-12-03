# image_browser.py
import glob
import PySimpleGUI as sg
from PIL import Image, ImageTk
from main import *


def load_image(path, window):
    try:
        img = image2lab(path)
        gray = lab2grayscale(img)
        gray = (gray * 255).astype(np.uint8)
        image = Image.fromarray(gray, mode=None)
        image.thumbnail((400, 400))
        photo_img = ImageTk.PhotoImage(image)
        window["image"].update(data=photo_img)
    except:
        print("unable to open file")


def color_image(path, window):
    try:
        img = color(path)
        img = (img * 255).astype(np.uint8)
        image = Image.fromarray(img, mode=None)
        image.thumbnail((400, 400))
        photo_img = ImageTk.PhotoImage(image)
        window["color"].update(data=photo_img)
    except:
        print("unable to open file")

def main():
    elements = [
        [
            sg.Image(
                key="image",
            ),
            sg.Image(key="color", expand_y=True),
        ],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), enable_events=True, key="file"),
            sg.FileBrowse(
                file_types=(
                    (
                        (
                            "All Picture Files",
                            "*.jpg *.png",
                        ),
                    )
                )
            ),
        ],
        [sg.Button("Colorize")],
    ]
    window = sg.Window("Portrait Colorizer", elements, size=(650, 500))
    images = False
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "file":
            load_image(values["file"], window)
            images = True
        if event == "Colorize" and images:
            print("hi")
            color_image(values["file"], window)
    window.close()


if __name__ == "__main__":
    main()
