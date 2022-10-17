from tkinter import *
from PIL import ImageTk, Image
import os
import json

from project_files import ImageGenerator


class JointsAnnotator:

    def __init__(self, im_folder_path):
        self.joints = ["Root", "Head", "Left Shoulder", "Left Arm", "Left Palm", "Right Shoulder", "Right Arm", "Right Palm",
          "Left Upper Leg", "Left Lower Leg", "Left Foot", "Right Upper Leg", "Right Lower Leg", "Right Foot"]
        self.parent_indices = [None, 0, 0, 2, 3, 0, 5, 6, 0, 8, 9, 0, 11, 12]
        if 'annotations.json' in os.listdir(im_folder_path):
            with open(f"{im_folder_path}\\annotations.json") as f:
                self.annotations = json.load(f)
        else:
            self.annotations = dict()
        self.joint_positions = dict()
        self.curr_joint_index = -1
        self.curr_joint_label = None
        self.curr_text_id = None
        self.canvas = None
        self.window = None
        self.folder_path = im_folder_path
        self.images = [path for path in os.listdir(self.folder_path) if path.endswith('.png')]
        self.curr_im_index = -1
        self.curr_im_name = None
        self.curr_im = None
        self.started = False
        self.ended = False

    def get_next_im(self):
        while True:
            self.curr_im_index += 1
            if self.curr_im_index == len(self.images):
                self.destroy_tool()
                return
            self.curr_im_name = self.images[self.curr_im_index]
            if self.curr_im_name not in self.annotations:
                break
            elif len(self.annotations[self.curr_im_name]) != len(self.joints):
                print(len(self.annotations[self.curr_im_name]))
                break
        with open(f"{self.folder_path}//annotations.json", "w") as outfile:
            json.dump(self.annotations, outfile)
        self.annotations[self.curr_im_name] = self.joint_positions
        self.curr_im = ImageTk.PhotoImage(Image.open(f"{self.folder_path}//{self.curr_im_name}"))
        self.canvas.create_image(0, 0, anchor=NW, image=self.curr_im)
        self.canvas.pack()
        self.move_to_next_joint()

    def get_joint(self, event):
        self.joint_positions[self.curr_joint_label] = (event.x, event.y)
        x1, y1 = (event.x - 2), (event.y - 2)
        x2, y2 = (event.x + 2), (event.y + 2)
        self.canvas.create_oval(x1, y1, x2, y2, fill="#00ff00")
        if self.parent_indices[self.curr_joint_index] is not None:
            px, py = self.joint_positions[self.joints[self.parent_indices[self.curr_joint_index]]]
            self.canvas.create_line(event.x, event.y, px, py, fill="#00ff00")
        self.canvas.pack()

        self.move_to_next_joint()

    def move_to_next_joint(self):
        self.curr_joint_index += 1
        if self.curr_joint_index == len(self.joints):
            self.curr_joint_index = -1
            self.joint_positions = dict()
            self.get_next_im()
            return
        self.curr_joint_label = self.joints[self.curr_joint_index]
        self.canvas.delete(self.curr_text_id)
        self.curr_text_id = self.canvas.create_text(ImageGenerator.char.image_size // 2, ImageGenerator.char.image_size // 10, text=self.curr_joint_label, fill="white",
                                                    font=('David 11'))
        self.canvas.pack()

    def start_annotation(self):
        self.window = Tk()
        self.window.geometry(f"{ImageGenerator.char.image_size}x{ImageGenerator.char.image_size}")

        self.canvas = Canvas(self.window, width=ImageGenerator.char.image_size, height=ImageGenerator.char.image_size)
        self.canvas.pack()

        # mouseclick event
        self.window.bind("<Button 1>", self.get_joint)

        self.get_next_im()

        if not self.ended:
            self.started = True
            self.window.mainloop()

    def destroy_tool(self):
        with open(f"{self.folder_path}//annotations.json", "w") as outfile:
            json.dump(self.annotations, outfile)
        if self.started:
            self.window.destroy()
        self.ended = True

    def convert_annotations(self):
        annotations_converted = []
        for i in range(len(self.annotations)):
            im = f"test_Pose{i + 1}.png"
            annotations_converted.append({"joints_vis": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        "joints": [list(self.annotations[im][part]) for part in
                                                   ImageGenerator.char.char_tree_array],
                                        "scale": 1,
                                        "center": list(self.annotations[im]["Root"]),
                                        "image": im})
        with open(r"AnnotationTool\annotations_converted.json", "w") as outfile:
            json.dump(annotations_converted, outfile)


if __name__ == "__main__":
    annotator = JointsAnnotator(r"AnnotationTool")
    annotator.start_annotation()
    annotator.convert_annotations()
    # annotations_renewed = []
    # with open(r"AnnotationTool\annotations.json") as f:
    #     annotations = json.load(f)
    # for i in range(len(annotations)):
    #     im = f"test_Pose{i + 1}.png"
    #     annotations_renewed.append({"joints_vis": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                                "joints": [list(annotations[im][part]) for part in ImageGenerator.char.char_tree_array],
    #                                "scale": 1,
    #                                "center": list(annotations[im]["Root"]),
    #                                "image": im})
    # with open(r"AnnotationTool\test.json", "w") as outfile:
    #     json.dump(annotations_renewed, outfile)

