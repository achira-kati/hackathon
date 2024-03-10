"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os

import numpy as np
from PIL import Image

from synthtiger import components, layers, templates


class Multiline(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.count = config.get("count", 100)
        self.bgsize = config.get("bgsize", {})
        self.corpus = components.BaseCorpus(**config.get("corpus", {}))
        self.font = components.BaseFont(**config.get("font", {}))
        self.color = components.RGB(**config.get("color", {}))
        self.layout = components.FlowLayout(**config.get("layout", {}))

        self.transform = components.Switch(
            components.Selector(
                [
                    components.Skew(),
                    components.Skew(),
                    components.Rotate(),
                ]
            ),
            **config.get("transform", {}),
        )

        self.texteffect = components.Iterator(
            [
                components.Switch(components.TextBorder()),
            ],
            **config.get("texteffect", {}),
        )

        self.postprocess = components.Iterator(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.Brightness()),
                components.Switch(components.Contrast()),
                components.Switch(components.MotionBlur()),
            ],
            **config.get("postprocess", {}),
        )

    def generate(self):
        texts = [self.corpus.data(self.corpus.sample()) for _ in range(self.count)]
        font = self.font.sample()
        color = self.color.data(self.color.sample())

        text_group = layers.Group(
            [
                layers.TextLayer(text, color=color, **font)
                for text in texts
            ]
        )
        self.layout.apply(text_group)
        self.transform.apply(text_group)
        self.texteffect.apply(text_group)

        bg_layer = layers.RectLayer(self.bgsize, (243, 230, 178, 255))
        
        topleft_x, topleft_y = bg_layer.topleft
        midtop_x, midtop_y = bg_layer.midtop
        new_position_x = (topleft_x + midtop_x) / 2
        new_position_y = (topleft_y + midtop_y) / 2

        text_group.midtop = (new_position_x, new_position_y + 30)

        self.postprocess.apply(text_group + bg_layer)
        
        image = (text_group + bg_layer).output()
        label = " ".join(texts)


        data = {
            "image": image,
            "label": label,
            "bboxes": [layer.bbox for layer in text_group]
        }

        return data

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)
        gt_path = os.path.join(root, "gt.txt")
        self.gt_file = open(gt_path, "w", encoding="utf-8")

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        bboxes = data["bboxes"]

        image_key = os.path.join("images", "train", f"{idx}.jpg")
        label_key = os.path.join("labels", "train", f"{idx}.txt")

        image_path = os.path.join(root, image_key)
        label_path = os.path.join(root, label_key)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_path, quality=95)

        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, 'w') as f:
            img_width, img_height = image.size
            bboxes = [[max(0, x), max(0, y), max(0, w), max(0, h)] for x, y, w, h in bboxes]
            coords = [[(x + w/2) / img_width, (y + h/2) / img_height, w / img_width, h / img_height] for x, y, w, h in bboxes]
            coords = " ".join([",".join(map(str, coord)) for coord in coords])
            for coord in coords.split(" "):
                f.write(f"0 {coord.replace(',', ' ')}\n")

        self.gt_file.write(f"{image_key}\t{label}\n")

    def end_save(self, root):
        self.gt_file.close()
