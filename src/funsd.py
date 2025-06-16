import os
import json

def collect_funsd_data(data_dir, label2id):
    ann_dir = os.path.join(data_dir, "annotations")
    img_dir = os.path.join(data_dir, "images")
    data = []
    for ann_file in os.listdir(ann_dir):
        if ann_file.endswith(".json"):
            with open(os.path.join(ann_dir, ann_file), "r") as f:
                ann = json.load(f)
            words = []
            bboxes = []
            labels = []
            for field in ann["form"]:
                label = field["label"]
                label_id = label2id[label]
                for word in field["words"]:
                    labels.append(label_id)
                    words.append(word["text"])
                    bboxes.append(word["box"])
            image_filename = ann_file.replace(".json", ".png")
            image_path = os.path.join(img_dir, image_filename)
            data.append({
                "id": ann_file,
                "words": words,
                "bboxes": bboxes,
                "labels": labels,
                "image_path": image_path,
            })
    return data

