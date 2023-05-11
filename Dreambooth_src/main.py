import torch
import csv
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

# Setup the Classes
from pathlib import Path
from torchvision import transforms

class DreamBoothDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        index = [],
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.index = index
        self.instance_images_path = []
            
        index_len = len(self.index)
        file = open('../CheXpert-v1.0-small/train.csv')
        csvreader = csv.reader(file)
        for row in csvreader:
            try_num = 0
            if row[3] == "Frontal":
                for i in index:
                    if row[i] == "1.0":
                        try_num += 1
                if try_num == index_len:
                    self.instance_images_path.append(f"../{row[0]}")

        self.num_instance_images = len(self.instance_images_path)
        print(f"{self.num_instance_images} images for training")
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_text = Path(self.instance_images_path[index % self.num_instance_images])
        id = str(instance_text.relative_to(*instance_text.parts[:1]))

        file = open('../CheXpert-v1.0-small/train.csv')
        csvreader = csv.reader(file)
        header = []
        header = next(csvreader)
        for row in csvreader:
            if row[0] == id:
                text = f"a photo of chest x-ray"
                for i, n in enumerate(row):
                    if n == "1.0":
                        text += f", {header[i]}"
                # print(text)
        
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        
        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example