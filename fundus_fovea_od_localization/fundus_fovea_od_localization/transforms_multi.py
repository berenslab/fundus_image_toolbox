from typing import Any
import torch
import torchvision.transforms.functional as tf
import numpy as np


class Resize:
    """Resize the image and coordinate labels."""

    def __init__(self, new_size=(350, 350)):
        self.new_width = new_size[0]
        self.new_height = new_size[1]

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        f_x, f_y = f_label
        od_x, od_y = od_label
        original_width, original_height = image.size
        image_new = tf.resize(image, (self.new_height, self.new_width))

        f_x_new = f_x * self.new_width / original_width
        f_y_new = f_y * self.new_height / original_height
        f_label_new = (f_x_new, f_y_new)

        od_x_new = od_x * self.new_width / original_width
        od_y_new = od_y * self.new_height / original_height
        od_label_new = (od_x_new, od_y_new)

        return image_new, (f_label_new, od_label_new)


class ResizedCenterCrop:
    """Resize the image and coordinate labels.
    First crops a square from the center, then resizes.
    """

    def __init__(self, new_size=(350, 350)):
        self.new_width = new_size[0]
        self.new_height = new_size[1]

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        f_x, f_y = f_label
        od_x, od_y = od_label
        original_width, original_height = image.size

        image = tf.center_crop(image, original_height)  # assume landscape
        f_x, f_y = f_x - (original_width - original_height) / 2, f_y
        od_x, od_y = od_x - (original_width - original_height) / 2, od_y
        original_width = original_height

        image = tf.resize(image, (self.new_height, self.new_width))
        f_x_new = f_x * self.new_width / original_width
        f_y_new = f_y * self.new_height / original_height
        f_label_new = (f_x_new, f_y_new)

        od_x_new = od_x * self.new_width / original_width
        od_y_new = od_y * self.new_height / original_height
        od_label_new = (od_x_new, od_y_new)

        return image, (f_label_new, od_label_new)


class ResizedCenterCropAndPad:
    """Resize the image and coordinate labels.
    First crops a square from the center, then resizes. Additionally pads black to top and bottom,
    if needed.
    """

    def __init__(self, new_size=(350, 350)):
        self.new_width = new_size[0]
        self.new_height = new_size[1]

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        f_x, f_y = f_label
        od_x, od_y = od_label

        if self._is_cut(image):
            pad = 50

            # Resize the image to a height of new_size + pad keeping the aspect ratio
            w, h = image.size
            ratio = w / h
            new_h = self.new_height + pad
            new_w = int(new_h * ratio)

            image = tf.resize(image, (new_h, new_w))

            f_x, f_y = f_x * new_w / w, f_y * new_h / h
            od_x, od_y = od_x * new_w / w, od_y * new_h / h

            assert (
                image.size[1] == new_h
            ), f"Image height should be {new_h}, but is {image.size[1]}"

            # Pad the image top and bottom
            image = tf.pad(
                image, (0, pad // 2, 0, pad // 2), padding_mode="constant", fill=0
            )
            f_y += pad // 2
            od_y += pad // 2

        f_label = (f_x, f_y)
        od_label = (od_x, od_y)

        image, (f_label, od_label) = ResizedCenterCrop(
            (self.new_width, self.new_height)
        )((image, (f_label, od_label)))

        return image, (f_label, od_label)

    def _is_cut(self, pil_image, thresh=0.5):
        # Center crop the image, then check if the top and bottom have enough black or are cut off
        image, (_, _) = ResizedCenterCrop((350, 350))((pil_image, ((0, 0), (0, 0))))
        image = np.array(image)

        top = image[:10, :, :]
        bottom = image[-10:, :, :]

        frac1 = np.mean(top == 0)
        frac2 = np.mean(bottom == 0)
        frac = (frac1 + frac2) / 2
        return frac < thresh


class RandomZoom:
    """Zoom a bit out of the image with probability p and adjust coordinate labels."""

    def __init__(self, p=0.5, max_zoom=0.25):
        if not 0 <= p <= 1:
            raise ValueError(
                f"Variable p is a probability, should be float between 0 to 1"
            )
        self.p = p  # float between 0 to 1 represents the probability of zooming
        if not -1 <= max_zoom <= 1:
            raise ValueError(f"Variable max_zoom should be float between -1 to 1")
        self.max_zoom = max_zoom

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        w, h = image.size
        f_x, f_y = f_label
        od_x, od_y = od_label
        if np.random.random() < self.p:
            if self.max_zoom < 0:
                zoom_factor = 1 - np.random.uniform(0, -self.max_zoom)
            else:
                zoom_factor = 1 + np.random.uniform(0, self.max_zoom)
            image = tf.affine(
                image, translate=(0, 0), angle=0, scale=zoom_factor, shear=0
            )
            f_label = f_x * zoom_factor, f_y * zoom_factor
            od_label = od_x * zoom_factor, od_y * zoom_factor
            # Shouldn't it be center_x + (x - center_x) * zoom_factor?
        return image, (f_label, od_label)


class RandomHorizontalFlip:
    """Horizontal flip the image with probability p and adjust coordinate labels."""

    def __init__(self, p=0.5):
        if not 0 <= p <= 1:
            raise ValueError(
                f"Variable p is a probability, should be float between 0 to 1"
            )
        self.p = p  # float between 0 to 1 represents the probability of flipping

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        w, h = image.size
        f_x, f_y = f_label
        od_x, od_y = od_label
        if np.random.random() < self.p:
            image = tf.hflip(image)
            f_label = w - f_x, f_y
            od_label = w - od_x, od_y
        return image, (f_label, od_label)


class RandomVerticalFlip:
    """Vertically flip the image with probability p and adjust coordinate labels."""

    def __init__(self, p=0.5):
        if not 0 <= p <= 1:
            raise ValueError(
                f"Variable p is a probability, should be float between 0 to 1"
            )
        self.p = p  # float between 0 to 1 represents the probability of flipping

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        w, h = image.size
        f_x, f_y = f_label
        od_x, od_y = od_label
        if np.random.random() < self.p:
            image = tf.vflip(image)
            f_label = f_x, h - f_y
            od_label = od_x, h - od_y
        return image, (f_label, od_label)


class RandomTranslation:
    """Translate the image and adjust the coordinate labels accordingly."""

    def __init__(self, max_translation=(0.1, 0.1)):
        if (not 0 <= max_translation[0] <= 1) or (not 0 <= max_translation[1] <= 1):
            raise ValueError(f"Variable max_translation should be float between 0 to 1")
        self.max_translation_x = max_translation[0]
        self.max_translation_y = max_translation[1]

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        w, h = image.size
        f_x, f_y = f_label
        od_x, od_y = od_label
        x_translate = int(
            np.random.uniform(-self.max_translation_x, self.max_translation_x) * w
        )
        y_translate = int(
            np.random.uniform(-self.max_translation_y, self.max_translation_y) * h
        )
        image = tf.affine(
            image, translate=(x_translate, y_translate), angle=0, scale=1, shear=0
        )
        f_label = f_x + x_translate, f_y + y_translate
        od_label = od_x + x_translate, od_y + y_translate
        return image, (f_label, od_label)


class RandomRotation:
    """Rotate the image and adjust the coordinate labels accordingly."""

    def __init__(self, max_angle=15):
        if not 0 <= max_angle <= 180:
            raise ValueError(f"Variable max_angle should be float between 0 to 180")
        self.max_angle = max_angle

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        w, h = image.size
        f_x, f_y = f_label
        od_x, od_y = od_label
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        image = tf.affine(image, translate=(0, 0), angle=angle, scale=1, shear=0)
        f_label = (f_x - w / 2) * np.cos(np.radians(angle)) - (f_y - h / 2) * np.sin(
            np.radians(angle)
        ) + w / 2, (f_x - w / 2) * np.sin(np.radians(angle)) + (f_y - h / 2) * np.cos(
            np.radians(angle)
        ) + h / 2
        od_label = (od_x - w / 2) * np.cos(np.radians(angle)) - (od_y - h / 2) * np.sin(
            np.radians(angle)
        ) + w / 2, (od_x - w / 2) * np.sin(np.radians(angle)) + (od_y - h / 2) * np.cos(
            np.radians(angle)
        ) + h / 2

        return image, (f_label, od_label)


class RandomSqueeze:
    """Squeeze the image inside a range of values in x and y dimensions and adjust the coordinate
    labels accordingly.
    """

    def __init__(self, max_squeeze=(0.1, 0.1)):
        if (not 0 <= max_squeeze[0] <= 1) or (not 0 <= max_squeeze[1] <= 1):
            raise ValueError(f"Variable max_squeeze should be float between 0 to 1")
        self.max_squeeze_x = max_squeeze[0]
        self.max_squeeze_y = max_squeeze[1]

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        w, h = image.size
        f_x, f_y = f_label
        od_x, od_y = od_label
        x_squeeze = 1 + np.random.uniform(-self.max_squeeze_x, self.max_squeeze_x)
        y_squeeze = 1 + np.random.uniform(-self.max_squeeze_y, self.max_squeeze_y)
        image, (f_label, od_label) = Resize((int(w * x_squeeze), int(h * y_squeeze)))(
            image_label_sample
        )
        f_label = f_x * x_squeeze, f_y * y_squeeze
        od_label = od_x * x_squeeze, od_y * y_squeeze

        return image, (f_label, od_label)


class ImageAdjustment:
    """Change the brightness and contrast of the image and apply Gamma correction.
    No need to change the label."""

    def __init__(
        self, p=0.5, brightness_factor=0.8, contrast_factor=0.8, gamma_factor=0.4
    ):
        if not 0 <= p <= 1:
            raise ValueError(
                f"Variable p is a probability, should be float between 0 to 1"
            )
        self.p = p
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.gamma_factor = gamma_factor

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]

        if np.random.random() < self.p:
            brightness_factor = 1 + np.random.uniform(
                -self.brightness_factor, self.brightness_factor
            )
            image = tf.adjust_brightness(image, brightness_factor)

        if np.random.random() < self.p:
            contrast_factor = 1 + np.random.uniform(
                -self.brightness_factor, self.brightness_factor
            )
            image = tf.adjust_contrast(image, contrast_factor)

        if np.random.random() < self.p:
            gamma_factor = 1 + np.random.uniform(
                -self.brightness_factor, self.brightness_factor
            )
            image = tf.adjust_gamma(image, gamma_factor)

        return image, (f_label, od_label)


class ToTensor:
    """Convert image and labels to tensors."""

    def __init__(self, scale_labels=True):
        self.scale_labels = scale_labels

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        w, h = image.size
        f_x, f_y = f_label
        od_x, od_y = od_label

        image = tf.to_tensor(image)

        if self.scale_labels:
            f_label = f_x / w, f_y / h
            od_label = od_x / w, od_y / h
        f_label = torch.tensor(f_label, dtype=torch.float32)
        od_label = torch.tensor(od_label, dtype=torch.float32)

        return image, (f_label, od_label)


class ToPILImage:
    """Convert a tensor image to a PIL image and convert coordinate labels to float (-lists)."""

    def __init__(self, unscale_labels=True):
        self.unscale_labels = unscale_labels

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        f_label, od_label = image_label_sample[1]
        f_label, od_label = f_label.tolist(), od_label.tolist()

        image = tf.to_pil_image(image)
        w, h = image.size

        if self.unscale_labels:
            f_x, f_y = f_label
            od_x, od_y = od_label

            f_label = f_x * w, f_y * h
            od_label = od_x * w, od_y * h

        return image, (f_label, od_label)
