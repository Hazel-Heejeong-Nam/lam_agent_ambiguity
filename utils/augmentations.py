import torch
import torch.nn.functional as F
import torchvision.transforms as TF


class RandomShiftAug:
    def __init__(self, pad=14):
        self.pad = pad

    def __call__(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class RandomRotateAug:
    def __init__(self, pad=17, degrees=180) -> None:
        self.degrees = degrees
        self.pad = pad
        self.rotate = TF.RandomRotation(degrees=degrees, interpolation=TF.InterpolationMode.BILINEAR, expand=False)

    def __call__(self, x):
        n, c, h, w = x.shape
        x = TF.functional.pad(x, self.pad, padding_mode="edge")
        x = self.rotate(x)
        x = TF.functional.center_crop(x, output_size=(h, w))
        return x


class RandomPerspectiveAug:
    def __init__(self, pad=30, scale=0.5, p=1.0) -> None:
        self.pad = pad
        self.perspective = TF.RandomPerspective(
            distortion_scale=scale,
            p=p,
        )

    def __call__(self, x):
        n, c, h, w = x.shape
        x = TF.functional.pad(x, self.pad, padding_mode="edge")
        x = self.perspective(x)
        x = TF.functional.center_crop(x, output_size=(h, w))
        return x


class ComposeAugs:
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, x):
        for aug in self.augs:
            x = aug(x)
        return x


def get_aug(aug_choice, img_resolution):
    if img_resolution == 64:
        rotate_pad, shift_pad = 17, 14
    elif img_resolution == 128 or img_resolution == 256:
        rotate_pad, shift_pad = 28, 28
    else:
        raise RuntimeError("only images with 64px and 128px are supported")

    augs = {
        "rotate": RandomRotateAug(rotate_pad, degrees=90),
        "shift": RandomShiftAug(shift_pad),
        "perspective": RandomPerspectiveAug(),
        "nothing": lambda x: x,
    }
    composed_aug = ComposeAugs(augs=[augs[aug] for aug in aug_choice.split("-")])
    return composed_aug


class Augmenter:
    def __init__(self, img_resolution):
        # yes, this is not optimal way of doing this...
        self.augs = [
            get_aug("nothing", img_resolution),
            get_aug("shift", img_resolution),
            get_aug("rotate", img_resolution),
            get_aug("perspective", img_resolution),
            get_aug("shift-rotate", img_resolution),
            get_aug("rotate-shift", img_resolution),
            get_aug("rotate-perspective", img_resolution),
            get_aug("perspective-rotate", img_resolution),
        ]
        self.num_augs = len(self.augs)

    def __call__(self, x):
        x_ = x.clone()
        n, c, h, w = x_.shape

        # Select one aug for each image uniformly random
        image_augs = torch.randint(self.num_augs, (n,))
        # Apply each aug
        for i, apply_aug in enumerate(self.augs):
            # Augment the indicies
            selected_inds = image_augs == i
            if torch.sum(selected_inds) > 0:
                x_[selected_inds] = apply_aug(x_[selected_inds])

        return x_