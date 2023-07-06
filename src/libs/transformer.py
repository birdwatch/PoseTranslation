import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # images, flows, labels = sample["images"], sample["flows"], sample["labels"]
        # images = images.transpose((0, 3, 1, 2))
        # flows = flows.transpose((0, 3, 1, 2))
        # return {
        #     "images": torch.from_numpy(images).float().div(255.0),
        #     "flows": torch.from_numpy(flows).float().div(255.0),
        #     "labels": torch.from_numpy(labels),
        # }
        images, labels = sample["images"], sample["labels"]
        images = images.transpose((0, 3, 1, 2))
        return {
            "images": torch.from_numpy(images).float().div(255.0),
            "labels": torch.from_numpy(labels),
        }


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        # images, flows, labels = sample["images"], sample["flows"], sample["labels"]
        # images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        # flows.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        # return {"images": images, "flows": flows, "labels": labels}
        images, labels = sample["images"], sample["labels"]
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {"images": images, "labels": labels}
