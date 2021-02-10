import transforms as T

def get_transform(train):
    base_size = crop_size = 385
    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

