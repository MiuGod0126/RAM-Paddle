import numpy as np
from utils import plot_images
import paddle
from paddle.vision import transforms
from paddle.io import Subset  # 用于拆分训练和验证集


def get_train_valid_loader(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
):
    """Train and validation data loaders.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        random_seed: fix seed for reproducibility.
        valid_size: percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
            In the paper, this number is set to 0.1.
        shuffle: whether to shuffle the train/validation indices.
        show_sample: plot 9x9 sample grid of the dataset.
        num_workers: number of subprocesses to use when loading the dataset.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # define transforms
    # normalize = transforms.Normalize((0.1307,), (0.3081,))
    # trans = transforms.Compose([transforms.ToTensor(), normalize])
    trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    # load dataset
    dataset = paddle.vision.datasets.MNIST(mode='train', download=True, transform=trans)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(dataset=dataset, indices=train_idx)  # subset是dataset不是sampler
    valid_subset = Subset(dataset=dataset, indices=valid_idx)

    train_loader = paddle.io.DataLoader(
        train_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    valid_loader = paddle.io.DataLoader(
        valid_subset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # visualize some images
    if show_sample:
        sample_loader = paddle.io.DataLoader(
            dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        # data_iter = iter(sample_loader)
        images, labels = sample_loader[0]
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir, batch_size, num_workers=4):
    """Test datalaoder.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    # define transforms
    trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    # load dataset
    dataset = paddle.vision.datasets.MNIST(mode='test', download=True, transform=trans)

    data_loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return data_loader
