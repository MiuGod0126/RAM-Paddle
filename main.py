import paddle
import utils
import data_loader
from trainer import Trainer
from config import get_config


def main(config):
    utils.prepare_dirs(config)

    # ensure reproducibility
    utils.same_seeds(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        paddle.set_device("gpu")
        kwargs = {"num_workers": 1}

    # instantiate data loaders
    if config.is_train:
        dloader = data_loader.get_train_valid_loader(
            config.data_dir,
            config.batch_size,
            config.random_seed,
            config.valid_size,
            config.shuffle,
            config.show_sample,
            **kwargs,
        )
    else:
        dloader = data_loader.get_test_loader(
            config.data_dir, config.batch_size, **kwargs,
        )

    trainer = Trainer(config, dloader)

    # either train
    if config.is_train:
        utils.save_config(config)
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
