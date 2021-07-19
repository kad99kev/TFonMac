import os
import shutil
import tempfile
import wandb
import tensorflow as tf
import tensorflow_datasets as tfds
from dataclasses import asdict
from .preprocess import prepare_dataset
from utils import info
from config import cfg


def train():
    """
    Training the model on the given dataset with WandB.
    """
    hardware = info.hardware()
    print(f"Using: {hardware}")

    train_dataset = tfds.load(name=cfg.dataset, as_supervised=True, split="train")
    test_dataset = tfds.load(name=cfg.dataset, as_supervised=True, split="test")

    wandb_config = asdict(cfg)
    wandb_config["train_size"] = len(train_dataset)
    wandb_config["test_size"] = len(test_dataset)

    with wandb.init(project="m1-benchmark", config=wandb_config) as run:
        cache = os.path.join(
            tempfile.mkdtemp(), str(hash(frozenset(run.config.items())))
        )

        base_model = getattr(tf.keras.applications, run.config.model)(
            input_shape=(
                run.config.image_size,
                run.config.image_size,
                run.config.image_channels,
            )
        )
        base_model.trainable = run.config.trainable

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            run.config.init_lr,
            decay_steps=run.config.train_size,
            decay_rate=run.config.decay,
        )

        model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(run.config.dropout),
                tf.keras.layers.Dense(run.config.n_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
            loss="categorical_crossentropy",
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )

        run.config.update(
            {
                "total_params": model.count_params(),
                "trainable_params": info.trainable_params(model),
            }
        )
        print(f"Model {run.config.model}:")
        print(f"Trainable Parameters: {run.config.trainable_params}")
        print(f"Total Parameters: {run.config.total_params}")
        print(f"Dataset: {run.config.dataset}")
        print(f"Training Size: {run.config.train_size}")
        print(f"Test Size: {run.config.test_size}")
        print(
            f"Image Size: {(run.config.image_size, run.config.image_size, run.config.image_channels)}"
        )

        print("Starting Training...")
        train_batches = prepare_dataset(
            train_dataset, batch_size=run.config.batch_size, cache=cache
        )
        test_batches = prepare_dataset(
            test_dataset, batch_size=run.config.batch_size, cache=cache
        )
        history = model.fit(
            train_batches,
            epochs=run.config.epochs,
            validation_data=test_batches,
            callbacks=[wandb.keras.WandbCallback(save_model=False)],
        )
        shutil.rmtree(os.path.dirname(run.config.cache))
