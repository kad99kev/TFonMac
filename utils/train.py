import os
import shutil
import tempfile
from turtle import st
from click import style
import wandb
import tensorflow as tf
import tensorflow_datasets as tfds
from dataclasses import asdict
from .preprocess import prepare_dataset
from utils import info
from config import cfg
from rich.console import Console

console = Console()


def train():
    """
    Training the model on the given dataset with WandB.
    """
    hardware = info.hardware()

    train_dataset = tfds.load(name=cfg.dataset, as_supervised=True, split="train")
    test_dataset = tfds.load(name=cfg.dataset, as_supervised=True, split="test")

    wandb_config = asdict(cfg)
    wandb_config["train_size"] = len(train_dataset)
    wandb_config["test_size"] = len(test_dataset)
    wandb_config["hardware"] = hardware

    with wandb.init(project="m1-benchmark", config=wandb_config) as run:
        cache = os.path.join(
            tempfile.mkdtemp(), str(hash(frozenset(run.config.items())))
        )

        base_model = getattr(tf.keras.applications, run.config.model)(
            input_shape=(
                run.config.image_size,
                run.config.image_size,
                run.config.image_channels,
            ),
            include_top=False,
            weights="imagenet",
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

        console.print(f"Using: {hardware}", style="bright_magenta")
        console.print(f"Model: {run.config.model}", style="bright_yellow")
        console.print(
            f"Trainable Parameters: {run.config.trainable_params}",
            style="deep_sky_blue2",
        )
        console.print(
            f"Total Parameters: {run.config.total_params}", style="deep_sky_blue1"
        )
        console.print(f"Dataset: {run.config.dataset}", style="spring_green2")
        console.print(f"Training Size: {run.config.train_size}", style="spring_green1")
        console.print(f"Test Size: {run.config.test_size}", style="medium_spring_green")
        console.print(
            f"Image Size: {(run.config.image_size, run.config.image_size, run.config.image_channels)}",
            style="pink1",
        )

        console.log("Starting Training...", style="bright_red")
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
