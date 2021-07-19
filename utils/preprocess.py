import tensorflow as tf
from config import cfg


def preprocess(image, label=None):
    """
    Preprocess function for dataset.
    """
    if label is None:
        label = image["label"]
        image = image["image"]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (cfg.image_size, cfg.image_size), method="nearest")
    label = tf.one_hot(label, cfg.n_classes)
    return image, label


def prepare_dataset(dataset, batch_size=None, cache=None, num_workers=4):
    """
    Prepare the dataset.
    """
    ds = dataset.map(preprocess, num_parallel_calls=num_workers)
    if cache:
        ds = ds.cache(cache)
    ds = ds.shuffle(1024)
    if batch_size:
        ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
