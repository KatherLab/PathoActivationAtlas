from torchvision import transforms as tf
# none of the classes here should include normalization or conversion to a tensor


class ResizeTransform:
    def __init__(self, size, antialias=None):
        self.tfms = [tf.Resize(size, antialias=antialias)]
