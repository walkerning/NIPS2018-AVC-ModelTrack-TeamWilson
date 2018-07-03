import numpy as np


# IMPORTANT
# ---------
# While it is in principle possible to implement a model class
# followin our conventions yourself, you should usually use
# one of our model wrappers from Foolbox. Those wrappers support
# Keras, PyTorch, TensorFlow, MXNet, Theano and more.
# https://github.com/bethgelab/foolbox


class MockModel:
    """
        This is just a mock.
        For a submission, you would normally use a foolbox wrapped model here,
        based on your framework of choice.
        For more details, have a look here:
        https://foolbox.readthedocs.io/en/latest/modules/models.html
    """

    def channel_axis(self):
        # your model must return 1 or 3 as the channel axis;
        # PyTorch ususally uses 1 (NCHW), TensorFlow 3 (NHWC);
        # if channel_axis is 1, predictions will be called with
        # images that have (3, 64, 64) as their shape;
        # if channel_axis is 3, predictions will be called with
        # images that have (64, 64, 3) as their shape
        return 1

    def predictions(self, image):
        # your model should expect to get a numpy array
        # with dtype float32 and values between 0 and 255
        # as input (image argument);
        # for details regarding the shape, see channel_axis

        lower_bound = self.bounds()[0]
        upper_bound = self.bounds()[1]
        prediction = np.random.randint(lower_bound, upper_bound)

        # your model should either return the predicted class
        # as an integer in [0, 199] or as a 200-dimensional vector
        # of e.g. logits or probabilities, from which the argmax
        # will be taken
        return prediction

    def bounds(self):
        # your model must return (0, 255) as bounds
        # and expect images with values between 0 and 255,
        # see predictions
        return (0, 255)


def create_mock_model():
    net = MockModel()
    return net
