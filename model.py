class MockModel:

    def channel_axis(self):
        return 1

    def predictions(self):
        return 42

    def bounds(self):
        return (0, 255)


def create_mock_model():
    net = MockModel()
    return net
