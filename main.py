from adversarial_vision_challenge import model_server
from foolbox.models import PyTorchModel
from model import create_mock_model


if __name__ == '__main__':
    '''
        Load your own model here.


        Normally you would load a persisted model using your library of choice
        and wrap it into a foolbox model. E.g. for PyTorch:

        model = YourModel(...)
        model.load_state_dict(torch.load('/path/to/your/model'))
        foolbox.models.PyTorchModel(model, ...)

    '''
    foolbox_model = create_mock_model()
    model_server(foolbox_model)
