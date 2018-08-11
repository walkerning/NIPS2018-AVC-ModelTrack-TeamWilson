from fmodel import create_fmodel
from adversarial_vision_challenge import model_server


if __name__ == '__main__':
    fmodel = create_fmodel()
    model_server(fmodel)