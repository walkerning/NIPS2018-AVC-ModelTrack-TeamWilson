import foolbox
from fmodel import create_fmodel
from adversarial_vision_challenge import model_server, load_model, read_images, store_adversarial


def run_attack(model, image, label):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.IterativeGradientAttack(model, criterion)    
    return attack(image, label, epsilons=35, steps=12)


def main():
    # instantiate blackbox and substitute model
    forward_model = load_model()
    backward_model = create_fmodel()
    
    # instantiate differntiable composite model (predictions from blackbox, gradients from substitute)
    model = foolbox.models.CompositeModel(forward_model=forward_model, backward_model=backward_model)

    for (file_name, image, label) in read_images():
        adversarial = run_attack(model, image, label)
        store_adversarial(file_name, adversarial)


if __name__ == '__main__':
    main()