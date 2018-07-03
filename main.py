import foolbox
from fmodel import create_fmodel
from adversarial_vision_challenge import load_model
from adversarial_vision_challenge import read_images
from adversarial_vision_challenge import store_adversarial
from adversarial_vision_challenge import attack_complete


def run_attack(model, image, label):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label)


def main():
    # instantiate blackbox and substitute model
    forward_model = load_model()
    backward_model = create_fmodel()

    # instantiate differntiable composite model
    # (predictions from blackbox, gradients from substitute)
    model = foolbox.models.CompositeModel(
        forward_model=forward_model,
        backward_model=backward_model)

    for (file_name, image, label) in read_images():
        adversarial = run_attack(model, image, label)
        store_adversarial(file_name, adversarial)

    # Announce that the attack is complete
    # NOTE: In the absence of this call, your submission will timeout
    # while being graded.
    attack_complete()


if __name__ == '__main__':
    main()
