# Mutual Adversarial Training with Early-stop black-box Adversarial Examples

Our team's code for the NIPS2018 vision adversarial robustness competition - model track. (Team Wilson, 2nd place)

Training codes are under `train` branch, some training configuration files are under `cfgs/`.

Local/docker evaluation codes are under `master` branch.

Our final submission (foolbox model-zoo compatible) is in https://github.com/walkerning/nips18comp_submission

## Techniques
**Training**
* Gaussian/Salt-and-Pepper noises
* Distill Adversarial Training/High level guidance (not useful in the query-based attack threat model)
* Mutual Adversarial Training
* Early-stop blackbox adversarial examples
* Gray-box adversarial training (we do not find this help)


**Inference**
* Model Ensemble
* Multi-crop/scale forward (do not help as the attacker can query many times)
