# Repository Structure

### Define your dependencies

Define your dependencies in a requirements.txt (or a conda environment.yml) file.

Please make sure to always have a reference to the latest `adversarial-vision-challenge` like so:
`git+https://github.com/bethgelab/adversarial-vision-challenge`

### Implement a model server (see main.py)

To run a model server, use the `model_server` utility method from the above defined `adversarial-vision-challenge` library, as follows:

```
from adversarial_vision_challenge import model_server

foolbox_model = load_your_model()
model_server(foolbox_model)

```


### Define a crowdai.json

Define the following properties in the crowdai.json:

- challenge_id: "NIPS18-ADVERSARIAL-VISION-CHALLENGE"
- track: "TARGETED_ATTACK", "UNTARGETED_ATTACK" or "MODEL"
- authors: your crowdai username
- description: a description of your model


### Define a run.sh

Define the entrypoint command that should be run when the built docker image is started, i.e. in this case: `python main.py`.

---

For further fully functional model examples, check the following repos:

- ResNet18 TensorFlow Model: https://gitlab.crowdai.org/bveliqi/resnet18-baseline-model
- ResNet18 Frozen Noise TensorFlow Model: https://gitlab.crowdai.org/jonasrauber/resnet18-frozen-noise-baseline-model
- TODO: Link 3
- TODO: Link 4
- TODO: Link 5



# Test

Within the root-folder of this repository, simple run:

```avc-test-model --gpu 1 .```

Please always make sure that you have the newest version of the `adversarial-vision-challenge` library installed.


# Submit

Submissions are done by pushing an arbitraty git tag to your repository, by running something like:

```TAG=your_submission_tag bash -c 'git tag $TAG && git push origin $TAG'```

Please use a new tag for every new submission.