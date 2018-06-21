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

For a more detailed description please check:
TODO: Link

---

For further fully functional model examples, check the following repos:

- ResNet18 TensorFlow Model: https://gitlab.crowdai.org/bveliqi/resnet18-baseline-model
- TODO: Link 2
- TODO: Link 3
- TODO: Link 4
- TODO: Link 5



# Test

TODO: Describe test package here.

```
docker rm -f model-template || true
nvidia-docker run -d \
      --shm-size 3G \
      --name model-template \
      -e NUM_OF_IMAGES=10 \
      -it bveliqi/model-template:0.1 \
      bash run.sh
```