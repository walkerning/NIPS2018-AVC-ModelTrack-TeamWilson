## Dirs and Files

* `run.sh` and `main.py`: entrypoint for docker-run, not neccesary when doing local test.
* `fmodel.py`: the ensemble fmodel.
* `benchmark/`: benchmark and plot scripts.
* `dmodels/`: models that encapsulate foolbox models.
  * **TF** resnet: resnet-20
  * **TF** inception/inception64: inception of input size 64
  * **TF** inception_res_v2: inception-res-v2 of input size 64
  * **TF** vgg: vgg of input size 64
  * **TF** denoise: encoder-decoder type denoiser
  * **TF** seq_tf: a sequential list of tf models
  * **caffe** caffe: caffe net, the net structure will be defined in the prototxt (whose path will be specified in the configuration)

