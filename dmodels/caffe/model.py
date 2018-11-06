# -*- coding: utf-8 -*-
import os

from ..base import BaseModel
from foolbox.models import CaffeModel

class Caffe(BaseModel):
    FRAMEWORK = "caffe"
    def __init__(self, proto, weights, caffe_path=None, log_level=2, use_cpu=False):
        # no configuration now
        super(Caffe, self).__init__()
        self.caffe_path = caffe_path
        self.use_cpu = use_cpu
        if caffe_path:
            import sys
            sys.path.insert(0, caffe_path)
        os.environ["GLOG_minloglevel"] = str(log_level)
        import caffe
        if use_cpu:
            caffe.set_mode_cpu()
        else:
            gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            caffe.set_mode_gpu()
            caffe.set_device(int(gpu))
            print("caffe use gpu: ", gpu)
            
        # Check if force_backward is set to true! important!
        from caffe.proto import caffe_pb2
        from google.protobuf import text_format
        net_proto = caffe_pb2.NetParameter()
        with open(proto, "r") as rf:
            text_format.Merge(rf.read(), net_proto)
        if not net_proto.force_backward:
            net_proto.force_backward = True
            proto = proto + "--automodified"
            print("Will add force_backward to this prototxt, will use the modified prototxt {}".format(proto))
            with open(proto, "w") as wf:
                wf.write(text_format.MessageToString(net_proto))
        self.proto = proto
        self.weights = weights
        self.net = caffe.Net(proto, weights, caffe.TEST)
        
    @classmethod
    def create_fmodel(cls, cfg):
        model = cls(**cfg["cfg"])
        bounds = tuple(cfg.get("bounds", [0, 1]))
        preprocessing = tuple(cfg.get("preprocessing", [0, 255]))
        fmodel = CaffeModel(model.net, bounds=bounds, channel_axis=1, preprocessing=preprocessing,
                            data_blob_name=cfg.get("data_blob_name", "data"),
                            label_blob_name=cfg.get("label_blob_name", "label"),
                            output_blob_name=cfg.get("output_blob_name", "output"))
        return fmodel
        
Model = Caffe
