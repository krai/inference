# Only support SDK Version up to 21.06

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import os
import backend
import shutil


class BackendTensorflowRT(backend.Backend):
    def __init__(self):
        super(BackendTensorflowRT, self).__init__()

    def set_extra_params (self, params):
        self.params = params

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tensorflowRT"

    def image_format(self):
        # By default tensorflow uses NHWC (and the cpu implementation only does NHWC)
        return "NHWC"


    def load(self, model_path, inputs=None, outputs=None):
        # there is no input/output meta data i the graph so it need to come from config.
        if not inputs:
            raise ValueError("BackendTensorflowRT needs inputs")
        if not outputs:
            raise ValueError("BackendTensorflowRT needs outputs")
        self.outputs = outputs
        self.inputs = inputs

        infer_config = tf.compat.v1.ConfigProto()
        infer_config.intra_op_parallelism_threads = int(os.environ['TF_INTRA_OP_PARALLELISM_THREADS']) \
                if 'TF_INTRA_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
        infer_config.inter_op_parallelism_threads = int(os.environ['TF_INTER_OP_PARALLELISM_THREADS']) \
                if 'TF_INTER_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
        infer_config.use_per_session_threads = 1
        
        # Convert model path to the following format: "$HOME/CK_TOOLS/$MODEL/saved_model/"
        # Save trt model to the following location: "$HOME/CK_TOOLS/$MODEL/saved_model_trt/"
        model_path = model_path[:-14]
        trt_model_path = model_path[:-1]+"_trt/"
        if not os.path.exists(trt_model_path):
            os.makedirs(trt_model_path)
        else:
            shutil.rmtree(trt_model_path)
            os.makedirs(trt_model_path)

        converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_path)
        converter.convert()
        converter.save(trt_model_path)

        # Support TF2 Saved Model format
        self.model = tf.saved_model.load(trt_model_path)
        self.model.signatures['serving_default'].output_dtypes
        self.model.signatures['serving_default'].output_shapes
        return self


    def predict(self, feed):
        key = list(feed)[0]
        input_tensor = tf.convert_to_tensor(feed[key])
        output = self.model(input_tensor)
        predictions = [output[key] for key in self.outputs]
        return predictions