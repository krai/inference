"""
tensorflow backend (https://github.com/tensorflow/tensorflow)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

import tensorflow as tf
from tensorflow import dtypes
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

import os
import backend


class BackendTensorflow(backend.Backend):
    def __init__(self, optimize_graph=None):
        self.optimize_graph=optimize_graph
        super(BackendTensorflow, self).__init__()

    def version(self):
        return tf.__version__ + "/" + tf.__git_version__

    def name(self):
        return "tensorflow"

    def image_format(self):
        # By default tensorflow uses NHWC (and the cpu implementation only does NHWC)
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        # there is no input/output meta data i the graph so it need to come from config.
        if not inputs:
            raise ValueError("BackendTensorflow needs inputs")
        if not outputs:
            raise ValueError("BackendTensorflow needs outputs")
        self.outputs = outputs
        self.inputs = inputs

        infer_config = tf.compat.v1.ConfigProto()
        infer_config.intra_op_parallelism_threads = int(os.environ['TF_INTRA_OP_PARALLELISM_THREADS']) \
                if 'TF_INTRA_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
        infer_config.inter_op_parallelism_threads = int(os.environ['TF_INTER_OP_PARALLELISM_THREADS']) \
                if 'TF_INTER_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
        infer_config.use_per_session_threads = 1

        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.FastGFile(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        if self.optimize_graph:
            try:
                optimized_graph_def = optimize_for_inference(graph_def, [item.split(':')[0] for item in inputs],
                        [item.split(':')[0] for item in outputs], dtypes.float32.as_datatype_enum, False)
                g = tf.compat.v1.import_graph_def(optimized_graph_def, name='')
            except ValueError:
                try:
                    optimized_graph_def = optimize_for_inference(graph_def, [item.split(':')[0] for item in inputs],
                            [item.split(':')[0] for item in outputs], dtypes.uint8.as_datatype_enum, False)
                    g = tf.compat.v1.import_graph_def(optimized_graph_def, name='')
                except ValueError:
                    g = tf.compat.v1.import_graph_def(graph_def, name='')
        else:
            g = tf.compat.v1.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=g, config=infer_config)
        print("self.outputs", self.outputs)

        return self

    def predict(self, feed):
        return self.sess.run(self.outputs, feed_dict=feed)



class BackendTensorflow2(BackendTensorflow):
    def load(self, model_path, inputs=None, outputs=None):
        # there is no input/output meta data i the graph so it need to come from config.
        if not inputs:
            raise ValueError("BackendTensorflow needs inputs")
        if not outputs:
            raise ValueError("BackendTensorflow needs outputs")
        self.outputs = outputs
        self.inputs = inputs

        infer_config = tf.compat.v1.ConfigProto()
        infer_config.intra_op_parallelism_threads = int(os.environ['TF_INTRA_OP_PARALLELISM_THREADS']) \
                if 'TF_INTRA_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
        infer_config.inter_op_parallelism_threads = int(os.environ['TF_INTER_OP_PARALLELISM_THREADS']) \
                if 'TF_INTER_OP_PARALLELISM_THREADS' in os.environ else os.cpu_count()
        infer_config.use_per_session_threads = 1
        

        # Convert model path to the following format: "$HOME/CK_TOOLS/$MODEL/saved_model/"
        model_path = model_path[:-14]
        # Support TF2 Saved Model format
        self.model = tf.saved_model.load(model_path)
        self.model.signatures['serving_default'].output_dtypes
        self.model.signatures['serving_default'].output_shapes
        return self

    def predict(self, feed):
        key = list(feed)[0]
        input_tensor = tf.convert_to_tensor(feed[key])
        output = self.model(input_tensor)
        predictions = [output[key] for key in self.outputs]
        return predictions