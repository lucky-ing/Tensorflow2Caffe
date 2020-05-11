import Tensorflow2Caffe
import cv2
import numpy as np


class CaffeGenerator(object):
    def __init__(self, pb_file=None, meta_file=None, ckpt_file=None):
        if pb_file:
            file_path = "CaffeFile"
            net_name = pb_file.split('/')[-1].split('.')[0]
            self.model = Tensorflow2Caffe.Tensorflow2Caffe(pb=pb_file,
                                                           filepath=file_path, netname=net_name, ROTATION=False,
                                                           INPLACE=True,
                                                           RELU6=False)
        else:
            assert meta_file and ckpt_file, "pb file or mata,ckpt file must be supply one"
            file_path = "CaffeFile"
            net_name = meta_file.split('/')[-1].split('.')[0]
            self.model = Tensorflow2Caffe.Tensorflow2Caffe(meta=meta_file,
                                                           ckpt=ckpt_file,
                                                           filepath=file_path, netname=net_name, ROTATION=False,
                                                           INPLACE=True,
                                                           RELU6=False)
        self.input_data = None

    def set_input(self, input_shape=(300, 300), input_tensor_name="image_tensor:0"):

        # only for inferring shape
        input_data = cv2.imread('image.jpg')
        input_data = cv2.resize(input_data, input_shape)
        self.input_data = np.expand_dims(input_data, 0)
        input_temp = np.zeros((1, 300, 300, 3))
        input_temp[0, :, :, :] = self.input_data[0, :, :, :]
        self.model.set_placehold(input_tensor_name, input_temp)

    def generate_prototxt(self):
        # generate prototxt at first time run, to check if some operator is wrong, especially reshape.
        self.model.generate_prototxt(True)

    def generate_caffemodel(self):
        # after check the right prototxt, the caffe model can dump right now.
        self.model.generate_prototxt(False)
        self.model.generate_caffemodel()

    def vector_diff(self,
                    caffe_input_name='import/image',
                    caffe_layer_name='import/MobilenetV1/Conv2d_7_pointwise/Relutsfw',
                    caffe_output_name='import/MobilenetV1/Conv2d_7_pointwise/Conv2D',
                    tensorflow_tensor_name='import/MobilenetV1/Conv2d_7_pointwise/Relu:0'):
        if not self.input_data:
            print("please run set_input first")
            return

        # we also provide a vector diff tool to compare the out data of tensorflow and caffe.
        image_data = np.swapaxes(self.input_data[0], 0, 2)
        image_data = np.swapaxes(image_data, 1, 2)
        self.model.set_caffe_input(caffe_input_name, image_data)
        self.model.check_output(tensorflow_tensor_name,
                                caffe_layer_name,
                                caffe_output_name)


if __name__ == "__main__":
    generator = CaffeGenerator(pb_file="test.pb")
    # or
    # generator = CaffeGenerator(meta_file="test.meta", ckpt_file="test.ckpt")

    # set input is helpful to infer shape, it is not must
    generator.set_input(input_shape=(300, 300), input_tensor_name="image_tensor:0")

    # please run generate_prototxt first, and after this, check the prototxt file
    generator.generate_prototxt()

    # you can set assert 0 here to wait for right prototxt.
    generator.generate_caffemodel()
