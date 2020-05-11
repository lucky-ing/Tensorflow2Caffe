import tensorflow as tf
import writecaffe
import caffe
import math
import numpy as np
import os

TAG = 'tsfw'
import cv2


# meta='alexnet/model/alexnet_10000.ckpt.meta'
# ckpt='alexnet/model/alexnet_10000.ckpt'
# meta='backup/model-20170512-110547.meta'
# ckpt='backup/model-20170512-110547.ckpt-250000.data-00000-of-00001'
# meta='lenetmodel/lenet_40000.ckpt.meta'
# ckpt='lenetmodel/lenet_40000.ckpt'

# meta='testmodel/cov/cov_model.meta'
# ckpt='testmodel/cov/cov_model'
# meta='/home/lucky/tensorflow/squeezeDet/data/model_checkpoints/squeezeDetPlus/model.ckpt-95000.meta'
# ckpt='/home/lucky/tensorflow/squeezeDet/data/model_checkpoints/squeezeDetPlus/model.ckpt-95000'
# meta='/home/lucky/tensorflow/squeezeDet/data/model_checkpoints/vgg16/model.ckpt-101500.meta'
# ckpt='/home/lucky/tensorflow/squeezeDet/data/model_checkpoints/vgg16/model.ckpt-101500'
# meta='/home/lucky/PycharmProjects/tensorflow2caffe/ssd_mobilenet_v1_coco_2017_11_17/model.ckpt.meta'
# ckpt='/home/lucky/PycharmProjects/tensorflow2caffe/ssd_mobilenet_v1_coco_2017_11_17/model.ckpt'
# meta='mtcnn_tf/Pnet/PNet-30.meta'
# ckpt='mtcnn_tf/Pnet/PNet-30'
# meta='mtcnn_tf/Onet/ONet-22.meta'
# ckpt='mtcnn_tf/Onet/ONet-22'
# meta='backup/sepatate/pnet/pnet-3000000.meta'
# ckpt='backup/sepatate/pnet/pnet-3000000'
# pb='openpose/mobilenet_thin/graph_freeze.pb'
# meta='model-ssd-openpose/model-388003.meta'
# ckpt='model-ssd-openpose/model-388003'

class Tensorflow2Caffe(object):
    def __init__(self, meta=None, ckpt=None, pb=None, ROTATION=False, filepath='./', netname='test', INPLACE=False,
                 RELU6=True):
        self.filepath = filepath
        self.netname = netname
        self.ROTATION = ROTATION
        self.INPLACE = INPLACE
        self.WRITE = False
        self.IMAGE = False
        self.RELU6 = RELU6
        self.first = True
        self.tensorname = ''
        self.tensordata = None
        self.operations_dict = []
        self.layers = []
        self.pointnames = []
        self.toplist = []
        self.weight2operation = {}
        self.ConstWeight_dict = {}
        self.feed_dict = {}
        self.reshape_operations = []
        self.reshape_boxs = {}
        self.caffe_input_data = None
        self.caffe_input_name = ''
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        if meta and ckpt:
            new_saver = tf.train.import_meta_graph(meta)
            self.sess.run(tf.initialize_all_variables())
            new_saver.restore(self.sess, ckpt)
        else:
            if pb:
                output_graph_def = tf.GraphDef()
                with open(pb, 'rb') as ff:
                    output_graph_def.ParseFromString(ff.read())
                    _ = tf.import_graph_def(output_graph_def)
            else:
                print('must hava a input ,ckpt or pb file need!')
                assert 1

    def add_concat_layer(self, input_name, out_name, shape, axis):
        keys = {}
        layer = writecaffe.InputLayer()
        input_name_input = input_name + '_input'
        layer.name = input_name_input + TAG
        inputshape = shape[:]
        inputshape[axis] = 1
        print(inputshape)
        layer.top.append(self.get_realname(layer.name))
        self.toplist.append(self.get_realname(layer.name))

        if len(inputshape) == 4:
            inputshape[0] = 1

        keys['shape'] = inputshape
        layer.input_param.append(keys)
        self.layers.append(layer)

        keys = {}
        layer = writecaffe.ConcatLayer()
        layer.name = input_name + '_concat' + TAG

        layer.top.append(self.get_realname(out_name))
        self.toplist.append(self.get_realname(out_name))

        layer.bottom.append(self.get_realname(input_name))
        layer.bottom.append(self.get_realname(input_name_input))
        keys['axis'] = axis
        layer.concat_param.append(keys)
        self.layers.append(layer)

    def get_operation_input(self, input_name):
        for i in self.operations_dict:
            for j in i[1]:
                if input_name == j.name:
                    return i[0]
        return None

    def get_operation_output(self, output_name):
        for i in self.operations_dict:
            for j in i[2]:
                if output_name == j.name:
                    return i[0]
        return None

    def get_realname(self, name):
        name = name.split(':')[0]
        if not self.pointnames:
            temp = []
            temp.append(name)

            self.pointnames.append(temp)
            return name
        for i in self.pointnames:
            if name in i:
                return i[0]
        temp = []
        temp.append(name)
        # if 'read' not in name:
        self.pointnames.append(temp)
        return name

    def add_realname(self, name, name0):
        if 'read' in name:
            return
        name = name.split(':')[0]
        name0 = name0.split(':')[0]
        if not self.pointnames:
            temp = []
            temp.append(name)
            temp.append(name0)
            self.pointnames.append(temp)
        for i in range(len(self.pointnames)):
            if name in self.pointnames[i]:
                self.pointnames[i].append(name0)
        temp = []
        temp.append(name)
        temp.append(name0)
        self.pointnames.append(temp)

    def get_transform(self, name):
        name = name.split(':')[0] + '/read'
        for (i, j) in self.reshape_operations:
            for i_temp in i:
                if name in i_temp.name.decode():
                    if name in i[0].name.decode():
                        if i[1].name.decode() in self.reshape_boxs.keys():
                            (shape0, shape1) = self.reshape_boxs[i[1].name.decode()]
                            return shape0.as_list()
                    else:
                        if i[0].name.decode() in self.reshape_boxs.keys():
                            (shape0, shape1) = self.reshape_boxs[i[0].name.decode()]
                            return shape0.as_list()
        return None

    def set_placehold(self, tensorname, tensordata):
        self.IMAGE = True
        self.tensorname = tensorname
        self.tensordata = tensordata
        image_tensor = self.sess.graph.get_tensor_by_name(self.tensorname)
        self.feed_dict[image_tensor] = self.tensordata

    def get_placehold(self, tensorname):
        image_tensor = self.sess.graph.get_tensor_by_name(tensorname)
        v_data = self.sess.run(image_tensor, feed_dict=self.feed_dict)
        return v_data

    def generate_prototxt(self, write=False):
        if write == True:
            ff = open(os.path.join(self.filepath, self.netname + ".prototxt"), 'w')
            ff.write("name:" + "\"" + self.netname + "\"" + "\n")
            ff.close()
        Batch_normal_dict = ['Add', 'Rsqrt', 'Mul', 'Mul', 'Mul', 'Sub', 'Add']
        Prelu_dict = ['Relu', 'Abs', 'Sub', 'Mul', 'Mul', 'Add']
        Batch_normal_index = 0
        Prelu_index = 0
        prelu_alphas_name = ''
        belta_name = ''
        gamma_name = ''
        moving_variance = ''
        moving_mean = ''
        esp_operation_name = ''
        eps = 0

        writer = tf.summary.FileWriter("./logs", tf.get_default_graph())
        writer.close()
        graph = tf.get_default_graph()
        # for t in tf.trainable_variables():
        #    print(t)
        # for t in tf.global_variables():
        #    print(t)
        for i in graph.get_operations():
            print(i.type)
            input_temp = []
            output_temp = []
            for j in i.inputs:
                print('input', j)
                input_temp.append(j)
            for j in i.outputs:
                print('out', j)
                output_temp.append(j)

            self.operations_dict.append([i, input_temp, output_temp])
            if 'gradient' in i.name:
                continue
            if 'Shape' == i.type:
                continue
            if 'MultiClassNonMaxSuppression' in i.name:
                continue
            if 'MultipleGridAnchorGenerator' in i.name:
                continue
            if 'Const' == i.type:
                continue
            if i.type == 'Placeholder':
                keys = {}
                layer = writecaffe.InputLayer()
                layer.name = i.name + TAG
                shape_ = i.get_attr('shape')
                if not len(shape_.dim):
                    continue
                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    self.toplist.append(self.get_realname(j.name))
                for j in i.inputs:
                    layer.bottom.append(self.get_realname(j.name))
                inputshape = []
                for j in shape_.dim:
                    inputshape.append(j.size)
                if len(inputshape) == 4:
                    inputshape[0] = 1
                    temp = inputshape[3]
                    inputshape[3] = inputshape[2]
                    inputshape[2] = inputshape[1]
                    inputshape[1] = temp
                for i in range(len(inputshape)):
                    if inputshape[i] <= 0:
                        inputshape[i] = 1
                keys['shape'] = inputshape
                layer.input_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'MatMul':
                keys = {}
                layer = writecaffe.FCNLayer()
                layer.name = i.name + TAG

                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    self.toplist.append(self.get_realname(j.name))
                for j in i.inputs:
                    if self.get_realname(j.name) not in self.toplist:
                        name_temp = j.name.encode()
                        name_temp = str.split(name_temp, ':')[0]
                        name_temp = name_temp.replace('/read', '')
                        self.weight2operation[name_temp] = [i.name.encode() + TAG, 0, 0]
                        continue
                    layer.bottom.append(self.get_realname(j.name))
                out_num = 0
                for j in i.outputs:
                    if not out_num == 0 and not out_num == j.shape[-1]:
                        assert 0, 'conv output num is not equael'
                    else:
                        out_num = j.shape[-1]
                layer.num_output = out_num
                keys['num_output'] = layer.num_output
                keys['bias_term'] = 'false'
                layer.inner_product_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'BiasAdd':
                bias_name = ''
                net_name = ''
                for j in i.inputs:
                    if 'read' in j.name:
                        bias_name = str(j.name).split(':')[0]
                    else:
                        net_name = str(j.name).split(':')[0]
                bias_name = bias_name.replace('/read', '')
                if net_name not in self.layers[-1].name:
                    assert ('NOT FIND')
                self.weight2operation[bias_name] = [self.layers[-1].name, 1, 0]
                if len(self.layers[-1].inner_product_param):
                    self.layers[-1].inner_product_param[0]['bias_term'] = 'true'
                if len(self.layers[-1].convolution_param):
                    self.layers[-1].convolution_param[0]['bias_term'] = 'true'
            if i.type == 'Add':
                keys = {}
                layer = writecaffe.AddLayer()
                layer.name = i.name + TAG
                B = True
                for j in i.inputs:
                    if self.get_realname(j.name) not in self.toplist:
                        B = False
                if B == True:
                    for j in i.inputs:
                        layer.bottom.append(self.get_realname(j.name))
                    for j in i.outputs:
                        layer.top.append(self.get_realname(j.name))
                        self.toplist.append(self.get_realname(j.name))
                    operation = 'SUM'
                    keys['operation'] = operation
                    layer.add_param.append(keys)
                    self.layers.append(layer)
                    continue
            if i.type == 'Conv2D':
                keys = {}
                layer = writecaffe.COV2DLayer()
                layer.name = i.name + TAG
                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    self.toplist.append(self.get_realname(j.name))
                shape_temp = [1, 1, 1, 1]
                kernel_shape_temp = [1, 1, 1, 1]
                caffe_shape_temp = [1, 1, 1, 1]
                caffe_kernel_shape_temp = [1, 1, 1, 1]
                for j in i.inputs:
                    # print('111222',j)
                    if '/read' not in j.name:
                        if self.IMAGE == True:
                            j_temp = self.sess.run(j, feed_dict=self.feed_dict)
                            shape_temp = np.shape(j_temp)
                        else:
                            shape_temp = [int(m) for m in j.shape]
                    else:
                        kernel_shape_temp = [int(m) for m in j.shape]

                    # if self.get_realname(j.name) not in self.toplist:
                    if '/read:0' in j.name:
                        name_temp = j.name.encode()
                        name_temp = str.split(name_temp, ':')[0]
                        name_temp = name_temp.replace('/read', '')
                        self.weight2operation[name_temp] = [i.name.encode() + TAG, 0, 0]
                        continue
                    layer.bottom.append(self.get_realname(j.name))
                caffe_shape_temp = [shape_temp[0], shape_temp[3], shape_temp[1], shape_temp[2]]
                stride = [float(m) for m in i.get_attr('strides')]
                layer.stride = stride[2]
                stride_h = stride[1]
                stride_w = stride[2]
                padding = i.get_attr('padding')
                if padding == 'SAME':
                    if stride_h == stride_w and stride_h == 1:
                        pad_2 = kernel_shape_temp[0] - 1
                        pad_1 = kernel_shape_temp[1] - 1
                    else:

                        pad_1 = int(math.ceil(shape_temp[1] / stride_h) - math.floor(
                            ((shape_temp[1]) - (kernel_shape_temp[0]) + 1) / stride_h) + 0.999)
                        pad_2 = int(math.ceil(shape_temp[2] / stride_w) - math.floor(
                            ((shape_temp[2]) - (kernel_shape_temp[1]) + 1) / stride_w) + 0.999)

                    if pad_1 != pad_2:
                        keys['pad_h'] = int((pad_1 + 1) / 2)
                        keys['pad_w'] = int((pad_2 + 1) / 2)
                    else:
                        layer.pad = pad_1 + 1
                        keys['pad'] = int(layer.pad / 2)
                input_len = len(i.inputs)
                if (input_len == 2):
                    layer.kernel_size = i.inputs[1].shape[0]
                out_num = 0
                for j in i.outputs:
                    if not out_num == 0 and not out_num == j.shape[-1]:
                        assert 0, 'conv output num is not equael'
                    else:
                        out_num = j.shape[-1]

                layer.num_output = out_num
                keys['num_output'] = layer.num_output
                keys['kernel_size'] = layer.kernel_size
                keys['stride'] = int(layer.stride + 0.01)
                keys['bias_term'] = 'false'
                layer.convolution_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'MaxPool':
                keys = {}
                layer = writecaffe.PoolLayer()
                layer.name = i.name + TAG
                shape_temp = [1, 1, 1, 1]
                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    self.toplist.append(self.get_realname(j.name))

                for j in i.inputs:
                    if self.get_realname(j.name) not in self.toplist:
                        continue
                    if self.IMAGE == True:
                        j_temp = self.sess.run(j, feed_dict=self.feed_dict)
                        shape_temp = np.shape(j_temp)
                    else:
                        shape_temp = [int(m) for m in j.shape]
                    layer.bottom.append(self.get_realname(j.name))

                layer.type = 'Pooling'
                layer.pool = 'MAX'
                layer.kernel_size = i.get_attr('ksize')[2]
                layer.stride = i.get_attr('strides')[2]
                keys['kernel_size'] = layer.kernel_size
                keys['stride'] = layer.stride
                keys['pool'] = layer.pool
                layer.pooling_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'Relu6' and self.RELU6:
                layer = writecaffe.ScaleLayer()
                layer.name = i.name.encode() + '_scale0' + TAG
                keys = {}
                shape_temp = [1, 1, 1, 1]
                for j in i.outputs:
                    layer.top.append(self.get_realname('scale0_' + j.name))
                    if self.IMAGE == True:
                        shape_temp = self.sess.run(tf.shape(j), feed_dict=self.feed_dict)
                    else:
                        shape_temp = [int(m) for m in j.shape]
                    print(shape_temp)
                    print(self.get_realname('scale0_' + j.name))
                    self.toplist.append(self.get_realname('scale0_' + j.name))
                for j in i.inputs:
                    if self.get_realname(j.name.encode()) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname(j.name.encode()))
                cost_temp = -np.ones((shape_temp[3]))
                self.ConstWeight_dict[i.name.encode() + '_scale0'] = [layer.name, 0, cost_temp]
                keys['bias_term'] = 'false'
                layer.scale_param.append(keys)
                self.layers.append(layer)

                layer = writecaffe.BaisLayer()
                layer.name = i.name.encode() + '_bais0' + TAG
                keys = {}
                for j in i.outputs:
                    layer.top.append(self.get_realname('bais0_' + j.name))

                    self.toplist.append(self.get_realname('bais0_' + j.name))
                for j in i.outputs:
                    if self.get_realname('scale0_' + j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname('scale0_' + j.name))
                cost_temp = 6.0 * np.ones((shape_temp[3]))
                self.ConstWeight_dict[i.name.encode() + '_bais0'] = [layer.name, 0, cost_temp]
                # keys['bias_term'] = 'false'
                # layer.scale_param.append(keys)
                self.layers.append(layer)

                keys = {}
                layer = writecaffe.ReLULayer()
                layer.name = i.name.encode() + '_relu0' + TAG
                for j in i.outputs:
                    layer.top.append(self.get_realname('relu0_' + j.name))
                    self.toplist.append(self.get_realname('relu0_' + j.name))

                for j in i.outputs:
                    if self.get_realname('bais0_' + j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname('bais0_' + j.name))
                # layer.relu_param.append(keys)
                self.layers.append(layer)

                layer = writecaffe.ScaleLayer()
                layer.name = i.name.encode() + '_scale1' + TAG
                keys = {}

                for j in i.outputs:
                    layer.top.append(self.get_realname('scale1_' + j.name))
                    self.toplist.append(self.get_realname('scale1_' + j.name))
                for j in i.outputs:
                    if self.get_realname('relu0_' + j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname('relu0_' + j.name))
                cost_temp = -np.ones((shape_temp[3]))
                self.ConstWeight_dict[i.name.encode() + '_scale1'] = [layer.name, 0, cost_temp]
                keys['bias_term'] = 'false'
                layer.scale_param.append(keys)
                self.layers.append(layer)

                layer = writecaffe.BaisLayer()
                layer.name = i.name.encode() + '_bais1'
                keys = {}
                for j in i.outputs:
                    layer.top.append(self.get_realname('bais1_' + j.name))

                    self.toplist.append(self.get_realname('bais1_' + j.name))
                for j in i.outputs:
                    if self.get_realname('scale1_' + j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname('scale1_' + j.name))
                cost_temp = 6.0 * np.ones((shape_temp[3]))
                self.ConstWeight_dict[i.name.encode() + '_bais1'] = [layer.name, 0, cost_temp]
                self.layers.append(layer)

                keys = {}
                layer = writecaffe.ReLULayer()
                layer.name = i.name.encode() + '_relu1' + TAG
                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    self.toplist.append(self.get_realname(j.name))
                for j in i.outputs:
                    if self.get_realname('bais1_' + j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname('bais1_' + j.name))
                self.layers.append(layer)

                continue
            if i.type == 'Relu' or i.type == 'Relu6':
                if i.type == Prelu_dict[Prelu_index]:
                    Prelu_index += 1
                keys = {}
                layer = writecaffe.ReLULayer()
                layer.name = i.name + TAG
                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    # print(j.shape[0])
                    self.toplist.append(self.get_realname(j.name))
                    # print(j)
                    # print(j.name)
                for j in i.inputs:
                    if self.get_realname(j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname(j.name))
                # layer.relu_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'Softmax':
                keys = {}
                layer = writecaffe.SoftmaxLayer()
                layer.name = i.name + TAG
                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    # print(j.shape[0])
                    self.toplist.append(self.get_realname(j.name))
                    # print(j)
                    # print(j.name)
                for j in i.inputs:
                    if self.get_realname(j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname(j.name))
                # layer.relu_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'DepthwiseConv2dNative':
                keys = {}
                layer = writecaffe.COV2DLayer()
                layer.name = i.name + TAG
                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    # print(j.shape[0])
                    self.toplist.append(self.get_realname(j.name))
                    # print(j)
                    # print(j.name)
                shape_temp = [1, 1, 1, 1]
                kernel_shape_temp = [1, 1, 1, 1]
                caffe_shape_temp = [1, 1, 1, 1]
                caffe_kernel_shape_temp = [1, 1, 1, 1]
                for j in i.inputs:
                    print('lucccc', j)
                    if '/read' not in j.name:
                        if self.IMAGE == True:
                            shape_temp = self.sess.run(tf.shape(j), feed_dict=self.feed_dict)
                        else:
                            shape_temp = [int(m) for m in j.shape]
                    else:
                        kernel_shape_temp = [int(m) for m in j.shape]

                    # if self.get_realname(j.name) not in self.toplist:
                    if '/read:0' in j.name:
                        name_temp = j.name.encode()
                        name_temp = str.split(name_temp, ':')[0]
                        name_temp = name_temp.replace('/read', '')
                        self.weight2operation[name_temp] = [i.name.encode() + TAG, 0, 1]
                        continue
                    layer.bottom.append(self.get_realname(j.name))
                caffe_shape_temp = [shape_temp[0], shape_temp[3], shape_temp[1], shape_temp[2]]
                stride = [float(m) for m in i.get_attr('strides')]
                # print(stride)
                layer.stride = stride[2]
                stride_h = stride[1]
                stride_w = stride[2]
                padding = i.get_attr('padding')
                if padding == 'SAME':
                    if stride_h == stride_w and stride_h == 1:
                        pad_2 = kernel_shape_temp[0] - 1
                        pad_1 = kernel_shape_temp[1] - 1
                    else:
                        # print(shape_temp, kernel_shape_temp, stride)
                        # print(shape_temp[1] / stride_h)
                        # print(math.ceil(shape_temp[1] / stride_h),
                        #      (math.ceil(((shape_temp[1]) - (kernel_shape_temp[0]) + 1) / stride_h)),
                        #      math.ceil(shape_temp[1] / stride_h) - (
                        #          math.ceil((shape_temp[1]) - (kernel_shape_temp[0]) + 1) / stride_h))
                        pad_1 = int(math.ceil(shape_temp[1] / stride_h) - math.floor(
                            ((shape_temp[1]) - (kernel_shape_temp[0]) + 1) / stride_h) + 0.999)
                        pad_2 = int(math.ceil(shape_temp[2] / stride_w) - math.floor(
                            ((shape_temp[2]) - (kernel_shape_temp[1]) + 1) / stride_w) + 0.999)
                    if pad_1 != pad_2:
                        keys['pad_h'] = int((pad_1 + 1) / 2)
                        keys['pad_w'] = int((pad_2 + 1) / 2)
                        print(
                            'connot set padding ,check weather some thing is wrong or you can turn off this assertment')
                    else:
                        layer.pad = pad_1 + 1
                        keys['pad'] = int(layer.pad / 2)

                input_len = len(i.inputs)
                if (input_len == 2):
                    layer.kernel_size = i.inputs[1].shape[0]
                out_num = 0
                for j in i.outputs:
                    if not out_num == 0 and not out_num == j.shape[-1]:
                        assert 0, 'conv output num is not equael'
                    else:
                        out_num = j.shape[-1]

                layer.num_output = out_num
                keys['num_output'] = layer.num_output
                keys['kernel_size'] = layer.kernel_size
                keys['stride'] = int(layer.stride + 0.01)
                keys['group'] = layer.num_output
                keys['bias_term'] = 'false'
                layer.convolution_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'Concat':
                # print(i)
                # assert 0
                keys = {}
                layer = writecaffe.ConcatLayer()
                layer.name = i.name + TAG
                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    # print(j.shape[0])
                    self.toplist.append(self.get_realname(j.name))
                    # print(j)
                    # print(j.name)
                for j in i.inputs:
                    if self.get_realname(j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname(j.name))
                # layer.relu_param.append(keys)
                keys['axis'] = layer.axis
                layer.concat_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'ConcatV2':
                # print(i)
                # assert 0
                keys = {}
                layer = writecaffe.ConcatLayer()
                layer.name = i.name + TAG
                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    # print(j.shape[0])
                    self.toplist.append(self.get_realname(j.name))
                    # print(j)
                    # print(j.name)
                for j in i.inputs:
                    if self.get_realname(j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname(j.name))
                # layer.relu_param.append(keys)
                if self.IMAGE == True:
                    axis = self.sess.run(i.inputs[-1], feed_dict=self.feed_dict)
                    print(axis)
                    if axis == 0:
                        layer.axis = 1
                keys['axis'] = layer.axis
                layer.concat_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'FusedBatchNorm':

                Batch_normal_index = 0
                keys = {}
                layer = writecaffe.BatchnormalLayer()
                layer.name = i.name + TAG

                layer.top.append(self.get_realname('bn_' + i.outputs[0].name.encode()))
                self.toplist.append(self.get_realname('bn_' + i.outputs[0].name.encode()))

                gamma_name = i.inputs[1].name
                belta_name = i.inputs[2].name
                moving_mean = i.inputs[3].name
                moving_variance = i.inputs[4].name

                for j in i.inputs:
                    if self.get_realname(j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname(j.name))
                # layer.relu_param.append(keys)
                try:
                    mean_ = graph.get_tensor_by_name(moving_mean)
                    variance_ = graph.get_tensor_by_name(moving_variance)
                    mean_ = self.sess.run(mean_)
                    variance_ = self.sess.run(variance_)
                    if (np.shape(mean_)[0] == 0 or np.shape(variance_)[0] == 0):
                        self.ConstWeight_dict[i.name.encode() + '_BLOB2'] = [i.name.encode() + TAG, 2, np.array([1.0])]
                        assert 0
                    name_temp = moving_mean.replace('/read:0', '')
                    name_temp = name_temp.replace(':0', '')
                    self.weight2operation[name_temp] = [i.name.encode() + TAG, 0, 0]
                    name_temp = moving_variance.replace('/read:0', '')
                    name_temp = name_temp.replace(':0', '')
                    self.weight2operation[name_temp] = [i.name.encode() + TAG, 1, 0]
                    # self.weight2operation[name_temp]=[i.name.encode()+'_BLOB2' , 2, 2]
                    self.ConstWeight_dict[i.name.encode() + '_BLOB2'] = [i.name.encode() + TAG, 2, np.array([1.0])]
                    keys['use_global_stats'] = 'true'
                    layer.batchnormal_param.append(keys)
                except BaseException:
                    print('using moment parameter')

                eps_float = i.get_attr('epsilon')
                eps = float(eps_float)

                if eps != 0:
                    keys['eps'] = str(eps)

                self.layers.append(layer)

                # scale
                layer = writecaffe.ScaleLayer()
                layer.name = i.name.encode() + '_scale' + TAG
                keys = {}
                layer.top.append(self.get_realname(i.outputs[0].name))
                self.toplist.append(self.get_realname(i.outputs[0].name))

                layer.bottom.append(self.get_realname('bn_' + i.outputs[0].name.encode()))

                name_temp = gamma_name.replace('/read:0', '')
                name_temp = name_temp.replace(':0', '')
                self.weight2operation[name_temp] = [i.name.encode() + '_scale' + TAG, 0, 0]
                name_temp = belta_name.replace('/read:0', '')
                name_temp = name_temp.replace(':0', '')
                self.weight2operation[name_temp] = [i.name.encode() + '_scale' + TAG, 1, 0]

                keys['bias_term'] = 'true'
                layer.scale_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'LRN':
                # print(i)
                keys = {}
                layer = writecaffe.LRNLayer()
                layer.name = i.name + TAG
                for j in i.outputs:
                    layer.top.append(self.get_realname(j.name))
                    # print(j.shape[0])
                    self.toplist.append(self.get_realname(j.name))
                    # print(j)
                    # print(j.name)
                for j in i.inputs:
                    # print(j.name)
                    if self.get_realname(j.name) not in self.toplist:
                        continue
                    layer.bottom.append(self.get_realname(j.name))
                layer.type = 'LRN'
                layer.local_size = i.get_attr('depth_radius')
                layer.alpha = i.get_attr('alpha')
                layer.beta = i.get_attr('beta')
                keys['local_size'] = layer.local_size
                keys['alpha'] = layer.alpha
                keys['beta'] = layer.beta
                layer.lrn_param.append(keys)
                self.layers.append(layer)
                continue
            if i.type == 'Reshape':
                keys = {}
                layer = writecaffe.ReshapeLayer()
                layer.name = i.name + TAG
                shape = []
                shape_flag = 0
                for j in i.outputs:
                    shape = j.shape
                    layer.top.append(self.get_realname(j.name))
                    self.toplist.append(self.get_realname(j.name))
                    # print(j.name)
                for j in i.inputs:
                    if 'shape' in j.name:
                        shape_flag = 1
                    if self.get_realname(j.name) not in self.toplist:
                        continue
                    # print(j.name,self.toplist)
                    layer.bottom.append(self.get_realname(j.name))
                if len(layer.bottom) != 0 and shape_flag == 1:
                    # print (shape)
                    if len(shape) == 4:
                        layer.shape = [0] * len(shape)
                        layer.shape[0] = 1
                        layer.shape[1] = int(shape[3])
                        layer.shape[2] = int(shape[2])
                        layer.shape[3] = int(shape[1])

                    if len(shape) == 2:
                        layer.shape = [0] * len(shape)
                        layer.shape[0] = 1
                        layer.shape[1] = int(shape[1])

                    keys['shape'] = layer.shape
                    layer.reshape_param.append(keys)
                    self.layers.append(layer)
                    continue
            # print('pre',Prelu_index)
            if i.type == Prelu_dict[Prelu_index] and Prelu_index >= 1:
                Prelu_index += 1
                if Prelu_index == 4:
                    prelu_alphas_name = i.inputs[0].name
                if Prelu_index >= 6:
                    self.layers[-1].type = 'PReLU'
                    prelu_alphas_name = prelu_alphas_name.replace('/read:0', '')
                    print(self.layers[-1].name)
                    self.weight2operation[prelu_alphas_name] = [self.layers[-1].name + TAG, 0, 0]
                    for j in i.outputs:
                        print(self.get_realname(j.name))
                        self.layers[-1].top[-1] = self.get_realname(j.name)
                        # print(j.shape[0])
                        self.toplist[-1] = (self.get_realname(j.name))
                    Prelu_index = 0
                    continue
            else:
                Prelu_index = 0
                prelu_alphas_name = ''
            if i.type == Batch_normal_dict[Batch_normal_index]:
                Batch_normal_index += 1
                if Batch_normal_index == 1:
                    if len(i.inputs) == 2:
                        moving_variance = i.inputs[0].name
                        esp_operation_name = i.inputs[1].name

                if Batch_normal_index == 5:
                    if len(i.inputs) == 2:
                        moving_mean = i.inputs[0].name
                if Batch_normal_index == 3:
                    if len(i.inputs) == 2:
                        gamma_name = i.inputs[1].name
                if Batch_normal_index == 6:
                    if len(i.inputs) == 2:
                        belta_name = i.inputs[0].name
                if Batch_normal_index >= 7:
                    Batch_normal_index = 0
                    keys = {}
                    layer = writecaffe.BatchnormalLayer()
                    layer.name = i.name + TAG
                    for j in i.outputs:
                        layer.top.append(self.get_realname('bn_' + j.name.encode()))

                        # print(j.shape[0])
                        self.toplist.append(self.get_realname('bn_' + j.name.encode()))
                        # print(j)
                        # print(j.name)
                    for j in i.inputs:
                        if self.get_realname(j.name) not in self.toplist:
                            continue
                        layer.bottom.append(self.get_realname(j.name))
                    # layer.relu_param.append(keys)
                    try:
                        mean_ = graph.get_tensor_by_name(moving_mean)
                        variance_ = graph.get_tensor_by_name(moving_variance)
                        mean_ = self.sess.run(mean_)
                        variance_ = self.sess.run(variance_)
                        name_temp = moving_mean.replace('/read:0', '')
                        name_temp = name_temp.replace(':0', '')
                        self.weight2operation[name_temp] = [i.name.encode() + TAG, 0, 0]
                        name_temp = moving_variance.replace('/read:0', '')
                        name_temp = name_temp.replace(':0', '')
                        self.weight2operation[name_temp] = [i.name.encode() + TAG, 1, 0]
                        # self.weight2operation[name_temp]=[i.name.encode()+'_BLOB2' , 2, 2]
                        self.ConstWeight_dict[i.name.encode() + '_BLOB2'] = [i.name.encode() + TAG, 2, np.array([1.0])]
                        keys['use_global_stats'] = 'true'
                        layer.batchnormal_param.append(keys)
                    except BaseException:
                        print('using moment parameter')
                    eps_operation = self.get_operation_output(esp_operation_name)
                    eps = 0
                    if eps_operation:
                        if eps_operation.type == 'Const':
                            eps = float(eps_operation.get_attr('value').float_val[0])

                    if eps != 0:
                        keys['eps'] = str(eps)

                    print(layer.top)
                    self.layers.append(layer)

                    # scale
                    layer = writecaffe.ScaleLayer()
                    layer.name = i.name.encode() + '_scale' + TAG
                    keys = {}
                    for j in i.outputs:
                        layer.top.append(self.get_realname(j.name))
                        # print(j.shape[0])
                        self.toplist.append(self.get_realname(j.name))
                        # print(j)
                        # print(j.name)
                    for j in i.outputs:
                        if self.get_realname('bn_' + j.name.encode()) not in self.toplist:
                            continue
                        layer.bottom.append(self.get_realname('bn_' + j.name.encode()))

                    name_temp = gamma_name.replace('/read:0', '')
                    name_temp = name_temp.replace(':0', '')
                    self.weight2operation[name_temp] = [i.name.encode() + '_scale' + TAG, 0, 0]
                    name_temp = belta_name.replace('/read:0', '')
                    name_temp = name_temp.replace(':0', '')
                    self.weight2operation[name_temp] = [i.name.encode() + '_scale' + TAG, 1, 0]

                    keys['bias_term'] = 'true'
                    layer.scale_param.append(keys)
                    self.layers.append(layer)
                    continue
            else:
                Batch_normal_index = 0
                belta_name = ''
                gamma_name = ''
                moving_variance = ''
                moving_mean = ''
                esp_operation_name = ''
                eps = 0
            for j in i.inputs:
                for k in i.outputs:
                    if self.get_realname(j.name) not in self.toplist:
                        continue
                    self.add_realname(j.name, k.name)
            continue
        if write == True:
            inplace_dict = {}
            if self.INPLACE == True:
                for index in range(len(self.layers)):
                    for j in range(len(self.layers[index].bottom)):
                        if self.layers[index].bottom[j] in inplace_dict.keys():
                            self.layers[index].bottom[j] = inplace_dict[self.layers[index].bottom[j]]
                    if self.layers[index].type == 'Scale' or self.layers[index].type == 'ReLU' or self.layers[
                        index].type == 'BatchNorm':
                        inplace_dict[self.layers[index].top[0]] = self.layers[index].bottom[0]
                        self.layers[index].top[0] = self.layers[index].bottom[0]
            for layer in self.layers:
                writecaffe.write2file(layer, os.path.join(self.filepath, self.netname + ".prototxt"))
        self.first = False

    def generate_caffemodel(self):
        TRANSFORM4TO2 = 1
        TRANSFORM2TO4 = 2
        NORMAL = 0
        DEPWISE = 1
        CONST = 2
        caffe.set_mode_cpu()
        net = caffe.Net(os.path.join(self.filepath, self.netname + ".prototxt"), caffe.TEST)
        if 1:
            graph = tf.get_default_graph()
            for i in graph.get_operations():
                if 'Initializer' in i.name.decode():
                    continue
                if 'GradientDescent' in i.name.decode():
                    continue
                if 'save' in i.name.decode():
                    continue
                if 'gradients' in i.name.decode():
                    continue
                if 'L2Loss' in i.name.decode():
                    continue
                if 'Reshape' in i.type.decode():
                    self.reshape_boxs[i.outputs[0].name.decode()] = [i.inputs[0].shape, i.inputs[1].shape]
                self.reshape_operations.append([i.inputs, i.outputs])
            dim_old = 0
            trans_type = 0

            caffemodel_params = []
            for param in net.params:
                print(param)
                caffemodel_params.append(str(param))
            import copy
            caffemodel_params_temp = copy.deepcopy(caffemodel_params)
            for var in tf.trainable_variables():
                print(var.name.encode())
                var_name = str.split(var.name.encode(), ':')[0]
                caffe_var_name = self.weight2operation[var_name][0]
                if caffe_var_name in caffemodel_params_temp:
                    if caffe_var_name in caffemodel_params:
                        index = caffemodel_params.index(caffe_var_name)
                        caffemodel_params.remove(caffe_var_name)
                else:
                    print('caffemodel donot hava params ', caffe_var_name)
                    continue
                var_type = self.weight2operation[var_name][2]
                var_site = self.weight2operation[var_name][1]
                v_4d = np.array(self.sess.run(var))
                print('recover ' + var.name + ' to ' + caffe_var_name + ' ' + str(var_site))
                if NORMAL == var_type:
                    if v_4d.ndim == 4:
                        if dim_old != 4:
                            if dim_old == 0:
                                trans_type = 1
                                dim_old = 4
                            else:
                                print('error')
                                assert 0
                        else:
                            dim_old = 4
                        v_4d = np.swapaxes(v_4d, 0, 2)  # swap H, I
                        v_4d = np.swapaxes(v_4d, 1, 3)  # swap W, O
                        v_4d = np.swapaxes(v_4d, 0, 1)  # swap I, O
                        if self.ROTATION == True:
                            shape_v = np.shape(v_4d)
                            H = shape_v[2]
                            W = shape_v[3]
                            v_4d_temp = np.copy(v_4d)
                            for m in range(H):
                                for n in range(W):
                                    v_4d[:, :, m, n] = v_4d_temp[:, :, H - m - 1, W - n - 1]
                        net.params[caffe_var_name][var_site].data[:, :, :, :] = v_4d[:, :, :, :]
                    if v_4d.ndim == 2:
                        if dim_old != 2:
                            if dim_old == 0:
                                trans_type = 0
                                dim_old = 2
                            else:
                                TRANS = self.get_transform(var.name.decode())
                                if TRANS:
                                    if TRANS[1] * TRANS[2] * TRANS[3] == list(np.shape(v_4d))[0]:
                                        temp = np.swapaxes(v_4d, 0, 1)
                                        temp = np.reshape(temp, (list(np.shape(v_4d))[1], TRANS[1], TRANS[2], TRANS[3]))
                                        temp = np.swapaxes(temp, 1, 3)  # swap H, I
                                        temp = np.swapaxes(temp, 2, 3)  # swap W, O
                                        v_4d = np.reshape(temp, np.shape(np.swapaxes(v_4d, 0, 1)))
                                        # v_4d = np.swapaxes(temp, 0, 1)  # swap H, I
                                        dim_old = 2
                                    else:
                                        print('error')
                                        assert 0
                        else:
                            dim_old = 2
                            v_4d = np.swapaxes(v_4d, 0, 1)  # swap H, I
                            # v_4d_temp=np.asarray(v_4d)
                        if self.ROTATION == True:
                            shape_v = np.shape(v_4d)
                            print(np.shape(v_4d))
                            W = shape_v[1]
                            v_4d_temp = np.copy(v_4d)
                            print(np.shape(v_4d_temp))
                            for m in range(H):
                                v_4d[:, m] = v_4d_temp[:, H - m - 1]
                        net.params[caffe_var_name][var_site].data[:, :] = v_4d[:, :]
                    if v_4d.ndim == 1:
                        net.params[caffe_var_name][var_site].data[:] = v_4d[:]
                if DEPWISE == var_type:
                    if v_4d.ndim == 4:
                        if dim_old != 4:
                            if dim_old == 0:
                                trans_type = 1
                                dim_old = 4
                            else:
                                print('error')
                                assert 0
                        else:
                            dim_old = 4
                        v_4d = np.swapaxes(v_4d, 0, 2)  # swap H, I
                        v_4d = np.swapaxes(v_4d, 1, 3)  # swap W, O
                        if self.ROTATION == True:
                            shape_v = np.shape(v_4d)
                            print(np.shape(v_4d))
                            H = shape_v[2]
                            W = shape_v[3]
                            v_4d_temp = np.copy(v_4d)
                            print(np.shape(v_4d_temp))
                            for m in range(H):
                                for n in range(W):
                                    v_4d[:, :, m, n] = v_4d_temp[:, :, H - m - 1, W - n - 1]
                        net.params[caffe_var_name][var_site].data[:, :, :, :] = v_4d[:, :, :, :]

            print(self.weight2operation)
            for param in caffemodel_params:
                for i in self.weight2operation.keys():
                    if param in self.weight2operation[i]:
                        print(param, i)
                        var = graph.get_tensor_by_name(i + ':0')
                        if var is not None:
                            # print(var.eval())

                            var_name = i
                            caffe_var_name = param

                            var_type = self.weight2operation[var_name][2]
                            var_site = self.weight2operation[var_name][1]
                            v_4d = np.array(self.sess.run(var))
                            print('recover ' + var.name + ' to ' + caffe_var_name + ' ' + str(var_site))
                            if NORMAL == var_type:
                                if v_4d.ndim == 4:
                                    if dim_old != 4:
                                        if dim_old == 0:
                                            trans_type = 1
                                            dim_old = 4
                                        else:
                                            print('error')
                                            assert 0
                                    else:
                                        dim_old = 4
                                    v_4d = np.swapaxes(v_4d, 0, 2)  # swap H, I
                                    v_4d = np.swapaxes(v_4d, 1, 3)  # swap W, O
                                    v_4d = np.swapaxes(v_4d, 0, 1)  # swap I, O
                                    if self.ROTATION == True:
                                        shape_v = np.shape(v_4d)
                                        print(np.shape(v_4d))
                                        H = shape_v[2]
                                        W = shape_v[3]
                                        v_4d_temp = np.copy(v_4d)
                                        print(np.shape(v_4d_temp))
                                        for m in range(H):
                                            for n in range(W):
                                                v_4d[:, :, m, n] = v_4d_temp[:, :, H - m - 1, W - n - 1]
                                    net.params[caffe_var_name][var_site].data[:, :, :, :] = v_4d[:, :, :, :]
                                if v_4d.ndim == 2:
                                    if dim_old != 2:
                                        if dim_old == 0:
                                            trans_type = 0
                                            dim_old = 2
                                        else:
                                            TRANS = self.get_transform(var.name.decode())
                                            if TRANS:
                                                if TRANS[1] * TRANS[2] * TRANS[3] == list(np.shape(v_4d))[0]:
                                                    temp = np.swapaxes(v_4d, 0, 1)
                                                    temp = np.reshape(temp, (
                                                        list(np.shape(v_4d))[1], TRANS[1], TRANS[2], TRANS[3]))
                                                    temp = np.swapaxes(temp, 1, 3)  # swap H, I
                                                    temp = np.swapaxes(temp, 2, 3)  # swap W, O
                                                    v_4d = np.reshape(temp, np.shape(np.swapaxes(v_4d, 0, 1)))
                                                    # v_4d = np.swapaxes(temp, 0, 1)  # swap H, I
                                                    dim_old = 2
                                                else:
                                                    print('error')
                                                    assert 0
                                    else:
                                        dim_old = 2
                                        v_4d = np.swapaxes(v_4d, 0, 1)  # swap H, I
                                    if self.ROTATION == True:
                                        shape_v = np.shape(v_4d)
                                        print(np.shape(v_4d))
                                        W = shape_v[1]
                                        v_4d_temp = np.copy(v_4d)
                                        print(np.shape(v_4d_temp))
                                        for m in range(H):
                                            v_4d[:, m] = v_4d_temp[:, H - m - 1]
                                    net.params[caffe_var_name][var_site].data[:, :] = v_4d[:, :]
                                if v_4d.ndim == 1:
                                    net.params[caffe_var_name][var_site].data[:] = v_4d[:]
                            if DEPWISE == var_type:
                                if v_4d.ndim == 4:
                                    if dim_old != 4:
                                        if dim_old == 0:
                                            trans_type = 1
                                            dim_old = 4
                                        else:
                                            print('error')
                                            assert 0
                                    else:
                                        dim_old = 4
                                    v_4d = np.swapaxes(v_4d, 0, 2)  # swap H, I
                                    v_4d = np.swapaxes(v_4d, 1, 3)  # swap W, O
                                    if self.ROTATION == True:
                                        shape_v = np.shape(v_4d)
                                        # print(np.shape(v_4d))
                                        H = shape_v[2]
                                        W = shape_v[3]
                                        v_4d_temp = np.copy(v_4d)
                                        # print(np.shape(v_4d_temp))
                                        for m in range(H):
                                            for n in range(W):
                                                v_4d[:, :, m, n] = v_4d_temp[:, :, H - m - 1, W - n - 1]
                                    net.params[caffe_var_name][var_site].data[:, :, :, :] = v_4d[:, :, :, :]
            for key in self.ConstWeight_dict.keys():
                caffe_var_name = self.ConstWeight_dict[key][0]
                var_site = self.ConstWeight_dict[key][1]
                if caffe_var_name in caffemodel_params:
                    net.params[caffe_var_name][var_site].data[...] = self.ConstWeight_dict[key][2][...]

        print('left', caffemodel_params)
        net.save(os.path.join(self.filepath, self.netname + ".caffemodel"))
        print('save done! artosyn~')

    def set_caffe_input(self, caffe_input_name='', caffe_input_data=None):
        self.caffe_input_data = caffe_input_data
        self.caffe_input_name = caffe_input_name

    def check_output(self, tensorname='', caffe_layer='', caffe_data_name='', LEN=None, SAVEING=False,
                     savafile='savedata.txt'):
        if self.caffe_input_name == '' or not np.array(self.caffe_input_data).all():
            print('please set caffe_input first !')
            return
        graph = tf.get_default_graph()
        out = graph.get_tensor_by_name(tensorname)
        temp_data = self.sess.run(out,
                                  feed_dict=self.feed_dict)
        if len(np.shape(temp_data)) == 4:
            temp_data = np.swapaxes(temp_data, 1, 3)  # swap H, I
            temp_data = np.swapaxes(temp_data, 2, 3)  # swap W, O
        print('tensorflow-data-shape', np.shape(temp_data))
        print('tensorflow-data', temp_data)
        if SAVEING == True:
            sava_data = np.reshape(temp_data, [-1])
            ff = open(savafile, 'w')
            for i in sava_data:
                ff.write(str(i) + '\n')
            ff.close()
        net = caffe.Net(os.path.join(self.filepath, self.netname + ".prototxt"),
                        os.path.join(self.filepath, self.netname + ".caffemodel"), caffe.TEST)
        import copy
        print(np.shape(self.caffe_input_data))
        image = copy.deepcopy(self.caffe_input_data)

        if len(np.shape(self.caffe_input_data)) == 4:
            if self.ROTATION == True:
                v_4d_temp = image.copy()
                shape_v = np.shape(v_4d_temp)
                H = shape_v[2]
                W = shape_v[3]
                for m in range(H):
                    for n in range(W):
                        image[:, :, m, n] = v_4d_temp[:, :, H - m - 1, W - n - 1]
        if len(np.shape(self.caffe_input_data)) == 3:
            if self.ROTATION == True:
                v_4d_temp = image.copy()
                shape_v = np.shape(v_4d_temp)
                H = shape_v[1]
                W = shape_v[2]
                for m in range(H):
                    for n in range(W):
                        image[:, m, n] = v_4d_temp[:, H - m - 1, W - n - 1]

        net.blobs[self.caffe_input_name].data[...] = image
        out = net.forward(end=caffe_layer)[caffe_data_name]
        if len(np.shape(out)) == 4:
            if self.ROTATION == True:
                shape_v = np.shape(out)
                H = shape_v[2]
                W = shape_v[3]
                v_4d_temp = np.copy(out)
                print(np.shape(v_4d_temp))
                for m in range(H):
                    for n in range(W):
                        out[:, :, m, n] = v_4d_temp[:, :, H - m - 1, W - n - 1]
        print('caffe-data-shape', np.shape(out))
        print('caffe-data', out)
        if SAVEING == True:
            sava_data = np.reshape(out, [-1])
            ff = open(savafile, 'a')
            ff.write('next\n')
            for i in sava_data:
                ff.write(str(i) + '\n')
            ff.close()
        tensor_out_data = np.reshape(temp_data, [-1])
        caffe_out_data = np.reshape(out, [-1])
        print('mx', tensor_out_data)
        print(caffe_out_data)
        if LEN:
            error = tensor_out_data[:LEN] - caffe_out_data[:LEN]
            square_error = np.sum(np.square(error))
            print(square_error)
        else:
            error = tensor_out_data - caffe_out_data
            square_error = np.sum(np.square(error))
            print(square_error)
