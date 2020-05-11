class Layer(object):
    def __init__(self):
        self.name=''
        self.type=''
        self.top=[]
        self.bottom=[]
        self.param=[]
        self.input_param=[]
        self.convolution_param=[]
        self.lrn_param=[]
        self.pooling_param=[]
        self.relu_param=[]
        self.inner_product_param=[]
        self.dropout_param=[]
        self.reshape_param=[]
        self.concat_param=[]
        self.batchnormal_param=[]
        self.scale_param=[]
        self.bais_param=[]
        self.add_param=[]

    def write2file(self):
        return 0

class InputLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='data'
        self.type='Input'
        self.inputshape=[]
        #self.file=filepath
class FCNLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='fcn'
        self.type='InnerProduct'
        self.num_output=0
class COV2DLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='cov2d'
        self.type='Convolution'
        self.num_output=0
        self.kernel_size=0
        self.pad=0
        self.stride=0
class PoolLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='pool'
        self.type='Pooling'
        self.pool='MAX'
        self.kernel_size=0
        self.stride=0
class ReLULayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='relu'
        self.type="ReLU"
class SoftmaxLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='softmax'
        self.type="Softmax"
class LRNLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='lrn'
        self.type='LRN'
        self.local_size=5
        self.alpha=0.0001
        self.beta=0.5
class ReshapeLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='reshape'
        self.type='Reshape'
        self.shape=[]
class ConcatLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='concat'
        self.type='Concat'
        self.axis=1

class BatchnormalLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='batchnorm'
        self.type='BatchNorm'

class ScaleLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='scale'
        self.type='Scale'
class BaisLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='bais'
        self.type='Bias'
class AddLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name='add'
        self.type='Eltwise'
SAFE=True






def write2file(layer,filepath):
    ff=open(filepath,'a')
    ff.write('layer {\n')
    ff.write("  name: \"%s\"\n"%layer.name)
    ff.write("  type: \"%s\"\n"%layer.type)
    for i in layer.bottom:
        ff.write("  bottom: \"%s\"\n" % i)
    for i in layer.top:
        ff.write("  top: \"%s\"\n" % i)
    for i in layer.param:
        ff.write("  param {\n")
        for j in i.keys():
            ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.input_param:
        ff.write("  input_param {\n")
        for j in i.keys():
            if j=='shape':
                ff.write("      %s:{" % j )
                for k in i[j]:
                    ff.write(" dim: %d"%k)
                ff.write('}\n')
            else:
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.reshape_param:
        ff.write("  reshape_param {\n")
        for j in i.keys():
            if j=='shape':
                ff.write("      %s:{" % j )
                for k in i[j]:
                    ff.write(" dim: %d"%k)
                ff.write('}\n')
            else:
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.add_param:
        ff.write("  eltwise_param {\n")
        for j in i.keys():
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.inner_product_param:
        ff.write("  inner_product_param {\n")
        for j in i.keys():
            if type(i[j]) != dict:
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.dropout_param:
        ff.write("  dropout_param {\n")
        for j in i.keys():
            if type(i[j]) != dict:
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.batchnormal_param:
        ff.write("  batch_norm_param {\n")
        for j in i.keys():
            if type(i[j]) != dict:
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.scale_param:
        ff.write("  scale_param {\n")
        for j in i.keys():
            if type(i[j]) != dict:
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.convolution_param:
        ff.write("  convolution_param {\n")
        for j in i.keys():
            if type(i[j]) != dict:
                if SAFE:
                    if str(i[j])=='?':
                        i[j]=1
                    ff.write("      %s:" % j + str(i[j]) + '\n')
                else:
                    ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.lrn_param:
        ff.write("  lrn_param {\n")
        for j in i.keys():
            if type(i[j]) != dict:
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.pooling_param:
        ff.write("  pooling_param {\n")
        for j in i.keys():
            if type(i[j])!= dict:
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.concat_param:
        ff.write("  concat_param {\n")
        for j in i.keys():
            if type(i[j])!= dict:
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    for i in layer.relu_param:
        ff.write("  relu_param {\n")
        for j in i.keys():
            if type(i[j])!= dict:
                ff.write("      %s:"%j+str(i[j])+'\n')
        ff.write("  }\n")
    ff.write("}\n")
    ff.close()






