import caffe
import numpy as np
import cv2
#image_file='/home/lucky/SSD/caffe/data/mydata/train/1/tuoxie337.jpg'
image_file='caffe/dog.jpg'
caffe.set_mode_cpu()
#net = caffe.Net('caffemodel/lenet.prototxt', 'caffemodel/test.caffemodel', caffe.TEST)
net = caffe.Net('mtcnn-caffe-master/48net/48net.prototxt','mtcnn-caffe-master/48net/48net.caffemodel', caffe.TEST)
print(net.params)
print('lucky',net.params['prelu1'][0].data)
assert 0
#net = caffe.Net('caffemodel/lenet.prototxt', 'caffemodel/lenet_iter_10000.caffemodel', caffe.TEST)
#print(net.params['conv1'][0].data[ :])
#assert 0
#net = caffe.Net('caffe/ssdlite_moiblenet.prototxt','caffe/ssdlite_moiblenet.caffemodel', caffe.TEST)
net.blobs['image_tensor'].reshape(1, 3, 300, 300)
transformer = caffe.io.Transformer({'image_tensor': net.blobs['image_tensor'].data.shape})
#image = caffe.io.load_image(image_file)
image=cv2.imread(image_file)
image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
image=cv2.resize(image,(300,300))
image=np.swapaxes(image,1,2)
image=np.swapaxes(image,0,1)

transformed_image=np.reshape(image,(1,3,300,300))
net.blobs['image_tensor'].data[...] = transformed_image
out=net.forward(end='BoxPredictor_5/ClassPredictor/Conv2Dtsfw')['BoxPredictor_5/ClassPredictor/Conv2D']
print(np.shape(out))
print('LUCKY0',out)
