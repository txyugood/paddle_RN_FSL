import paddle
import paddle.fluid as fluid
from paddle.fluid import ParamAttr

import numpy as np
import math


class BaseNet:
    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      padding=0,
                      act=None,
                      name=None,
                      data_format='NCHW'):
        n = filter_size * filter_size * num_filters
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights",
                                 initializer=fluid.initializer.Normal(0, math.sqrt(2. / n))),
            bias_attr=ParamAttr(name=name + "_bias",
                                initializer=fluid.initializer.Constant(0.0)),
            name=name + '.conv2d.output.1',
            data_format=data_format)

        bn_name = "bn_" + name

        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            momentum=1,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale',
                                 initializer=fluid.initializer.Constant(1)),
            bias_attr=ParamAttr(bn_name + '_offset',
                                initializer=fluid.initializer.Constant(0)),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            data_layout=data_format)

class EmbeddingNet(BaseNet):
    def net(self,input):
        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=3,
            padding=0,
            act='relu',
            name='embed_conv1')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=2,
            pool_stride=2,
            pool_type='max')
        conv = self.conv_bn_layer(
            input=conv,
            num_filters=64,
            filter_size=3,
            padding=0,
            act='relu',
            name='embed_conv2')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=2,
            pool_stride=2,
            pool_type='max')
        conv = self.conv_bn_layer(
            input=conv,
            num_filters=64,
            filter_size=3,
            padding=1,
            act='relu',
            name='embed_conv3')
        conv = self.conv_bn_layer(
            input=conv,
            num_filters=64,
            filter_size=3,
            padding=1,
            act='relu',
            name='embed_conv4')
        return conv


class RelationNet(BaseNet):
    def net(self, input, hidden_size):
        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=3,
            padding=0,
            act='relu',
            name='rn_conv1')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=2,
            pool_stride=2,
            pool_type='max')
        conv = self.conv_bn_layer(
            input=conv,
            num_filters=64,
            filter_size=3,
            padding=0,
            act='relu',
            name='rn_conv2')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=2,
            pool_stride=2,
            pool_type='max')
        fc = fluid.layers.fc(conv,size=hidden_size,act='relu',
                             param_attr=ParamAttr(name='fc1_weights',
                                                  initializer=fluid.initializer.Normal(0,0.01)),
                             bias_attr=ParamAttr(name='fc1_bias',
                                                 initializer=fluid.initializer.Constant(1)),
                             )
        fc = fluid.layers.fc(fc, size=1,act='sigmoid',
                             param_attr=ParamAttr(name='fc2_weights',
                                                  initializer=fluid.initializer.Normal(0,0.01)),
                             bias_attr=ParamAttr(name='fc2_bias',
                                                 initializer=fluid.initializer.Constant(1)),
                             )
        return fc