import paddle
from paddle import fluid
from RelationNet import EmbeddingNet, RelationNet
import numpy as np
from paddle.fluid.optimizer import Adam


def build(main_prog, startup_prog, mode, c_way, k_shot):
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            sample_image = fluid.layers.data('sample_image', shape=[3, 84, 84], dtype='float32')
            query_image = fluid.layers.data('query_image', shape=[3, 84, 84], dtype='float32')
            query_label = fluid.layers.data('query_label', shape=[1], dtype='int64')

            embed_model = EmbeddingNet()
            RN_model = RelationNet()

            sample_shape = fluid.layers.shape(sample_image)[0]
            query_shape = fluid.layers.shape(query_image)[0]

            sample_query_image = fluid.layers.concat([sample_image, query_image], axis=0)
            sample_query_feature = embed_model.net(sample_query_image)

            sample_feature = fluid.layers.slice(
                sample_query_feature,
                axes=[0],
                starts=[0],
                ends=[sample_shape])
            if k_shot > 1:
            # few_shot
                sample_feature = fluid.layers.reshape(sample_feature, shape=[c_way, k_shot, 64, 19, 19])
                sample_feature = fluid.layers.reduce_sum(sample_feature, dim=1)
            query_feature = fluid.layers.slice(
                sample_query_feature,
                axes=[0],
                starts=[sample_shape],
                ends=[sample_shape + query_shape])

            sample_feature_ext = fluid.layers.unsqueeze(sample_feature, axes=0)
            query_shape = fluid.layers.concat(
                [query_shape, fluid.layers.assign(np.array([1, 1, 1, 1]).astype('int32'))])
            sample_feature_ext = fluid.layers.expand(sample_feature_ext, query_shape)

            query_feature_ext = fluid.layers.unsqueeze(query_feature, axes=0)
            if k_shot > 1:
                sample_shape = sample_shape / float(k_shot)
            sample_shape = fluid.layers.concat(
                [sample_shape, fluid.layers.assign(np.array([1, 1, 1, 1]).astype('int32'))])
            query_feature_ext = fluid.layers.expand(query_feature_ext, sample_shape)

            query_feature_ext = fluid.layers.transpose(query_feature_ext, [1, 0, 2, 3, 4])
            relation_pairs = fluid.layers.concat([sample_feature_ext, query_feature_ext], axis=2)
            relation_pairs = fluid.layers.reshape(relation_pairs, shape=[-1, 128, 19, 19])

            relation = RN_model.net(relation_pairs, hidden_size=8)
            relation = fluid.layers.reshape(relation, shape=[-1, c_way])
            fetch_list = [relation]
            feed_list = ['sample_image', 'query_image']
            if mode == 'train':
                one_hot_label = fluid.layers.one_hot(query_label, depth=c_way)
                loss = fluid.layers.square_error_cost(relation, one_hot_label)
                loss = fluid.layers.reduce_mean(loss)
                opt = Adam(learning_rate=0.001)
                opt.minimize(loss)
                fetch_list.append(loss)
                feed_list.append('query_label')


            return feed_list, fetch_list
