import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.layers import BatchNormalization, Conv2D
from params import Params


def conv2d_blk(img_L, img_R, name, kernel, filters, stride, phase):
    conv2_scope = k.layers.Conv2D(name=name, kernel_size=kernel, filters=filters, strides=[stride, stride],
                                  padding="same", trainable=phase)
    btn = k.layers.BatchNormalization(axis=-1, trainable=phase)
    act = k.layers.Activation("relu", trainable=phase)
    h_1_L = act(btn(conv2_scope(img_L)))
    h_1_R = act(btn(conv2_scope(img_R)))
    return h_1_L, h_1_R


def conv3d_blk(x, name, kernel, filters, strid, phase):
    conv3d = k.layers.Conv3D(name=name, kernel_size=kernel, filters=filters, strides=strid, padding="same",
                             trainable=phase)(x)
    x = k.layers.BatchNormalization()(conv3d)
    x = k.layers.Activation('relu')(x)
    return x


def deconv3d_blk(x, name, kernal, filters, strid, pahse):
    deconv3d_blk = k.layers.Conv3DTranspose(name=name, kernel_size=kernal, filters=filters, strides=strid,
                                            padding="same", activation=k.activations.relu, trainable=pahse)(x)
    deconv = BatchNormalization(axis=-1)(deconv3d_blk)
    deconv = k.layers.Activation("relu")(deconv)
    return deconv


def res_blk(h_conv1_L, h_conv1_R, name, kernel, filters, stride, phase):
    h_L_a, h_R_a = conv2d_blk(h_conv1_L, h_conv1_R, name=name + "a", kernel=kernel, filters=filters, stride=stride,
                              phase=phase)
    h_L_b, h_R_b = conv2d_blk(h_L_a, h_R_a, name=name + "b", kernel=kernel, filters=filters, stride=stride, phase=phase)

    h_conv3_L_c = k.layers.Add()([h_L_b, h_conv1_L])
    h_conv3_R_c = k.layers.Add()([h_R_b, h_conv1_R])

    return h_conv3_L_c, h_conv3_R_c


def cost_volume(img_L, img_R, d_size):
    """
    Cost Volume - each pixel in img_L concat horizontally across img_R
    """
    d = int(d_size / 2 - 1)
    dp_list = []

    # when disparity is 0
    elw_tf = tf.concat([img_L, img_R], axis=3, name="concat0")
    dp_list.append(elw_tf)

    # right side
    for dis in range(d):
        # moving the features by disparity d can be done by padding zeros
        pad = tf.constant([[0, 0], [0, 0], [dis + 1, 0], [0, 0]], dtype=tf.int32, name="con" + str(dis))
        pad_R = tf.pad(img_R[:, :, :-1 - dis, :], pad, "CONSTANT", name="pad" + str(dis))
        elw_tf = tf.concat([img_L, pad_R], axis=3, name="concat" + str(dis + 1))
        dp_list.append(elw_tf)
    # print("a", tf.convert_to_tensor(dp_list).shape)
    total_pack_tf = tf.concat(dp_list, axis=0, name="concat_x")
    total_pack_tf = tf.expand_dims(total_pack_tf, 0, name="total_pack_tf")
    return total_pack_tf


def _getCostVolume_(inputs, max_d):
    max_d = int(max_d)
    left_tensor, right_tensor = inputs
    shape = k.backend.shape(right_tensor)
    right_tensor = k.backend.spatial_2d_padding(right_tensor, padding=((0, 0), (max_d, 0)))
    disparity_costs = []
    for d in reversed(range(max_d)):
        left_tensor_slice = left_tensor
        right_tensor_slice = tf.slice(right_tensor, begin=[0, 0, d, 0], size=[-1, -1, shape[2], -1])
        cost = k.backend.concatenate([left_tensor_slice, right_tensor_slice], axis=3)
        disparity_costs.append(cost)
    cost_volume = k.backend.stack(disparity_costs, axis=1)
    return cost_volume


def build_model(img_l, img_r, phase=True):
    parameters = Params()
    input_l = k.layers.Input(tensor=img_l, name="img_l")
    input_r = k.layers.Input(tensor=img_r, name="img_r")
    h_1_L, h_1_R = conv2d_blk(input_l, input_r, name="conv1", kernel=(5, 5), filters=32, stride=2, phase=phase)
    h_3_L, h_3_R = res_blk(h_1_L, h_1_R, name="res2-3", kernel=(3, 3), filters=32, stride=1, phase=phase)
    h_5_L, h_5_R = res_blk(h_3_L, h_3_R, name="res4-5", kernel=(3, 3), filters=32, stride=1, phase=phase)
    h_7_L, h_7_R = res_blk(h_5_L, h_5_R, name="res6-7", kernel=(3, 3), filters=32, stride=1, phase=phase)
    h_9_L, h_9_R = res_blk(h_7_L, h_7_R, name="res8-9", kernel=(3, 3), filters=32, stride=1, phase=phase)
    h_11_L, h_11_R = res_blk(h_9_L, h_9_R, name="res10-11", kernel=(3, 3), filters=32, stride=1, phase=phase)
    h_13_L, h_13_R = res_blk(h_11_L, h_11_R, name="res11-13", kernel=(3, 3), filters=32, stride=1, phase=phase)
    h_15_L, h_15_R = res_blk(h_13_L, h_13_R, name="res14-15", kernel=(3, 3), filters=32, stride=1, phase=phase)
    h_17_L, h_17_R = res_blk(h_15_L, h_15_R, name="res16-17", kernel=(3, 3), filters=32, stride=1, phase=phase)
    h_18_L, h_18_R = conv2d_blk(h_17_L, h_17_R, name="conv18", kernel=(3, 3), filters=32, stride=1, phase=phase)

    # corr = cost_volume(h_18_L, h_18_R, parameters.max_disparity)
    corr = k.layers.Lambda(_getCostVolume_, arguments={'max_d': parameters.max_disparity / 2},
                           output_shape=(parameters.max_disparity / 2, None, None, 32 * 2))([h_18_L, h_18_R])

    # print("Shape corr", corr.get_shape())
    h_19 = conv3d_blk(x=corr, name="conv19", kernel=(3, 3, 3), filters=32, strid=1, phase=phase)
    h_20 = conv3d_blk(x=h_19, name="conv20", kernel=(3, 3, 3), filters=32, strid=1, phase=phase)
    h_21 = conv3d_blk(x=corr, name="conv21", kernel=(3, 3, 3), filters=64, strid=2, phase=phase)
    h_22 = conv3d_blk(x=h_21, name="conv22", kernel=(3, 3, 3), filters=64, strid=1, phase=phase)
    h_23 = conv3d_blk(x=h_22, name="conv23", kernel=(3, 3, 3), filters=64, strid=1, phase=phase)
    h_24 = conv3d_blk(x=h_21, name="conv24", kernel=(3, 3, 3), filters=64, strid=2, phase=phase)
    h_25 = conv3d_blk(x=h_24, name="conv25", kernel=(3, 3, 3), filters=64, strid=1, phase=phase)
    h_26 = conv3d_blk(x=h_25, name="conv26", kernel=(3, 3, 3), filters=64, strid=1, phase=phase)
    h_27 = conv3d_blk(x=h_24, name="conv27", kernel=(3, 3, 3), filters=64, strid=2, phase=phase)
    h_28 = conv3d_blk(x=h_27, name="conv28", kernel=(3, 3, 3), filters=64, strid=1, phase=phase)
    h_29 = conv3d_blk(x=h_28, name="conv29", kernel=(3, 3, 3), filters=64, strid=1, phase=phase)
    h_30 = conv3d_blk(x=h_27, name="conv30", kernel=(3, 3, 3), filters=128, strid=2, phase=phase)
    h_31 = conv3d_blk(x=h_30, name="conv31", kernel=(3, 3, 3), filters=128, strid=1, phase=phase)
    h_32 = conv3d_blk(x=h_31, name="conv32", kernel=(3, 3, 3), filters=128, strid=1, phase=phase)

    h_33_a = deconv3d_blk(x=h_32, name="deconv33", kernal=(3, 3, 3), filters=64, strid=2, pahse=phase)
    h_33_b = k.layers.Add()([h_33_a, h_29])

    h_34_a = deconv3d_blk(x=h_33_b, name="deconv34", kernal=(3, 3, 3), filters=64, strid=2, pahse=phase)
    h_34_b = k.layers.Add()([h_34_a, h_26])

    h_35_a = deconv3d_blk(x=h_34_b, name="deconv35", kernal=(3, 3, 3), filters=64, strid=2, pahse=phase)
    h_35_b = k.layers.Add()([h_35_a, h_23])

    h_36_a = deconv3d_blk(x=h_35_b, name="deconv36", kernal=(3, 3, 3), filters=32, strid=2, pahse=phase)
    h_36_b = k.layers.Add()([h_36_a, h_20])

    h_37 = deconv3d_blk(x=h_36_b, name="deconv37", kernal=(3, 3, 3), filters=1, strid=2, pahse=phase)

    sqz = tf.squeeze(h_37, 4)

    trans = tf.transpose(sqz, perm=[0, 2, 3, 1])

    neg = tf.negative(trans)
    logits = k.activations.softmax(neg)

    distrib = k.layers.Conv2D(kernel_size=(1, 1), padding="same",
                              filters=1, strides=1, trainable=phase)(logits)

    return k.models.Model(inputs=[input_l, input_r], outputs=distrib)


def keras_asl(tgt, pred):
    return tf.losses.absolute_difference(predictions=pred, labels=tgt)


if __name__ == '__main__':
    import util
    import params

    p = params.Params()
    SUM_OF_ALL_DATASAMPLES = 44780
    STEPS_PER_EPOCH = SUM_OF_ALL_DATASAMPLES / p.batch_size

    train_dir = 'saved_model/'
    data_record = ["dataset/fly_train.tfrecords", "dataset/fly_test.tfrecords"]

    train_img_l_b, train_img_r_b, train_d_b = util.read_and_decode(p, data_record[0])
    test_img_l_b, test_img_r_b, test_d_b = util.read_and_decode(p, data_record[1])

    model = build_model(train_img_l_b, train_img_r_b)
    opt = k.optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=opt, loss=keras_asl, target_tensors=[train_d_b])
    print(model.summary())
    print(type(train_img_r_b))
    model.fit(epochs=10, verbose=1, steps_per_epoch=STEPS_PER_EPOCH)
