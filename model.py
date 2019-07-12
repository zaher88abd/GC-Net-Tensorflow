import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.layers import BatchNormalization, Conv2D
from params import Params


def conv2d_blk(img_L, img_R, name, kernel, filters, stride, phase):
    conv2_scope = k.layers.Conv2D(name=name, kernel_size=kernel, filters=filters, strides=[stride, stride],
                                  padding="same", trainable=phase)
    h_1_L = conv2_scope(img_L)
    h_1_R = conv2_scope(img_R)
    return h_1_L, h_1_R


def conv3d_blk(x, name, kernel, filters, strid, phase):
    x = k.layers.BatchNormalization()(x)
    conv3d = k.layers.Conv3D(name=name, kernel_size=kernel, filters=filters, strides=strid, padding="same",
                             activation=k.activations.relu, trainable=phase)(x)
    return conv3d


def deconv3d_blk(x, name, kernal, filters, strid, pahse):
    deconv3d_blk = k.layers.Conv3DTranspose(name=name, kernel_size=kernal, filters=filters, strides=strid,
                                            padding="same", activation=k.activations.relu, trainable=pahse)(x)
    return deconv3d_blk


def res_blk(h_conv1_L, h_conv1_R, name, kernel, filters, stride, phase):
    h_conv2_L_a = k.layers.BatchNormalization(trainable=phase)(h_conv1_L)
    h_conv2_R_a = k.layers.BatchNormalization(trainable=phase)(h_conv1_R)

    conv2_scope = k.layers.Conv2D(name=name + "conv_a", kernel_size=kernel, filters=filters, strides=stride,
                                  padding="same", trainable=phase)
    h_conv2_L_b = conv2_scope(h_conv2_L_a)
    h_conv2_r_b = conv2_scope(h_conv2_R_a)

    h_conv3_L_a = k.layers.BatchNormalization(trainable=phase)(h_conv2_L_b)
    h_conv3_R_a = k.layers.BatchNormalization(trainable=phase)(h_conv2_r_b)

    conv3_scope = k.layers.Conv2D(name=name + "conv_b", kernel_size=kernel, filters=filters, strides=stride,
                                  padding="same", trainable=phase)
    h_conv3_L_b = conv3_scope(h_conv3_L_a)
    h_conv3_R_b = conv3_scope(h_conv3_R_a)

    h_conv3_L_c = h_conv3_L_b + h_conv1_L
    h_conv3_R_c = h_conv3_R_b + h_conv1_R

    return h_conv3_L_c, h_conv3_R_c


def cost_volume(img_L, img_R, d_size):
    """
    Cost Volume - each pixel in img_L concat horizontally across img_R
    """
    d = int(d_size / 2 - 1)
    dp_list = []

    # when disparity is 0
    elw_tf = tf.concat([img_L, img_R], axis=3)
    dp_list.append(elw_tf)

    # right side
    for dis in range(d):
        # moving the features by disparity d can be done by padding zeros
        pad = tf.constant([[0, 0], [0, 0], [dis + 1, 0], [0, 0]], dtype=tf.int32)
        pad_R = tf.pad(img_R[:, :, :-1 - dis, :], pad, "CONSTANT")
        elw_tf = tf.concat([img_L, pad_R], axis=3)
        dp_list.append(elw_tf)
    # print("a", tf.convert_to_tensor(dp_list).shape)
    total_pack_tf = tf.concat(dp_list, axis=0)
    total_pack_tf = tf.expand_dims(total_pack_tf, 0)
    return total_pack_tf


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

    corr = cost_volume(h_18_L, h_18_R, parameters.max_disparity)

    print("Shape corr", corr.get_shape())
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
    h_33_b = h_33_a + h_29

    h_34_a = deconv3d_blk(x=h_33_b, name="deconv34", kernal=(3, 3, 3), filters=64, strid=2, pahse=phase)
    h_34_b = h_34_a + h_26

    h_35_a = deconv3d_blk(x=h_34_b, name="deconv35", kernal=(3, 3, 3), filters=64, strid=2, pahse=phase)
    h_35_b = h_35_a + h_23

    h_36_a = deconv3d_blk(x=h_35_b, name="deconv36", kernal=(3, 3, 3), filters=32, strid=2, pahse=phase)
    h_36_b = h_36_a + h_20

    h_37 = deconv3d_blk(x=h_36_b, name="deconv37", kernal=(3, 3, 3), filters=1, strid=2, pahse=phase)

    sqz = tf.squeeze(h_37, 4)

    trans = tf.transpose(sqz, perm=[0, 2, 3, 1])

    neg = tf.negative(trans)
    logits = k.activations.softmax(neg)

    distrib = k.layers.Conv2D(kernel_size=(1, 1), padding="same",
                              filters=1, strides=1, trainable=phase)(logits)

    return k.models.Model(inputs=[input_l, input_r], outputs=distrib)


def keras_asl(tgt, pred):
    return tf.losses.absolute_difference(pred, tgt)


if __name__ == '__main__':
    import util
    import params

    train_dir = 'saved_model/'
    data_record = ["dataset/fly_train.tfrecords", "dataset/fly_test.tfrecords"]

    p = params.Params()

    train_img_l_b, train_img_r_b, train_d_b = util.read_and_decode(p, data_record[0])
    test_img_l_b, test_img_r_b, test_d_b = util.read_and_decode(p, data_record[1])

    model = build_model(train_img_l_b, train_img_r_b)
    opt = k.optimizers.RMSprop()
    model.compile(optimizer=opt, loss=keras_asl, target_tensors=[train_d_b])
    print(model.summary())
    print(type(train_img_r_b))
    # model.fit(epochs=10, verbose=1,steps_per_epoch=)
