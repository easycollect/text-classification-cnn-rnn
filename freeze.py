import tensorflow as tf

version = 1.0
output_graph = "./pb/model_{}.pb".format(version)
output_tflite_model = "./tflite/model_{}.tflite".format(version)


def freeze_graph(input_checkpoint):
    """
    :param input_checkpoint:
    :return:
    """
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "score/fc2/BiasAdd"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(",")
        )  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出


def convert_to_tflite():
    input_tensors = [
        "input_x"
    ]
    output_tensors = [
        "score/fc2/BiasAdd"
    ]
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        output_graph,
        input_tensors,
        output_tensors)
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                            tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(output_tflite_model, "wb").write(tflite_model)


if __name__ == "__main__":
    # freeze_graph("./checkpoints/textcnn/best_validation")
    convert_to_tflite()

