# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import tensorflow as tf
from tensorflow_addons.text.crf import crf_decode, crf_log_likelihood

# 关闭eager模式
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


class LstmWithCRFModel(object):
    """ lstm + crf解码模型

    shape 输入形状 (句子数 3，句子长度 6)
    batch_size （批次大小）：表示每个训练迭代中用于更新模型的样本数量。较大的批次大小可以提高训练速度，但也可能导致内存消耗增加。
    embedding_size （嵌入维度）：定义了将词语映射到连续向量空间的维度。嵌入维度的选择通常是模型超参数，较大的嵌入维度可以提供更丰富的语义信息，但也会增加模型复杂度和训练时间。
    决定了将词语转换为连续向量表示时的维度，用于捕捉词语之间的语义关系。
    word_size（词汇大小）：指定词汇表的大小，即模型可以接受的不同词汇数量。较大的词汇大小可以覆盖更多的词汇，但也会增加模型的复杂度和训练时间。

    hidden_size（隐藏状态维度）：指定LSTM隐藏状态的维度大小。较大的隐藏状态维度可以提供更强大的模型表示能力，但也会增加模型复杂度和训练时间。
    layer_size（LSTM层数）：定义LSTM模型中的LSTM层数。较深的LSTM模型可以学习更复杂的语义表示，但也会增加模型的复杂度和训练时间。
    learning_rate（学习率）：控制模型在每次迭代中更新参数的步长。较小的学习率可以使模型更加稳定，但训练速度可能较慢；较大的学习率可能导致训练不稳定或无法收敛。
    num_epochs（训练轮数）：指定模型在训练数据上进行迭代的次数。较大的训练轮数可以增加模型的训练时间，但也有助于提高模型的性能和泛化能力。
    """

    def __init__(self, batch_size, n_class, ball_num, w_size, embedding_size, words_size, hidden_size, layer_size):
        self._inputs = tf.keras.layers.Input(
            shape=(w_size, ball_num), batch_size=batch_size, name="inputs"
        )
        self._tag_indices = tf.keras.layers.Input(
            shape=(ball_num, ), batch_size=batch_size, dtype=tf.int32, name="tag_indices"
        )
        self._sequence_length = tf.keras.layers.Input(
            shape=(), batch_size=batch_size, dtype=tf.int32, name="sequence_length"
        )
        # 构建特征抽取
        embedding = tf.keras.layers.Embedding(words_size, embedding_size)(self._inputs)
        first_lstm = tf.convert_to_tensor(
            [tf.keras.layers.LSTM(hidden_size)(embedding[:, :, i, :]) for i in range(ball_num)]
        )
        first_lstm = tf.transpose(first_lstm, perm=[1, 0, 2])
        second_lstm = None
        for _ in range(layer_size):
            second_lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(first_lstm)
        self._outputs = tf.keras.layers.Dense(n_class)(second_lstm)
        # 构建损失函数
        self._log_likelihood, self._transition_params = crf_log_likelihood(
            self._outputs, self._tag_indices, self._sequence_length
        )
        self._loss = tf.reduce_sum(-self._log_likelihood)
        #  构建预测
        self._pred_sequence, self._viterbi_score = crf_decode(
            self._outputs, self._transition_params, self._sequence_length
        )

    @property
    def inputs(self):
        return self._inputs

    @property
    def tag_indices(self):
        return self._tag_indices

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def outputs(self):
        return self._outputs

    @property
    def transition_params(self):
        return self._transition_params

    @property
    def loss(self):
        return self._loss

    @property
    def pred_sequence(self):
        return self._pred_sequence


class SignalLstmModel(object):
    """ 单向lstm序列模型
    """

    def __init__(self, batch_size, n_class, w_size, embedding_size, hidden_size, outputs_size, layer_size, num_outputs):
        self._inputs = tf.keras.layers.Input(
            shape=(w_size, ), batch_size=batch_size, dtype=tf.int32, name="inputs"
        )
        self._tag_indices = tf.keras.layers.Input(
            shape=(n_class, ), batch_size=batch_size, dtype=tf.float32, name="tag_indices"
        )
        embedding = tf.keras.layers.Embedding(outputs_size, embedding_size)(self._inputs)
        lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(embedding)
        for _ in range(layer_size):
            lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(lstm)
        final_lstm = tf.keras.layers.LSTM(hidden_size, recurrent_dropout=0.2)(lstm)

        # 添加多个输出层
        outputs = []
        for _ in range(num_outputs):
            dense_layer = tf.keras.layers.Dense(outputs_size, activation="softmax")(final_lstm)
            outputs.append(dense_layer)

        # 输出为一个列表，每个元素对应一个输出结果
        self._outputs = outputs

        # self._outputs = tf.keras.layers.Dense(outputs_size, activation="softmax")(final_lstm)
        # 构建损失函数
        self._loss = - tf.reduce_sum(self._tag_indices * tf.math.log(self._outputs))
        # 预测结果
        self._pred_label = tf.argmax(self.outputs, axis=1)

    @property
    def inputs(self):
        return self._inputs

    @property
    def tag_indices(self):
        return self._tag_indices

    @property
    def outputs(self):
        return self._outputs

    @property
    def loss(self):
        return self._loss

    @property
    def pred_label(self):
        return self._pred_label
