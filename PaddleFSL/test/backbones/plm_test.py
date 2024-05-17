import numpy as np
import paddle
from paddlefsl.backbones.plm import ErnieForPretraining,BertForPretraining,AlbertForPretraining,ErnieMLMCriterion
import paddlenlp as ppnlp


def test_ErnieForPretraining():

    model = ErnieForPretraining.from_pretrained('ernie-1.0')
    src_ids = paddle.randint(low=0,high=5,shape=[3, 10])
    token_type_ids=paddle.randint(low=0,high=5,shape=[3, 10])
    masked_positions=paddle.randint(low=0,high=5,shape=[3, 1])
    prediction_scores = model.predict(
        input_ids=src_ids,
        token_type_ids=token_type_ids,
        masked_positions=masked_positions)
    print(prediction_scores)  #Tensor(shape=[3, 18000])


def test_BertForPretraining():

    model = BertForPretraining.from_pretrained('bert-base-uncased')
    src_ids = paddle.randint(low=0,high=5,shape=[3, 10])
    token_type_ids=paddle.randint(low=0,high=5,shape=[3, 10])
    masked_positions=paddle.randint(low=0,high=5,shape=[3, 1])
    prediction_scores = model.predict(
        input_ids=src_ids,
        token_type_ids=token_type_ids,
        masked_positions=masked_positions)
    print(prediction_scores)  #Tensor(shape=[3, 30522])

def test_AlbertForPretraining():

    model = AlbertForPretraining.from_pretrained('albert-base-v1')
    src_ids = paddle.randint(low=0,high=5,shape=[3, 10])
    token_type_ids=paddle.randint(low=0,high=5,shape=[3, 10])
    masked_positions=paddle.randint(low=0,high=5,shape=[3, 1])
    prediction_scores = model.predict(
        input_ids=src_ids,
        token_type_ids=token_type_ids,
        masked_positions=masked_positions)
    print(prediction_scores)  #Tensor(shape=[3, 10, 30000])

def test_ErnieMLMCriterion():
    model = ErnieMLMCriterion()
    prediction_scores = paddle.rand(shape=[16, 10])
    masked_lm_labels=paddle.randint(low=1,high=5,shape=[16, 1])
    print(model(prediction_scores, masked_lm_labels, weights=None))  #Tensor(shape=[1])


if __name__ == "__main__":
    test_ErnieForPretraining()
    test_BertForPretraining()
    test_AlbertForPretraining()
    test_ErnieMLMCriterion()