import paddlefsl.utils as utils
import paddle.nn as nn

class test(nn.Layer):
    def __init__(self):
        super(test, self).__init__()
        self.fc_n = nn.Linear(10, 2)



if __name__ == '__main__':
    model_0 = test()
    # test(
    #   (fc_n): Linear(in_features=10, out_features=2, dtype=float32)
    # )
    # Number of parameters for fc_n = 0.02 k
    # Total parameters of model = 0.02 k
    print(utils.count_model_params(model_0))