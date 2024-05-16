# Copyright 2022 PaddleFSL Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddlefsl.backbones import GIN
from paddlefsl.backbones import All_Embedding
from examples.molecular_property_prediction.utils import get_args
import os
import pgl.graph as G
import numpy as np


def gin_test():
    METHOD = 'maml'
    SHOT = 10
    N_QUERY = 16
    META_LR = 0.001
    WEIGHT_DECAY = 5e-5
    INNER_LR = 0.05
    EPOCHS = 1000
    EVAL_STEPS = 10
    SEED = 0
    SAVE_MODEL_ITER = 1000
    args = get_args(root_dir = os.path.abspath(os.path.dirname(__file__)),
                    n_shot = SHOT,
                    n_query = N_QUERY,
                    meta_lr = META_LR,
                    weight_decay = WEIGHT_DECAY,
                    inner_lr = INNER_LR,
                    epochs = EPOCHS,
                    eval_steps = EVAL_STEPS,
                    seed = SEED,
                    save_model_iter = SAVE_MODEL_ITER,
                    method = METHOD)
    gin=GIN(args)

    num_nodes = 5
    edges = paddle.to_tensor([ (0, 1), (1, 2), (3, 4)])
    feature = paddle.randint(low=0,high=5,shape=[5, 100])
    edge_feature = paddle.randint(low=0,high=5,shape=[3, 100])
    graph = G.Graph(num_nodes=num_nodes,
                edges=edges,
                node_feat={
                    "feature": feature
                },
                edge_feat={
                    "feature": edge_feature
                })
    
    print(gin(graph)) # Tensor(shape=[5, 300]

def all_Embedding_test():
    num_nodes = 5
    edges = paddle.to_tensor([ (0, 1), (1, 2), (3, 4)])
    feature = paddle.randint(low=0,high=5,shape=[5, 100])
    edge_feature = paddle.randint(low=0,high=5,shape=[3, 100])
    graph = G.Graph(num_nodes=num_nodes,
                edges=edges,
                node_feat={
                    "feature": feature
                },
                edge_feat={
                    "feature": edge_feature
                })
    all_Embedding = All_Embedding(120, 3, 300)
    print(all_Embedding(graph.node_feat)) # Tensor(shape=[5, 300]

if __name__ == '__main__':
    all_Embedding_test()
    gin_test()
    

