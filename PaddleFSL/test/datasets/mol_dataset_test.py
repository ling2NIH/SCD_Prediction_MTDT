from paddlefsl.model_zoo import maml_mol
import paddlefsl.datasets as datasets
from examples.molecular_property_prediction.utils import get_args
import os

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

METHOD = 'maml'
DATASET = 'tox21'
TRAIN_DATASET = datasets.mol_dataset.load_dataset(dataset = DATASET, type = 'train')
TEST_DATASET = datasets.mol_dataset.load_dataset(dataset = DATASET, type = 'test')
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

test_task = args.test_tasks[0]
train_task = args.train_tasks[0]

def test_data_sample_test():
    adapt_data, eval_data, query = TEST_DATASET[test_task].get_test_data_sample(test_task, args.n_shot_test, args.n_query, args.update_step_test)
    print(adapt_data) #dict{'s_data': {}, 's_label':Tensor(shape=[20,1])}
    print(eval_data) #dict{'s_data': {}, 's_label':Tensor(shape=[20,1])}
    print(query)

def train_data_sample_test():
    adapt_data, eval_data = TRAIN_DATASET[train_task].get_train_data_sample(train_task, args.n_shot_train, args.n_query)
    print(adapt_data) #dict{'s_data': {}, 's_label':Tensor(shape=[16,1],'q_data': {}, 'q_label':Tensor(shape=[16,1])}

if __name__ == '__main__':
    test_data_sample_test()
    train_data_sample_test()