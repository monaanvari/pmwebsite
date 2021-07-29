'''
Credit: github.com/lynetcha
there to recreate the saved model predicting the parametric material from a state dictionary.
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Template-based parametric material prediction')

    # Optimization arguments
    parser.add_argument('--optim', default='adagrad', help='Optimizer: sgd|adam|adagrad|adadelta|rmsprop')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')

    #Training/Testing arguments
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--resume', type=int, default=0, help='If 1, resume training')
    parser.add_argument('--eval', default=0, type=int, help='If 1, evaluate using best model')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--nworkers', default=4, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')


    # Model
    parser.add_argument('--NET', default='Network', help='Network used')
    parser.add_argument('--base_model', default='resnet34', help='Encoder')
    parser.add_argument('--regression', default=1, type=int, help='Use the model with regression')
    parser.add_argument('--mlp', default=1, type=int, help='Number of MLP layers')
    parser.add_argument('--alpha', default=1e-2, type=float, help='Loss coefficient')

    # Dataset
    parser.add_argument('--dataset', default='vMaterials', help='Dataset name')
    parser.add_argument('--num_classes', default=314, type=int, help='Number of material classes')

    args = parser.parse_args()
    args.start_epoch = 0
    return args
