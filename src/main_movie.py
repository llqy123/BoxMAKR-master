import argparse
import numpy as np
from data_loader import RSDataset, KGDataset
from train import train

np.random.seed(555)

parser = argparse.ArgumentParser()

# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=0.01, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=0.01, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')
parser.add_argument('--attention', type=bool, default=False, help='use attention mechanism')
parser.add_argument('--channel', type=int, default=8, help='number of channels in attention unit')
parser.add_argument('--hge', type=bool, default=False, help='use HGE')

parser.add_argument('--workers', type=int, default=0,
                    help='number of data loading workers')
parser.add_argument('-sl', '--show_loss', action='store_true',
                    help='show loss or not')
parser.add_argument('-st', '--show_topk', action='store_true',
                    help='show topK or not')
parser.add_argument('-sum', '--summary_path', type=str, default='..\\result',
                    help='path to store training summary')
args = parser.parse_args()

rs_dataset = RSDataset(args)
kg_dataset = KGDataset(args)
train(args, rs_dataset, kg_dataset)