import argparse
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"


class argParser:
    """
    Temporary Argument Settings
    """

    def __init__(self):
        self.batch_size = 10240
        self.test_batch_size = 10240
        self.embedding_dim = 64
        self.final_integration = 'ATT'
        self.att_norm = 'softmax'
        self.att_hidden_dims = [128, 64, 16, 1]
        self.num_layers = 3
        self.lr = 1e-2
        self.epochs = 1000
        self.seed = 2024
        self.tauk = 0.1
        self.taul = 0.1
        self.taus = 0.1
        self.prefix = "pdagnn-"
        self.dataset = 'Amazon-Book'
        self.topK = [5, 10, 20, 30, 50, 100]
        self.ifRegEmbedding = True
        self.weight_decay_embed = 1e-4
        self.ifRegBehav = True
        self.weight_decay_behavior = 1e-4
        self.ifLoad = False
        self.ifDropOut = True
        self.keep_prob = 0.6
        self.load_model_name = "pdagnn-epoch-1000.pth.tar"


def argParserV2():
    parser = argparse.ArgumentParser(
        description="A Propagation Depth Oriented Data Augmentation Architecture for Recommendation")

    parser.add_argument('--batch_size', type=int, default=10240,
                        help="Batch Size")
    parser.add_argument('--test_batch_size', type=int, default=10240,
                        help="Test Batch Size")
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help="Embedding Size")
    # final_integration: final embedding integration("MEAN"/ "ATT"/ "NONE")
    # "MEAN" means that final embedding is integrated equally with different attribute embeddings
    # "ATT" means that final embedding is integrated with attented attribute embeddings
    # "NONE" means that final embedding is replaced with the last attribute embedding
    parser.add_argument(
        '--final_integration',
        type=str,
        choices=[
            'MEAN',
            'ATT',
            'NONE'],
        default='ATT',
        help="Final Embedding Integration Method(MEAN, ATT, or NONE)")
    # att_norm: Attention Normalization Implementation("sum-ratio"/ "softmax"/ "gat-like")
    # "gat-like" means that different final embeddings are integrated with a GAT network
    # "sum-ratio" means that different final embeddings are integrated with an MLP network normalized linearly
    # "softmax" means that different final embeddings are integrated with an MLP network normalized by softmax function
    parser.add_argument(
        '--att_norm',
        type=str,
        choices=[
            'sum-ratio',
            'softmax',
            'gat-like'],
        default='softmax',
        help="Attention Normalization")
    parser.add_argument(
        '--att_hidden_dims',
        nargs='+',
        type=int,
        default=[
            128,
            64,
            16,
            1],
        help="Attention MLP Hidden Layer Dimensions")
    parser.add_argument('--num_layers', type=int, default=3,
                        help="GCN Layer Depth")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning Rate")
    parser.add_argument('--epochs', type=int, default=1000,
                        help="Max Epochs")
    parser.add_argument('--seed', type=int, default=2022,
                        help="Global Random Seed")
    parser.add_argument('--prefix', type=str, default="pdagnn-",
                        help="Model Abbreviation")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=[
            'Amazon-Book',
            'Amazon-CDs',
            'MovieLens-1M'],
        default='Amazon-Book',
        help="Dataset")
    parser.add_argument('--topK', nargs='+', type=int, default=[5, 10, 20],
                        help="Value of K for Metric@K")
    # L2 Loss
    parser.add_argument(
        '--ifRegEmbedding',
        type=bool,
        default=True,
        help="If Embedding Regularization Term ($L_2$ Loss) Added in the Loss Function")
    parser.add_argument('--weight_decay_embed', type=float, default=1e-4,
                        help="Embedding Regularization Weight")
    # Distance Loss
    parser.add_argument(
        '--ifRegBehav',
        type=bool,
        default=True,
        help="If Behavior Regularization Term (Attribute Distance Loss) Added in the Loss Function")
    parser.add_argument('--weight_decay_behavior', type=float, default=1e-4,
                        help="Behavior Regularization Weight")
    parser.add_argument('--ifLoad', type=bool, default=False,
                        help="If Model Is Loaded from File")
    parser.add_argument('--ifDropOut', type=bool, default=True,
                        help="Whether Apply Dropout")
    parser.add_argument('--keep_prob', type=float, default=0.6,
                        help="Dropout Ratio")
    parser.add_argument(
        '--load_model_name',
        type=str,
        default="pdagnn-epoch-1000.pth.tar",
        help="Loaded Model FileName")
    return parser.parse_args()


if __name__ == '__main__':
    args = argParser()
    # print args
    for key, value in args._get_kwargs():
        if value is not None:
            print("%s -\t%s" % (key, value))
