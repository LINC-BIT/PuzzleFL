# from p2pfl.node import Node
# from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
# from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
# import sys
# import random
# import torch
# if __name__ == "__main__":
#     node = Node(
#         MLP(),
#         MnistFederatedDM(sub_id=0, number_sub=2),
#         port=9001,
#     )
#     node.start()
#     input("Press any key to stop\n")


import time

print(str(int(time.time()))[-4:])
