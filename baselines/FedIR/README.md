**FedIR** comes from "Decentralized Federated Learning With Intermediate Results in Mobile Edge Computing". It provides theoretical analysis of DFL based on intermediate result exchanging, which reveals the relationship between the training performance and the exchanging interval (i.e., the number of local updating iterations) of intermediate results. According to the convergence bound, it proposes an adaptive exchanging interval (or frequency) algorithm called Fed-IR, which optimizes the trade-off between communication cost and training performance.



We set two continual learning methods on FedIR. You can use the following command to run FedIR:



~~~sh
```shell
cd Baselines
cd FedIR  # method name
python mainFedIR_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

python mainFedIR_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM
~~~

