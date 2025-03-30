**HDFL** comes from "A Novel Hierarchically Decentralized Federated Learning Framework in 6G Wireless Networks". It proposes an integrated hierarchically decentralized federated learning (HDFL) framework, where devices from different cells collaboratively train a global model under periodically intra-cell D2D consensus and inter-cell aggregation. It establishes strong convergence guarantees for the proposed HDFL algorithm without assuming convex objectives. The convergence rate of HDFL can be optimized to achieve the balance of model accuracy and communication overhead. To improve the wireless performance of HDFL, we formulate an optimization problem to minimize the training latency and energy overhead.



We set two continual learning methods on HDFL. You can use the following command to run HDFL:



~~~sh
```shell
cd Baselines
cd HDFL  # method name
python mainHDFL_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

python mainHDFL_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM
~~~

