**DPFL** comes from "Like Attracts Like: Personalized Federated Learning in Decentralized Edge Computing". It takes the communication constraint and heterogeneity into consideration and proposes to realize communication-efficient DPFL with adaptive model pruning and neighbor selection. It theoretically analyzes the convergence of the proposed DPFL method, and studies the impacts of both model pruning and neighbor selection on training performance. Furthermore, we propose an efficient algorithm that combines model pruning and neighbor selection to achieve a trade-off between model quality and communication cost.



We set two continual learning methods on DPFL. You can use the following command to run DPFL:



~~~sh
```shell
cd Baselines
cd DPFL  # method name
python mainDPFL_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

python mainDPFL_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM
~~~

