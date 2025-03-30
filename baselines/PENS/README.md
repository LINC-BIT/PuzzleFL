**PENS** comes from "Decentralized federated learning of deep neural networks on non-iid data". It proposes a method named Performance-Based Neighbor Selection (PENS) where clients with similar data distributions detect each other and cooperate by evaluating their training losses on each other's data to learn a model suitable for the local data distribution. 



We set two continual learning methods on PENS. You can use the following command to run PENS:



~~~sh
```shell
cd Baselines
cd PENS  # method name
python mainPen_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

python mainPen_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM
~~~

