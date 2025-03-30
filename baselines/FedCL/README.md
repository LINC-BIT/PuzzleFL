**FedCL** comes from "Continual Local Training For Better Initialization Of Federated Models". It proposes the local continual training strategy to address this problem. Importance weights are evaluated on a small proxy dataset on the central server and then used to constrain the local training. With this additional term, it alleviates the weight divergence and continually integrate the knowledge on different local clients into the global model, which ensures a better generalization ability.

You can use the following command to run FedCL:



~~~sh
```shell
cd baselines
cd FedCL  # method name
python main_FedCL.py --task_number=10 --class_number=100 --dataset=cifar100 
~~~

