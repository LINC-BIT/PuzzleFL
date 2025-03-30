**FedWeiT** comes from "Federated Continual Learning with Weighted Inter-client Transfer". It proposes a novel federated continual learning framework, Federated Weighted Inter-client Transfer (FedWeIT), which decomposes the network weights into global federated parameters and sparse task-specific parameters, and each client receives selective knowledge from other clients by taking a weighted combination of their task-specific parameters. FedWeIT minimizes interference between incompatible tasks, and also allows positive knowledge transfer across clients during learning. 

You can use the following command to run FedWeIT:



~~~sh
```shell
cd baselines
cd FedWeIT  # method name
python main_FedWeIT.py --task_number=10 --class_number=100 --dataset=cifar100 
~~~

