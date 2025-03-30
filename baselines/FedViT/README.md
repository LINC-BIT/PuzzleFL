**FedViT** comes from "FedViT: Federated continual learning of vision transformer at edge". It is a client-side solution that continuously extracts and integrates the knowledge of signature tasks which are highly influenced by the current task. Each client of FedViT is composed of a knowledge extractor, a gradient restorer and, most importantly, a gradient integrator. Upon training for a new task, the gradient integrator ensures the prevention of catastrophic forgetting and mitigation of negative knowledge transfer by effectively combining signature tasks identified from the past local tasks and other clients’ current tasks through the global model.

You can use the following command to run FedViT:



~~~sh
```shell
cd baselines
cd FedViT  # method name
python main_FedViT.py --task_number=10 --class_number=100 --dataset=cifar100 
~~~

