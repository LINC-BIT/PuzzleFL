**Loci** comes from "Loci: Federated Continual Learning of Heterogeneous Tasks at Edge". it propose Loci to provide abstractions for clients’ past and peer task knowledge using compact model weights, and develop a communication-efficient approach to train each client’s local model by exchanging its tasks’ knowledge with the most accuracy relevant one from other clients. Through its general-purpose API, Loci can be used to provide efficient on-device training for existing deep learning applications of graph, image, nature language processing, and multimodal data.

You can use the following command to run Loci:



~~~sh
```shell
cd Baselines
cd Loci  # method name
python main_Loci.py --task_number=10 --class_number=100 --dataset=cifar100 
~~~

