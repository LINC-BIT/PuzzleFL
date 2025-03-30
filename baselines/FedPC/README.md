**FedPC** comes from "Peer-to-Peer Federated Continual Learning for Naturalistic Driving Action Recognition". It proposes a novel peer-to-peer (P2P) federated learning (FL) framework with continual learning, namely FedPC, which ensures privacy and enhances learning efficiency while reducing communication, computational, and storage overheads. The framework focuses on addressing the clients' objectives within a serverless FL framework, with the goal of delivering personalized and accurate NDAR models.



We set two continual learning methods on FedPC. You can use the following command to run FedPC:



~~~sh
```shell
cd Baselines
cd FedPC  # method name
python mainFedPC_EWC.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy EWC

python mainFedPC_GEM.py --task_number=10 --class_number=100 --dataset=cifar100 ## deploy GEM
~~~

