**Cross-FCL** comes from "Cross-FCL: Toward a Cross-Edge Federated Continual Learning Framework in Mobile Edge Computing Systems". it enables devices to retain the knowledge learned in the past when participating in new task training through a parameter decomposition based FCL model. Then various cross-edge strategies are introduced, including biased global aggregation and local optimization, to trade off memory and adaptation.

You can use the following command to run Cross-FCL:



~~~sh
```shell
cd baselines
cd Cross-FCL  # method name
python main_Cross.py --task_number=10 --class_number=100 --dataset=cifar100 
~~~

