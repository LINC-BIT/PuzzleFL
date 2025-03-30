**TFCL** comes from "Traceable Federated Continual Learning". It develops TagFed a framework that enables accurate and effective Tracing augmentation and Federation for TFCL. The key idea is to decompose the whole model into a series of marked sub-models for optimizing each client task before conducting group-wise knowledge aggregation such that the repetitive tasks can be located precisely and federated selectively for improved performance.

You can use the following command to run TFCL:



~~~sh
```shell
cd baselines
cd TFCL  # method name
python main_TFCL.py --task_number=10 --class_number=100 --dataset=cifar100 
~~~

