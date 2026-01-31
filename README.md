# Code of HFR-NO

## Environment Installation
---
Create and activate an Anaconda Environment:
```
conda create -n lab python=3.10
conda activate lab
```
Install required packages with the following commands:
```
pip install -r requirement.txt
````
## Data Preparation
---
Download the dataset from the following links, and then unzip them in a specific directory.<br>
**Structured Mesh Problems**  
+ [Darcy Flow](https://drive.google.com/file/d/1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5/view?usp=sharing)
+ [Airfoil](https://drive.google.com/file/d/1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5/view?usp=sharing)
+ [Plasticity](https://drive.google.com/file/d/1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5/view?usp=sharing)
**Unstructured Mesh Problems**  
+ [Darcy Flow](https://drive.google.com/drive/folders/1jS7YwY1Gs7rGOm1VXrkN_KvTzxGxTw6G?usp=sharing) 
+ [PipeTurbulence](https://drive.google.com/drive/folders/1jS7YwY1Gs7rGOm1VXrkN_KvTzxGxTw6G?usp=sharing) 
+ [HeatTransfer](https://drive.google.com/drive/folders/1jS7YwY1Gs7rGOm1VXrkN_KvTzxGxTw6G?usp=sharing)
+ [Composites](https://drive.google.com/drive/folders/1jS7YwY1Gs7rGOm1VXrkN_KvTzxGxTw6G?usp=sharing)
+ [BloodFlow](https://drive.google.com/drive/folders/1jS7YwY1Gs7rGOm1VXrkN_KvTzxGxTw6G?usp=sharing)
## Experiment Running
---
Run the experiments with the following scripts. All dataset paths can be specified via the parameters (data_path).
+ Darcy Flow
  ```
  bash ./exp_scripts/darcy.sh 
  ```
+ Airfoil
  ```
  bash ./exp_scripts/darcy.sh 
  ```
+ Plasticity
  ```
  bash ./exp_scripts/darcy.sh
  ```
+ Darcy Flow
  ```
  python exp_unstructed_darcy.py
  ```
+ PipeTurbulence
  ```
  python exp_unstructed_pipe.py
  ```
+ HeatTransfer
  ```
  python exp_heat.py
  ```
+ Composites
  ```
  python composites.py
  ```
+ BloodFlow
  ```
  python exp_blood.py
  ```
## Acknowledge
We thank the following open-sourced projects, which provide the basis of this work.
+ [https://github.com/yuexihang/HPM](https://github.com/yuexihang/HPM)
+ [https://github.com/neuraloperator/Geo-FNO](https://github.com/neuraloperator/Geo-FNO)
+ [https://github.com/neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)
+ [https://github.com/gengxiangc/NORM](https://github.com/gengxiangc/NORM)
+ [https://github.com/thuml/Transolver](https://github.com/thuml/Transolver)
+ [https://github.com/nmwsharp/nonmanifold-laplacian](https://github.com/nmwsharp/nonmanifold-laplacian)
