## 使用说明

### 训练

```
python cnnMNIST_train.py
```

### 模型冻结

```
sh freeze.sh
```

### 模型预测

```
python predict.py
```

### 转换fpga需要的verilog项目

```
../../src/LeFlow cnnMNIST_to_fpga.py
```

### 模型仿真

```
cd cnnMNIST_to_fpga_files
make w
```

需要等待几个小时
输出结果可在temp10对应的ram中找到