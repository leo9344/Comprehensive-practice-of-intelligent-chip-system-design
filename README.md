# Comprehensive-practice-of-intelligent-chip-system-design
COPY: Comprehensive practice Of intelligent chiP sYstem design

**Caution: The project still needs maintenance and some**

# Overview

We trained, implemented and deployed an FPGA-based Convolutional Neural Network (CNN) in Verilog and Python.

# Model

![fig_cnn](https://github.com/leo9344/Comprehensive-practice-of-intelligent-chip-system-design/blob/main/fig_cnn.png)

## Training

```
python model.py
```

## Get information (FLOPs, Params, Mem Size) of our model

Thanks to torchsummary and thop, we can easily get model information by a few lines of code.

**Uncomment the code on line 129-133 in `model.py`** then run it.

## Quantization

Use `utils.py` to get **Q8.8** result of our model

```
python utils.py --model /path/to/your/model --ori_path /path/to/save/original/results --q_path /path/to/quantized/results
```

Ex:

```
python utils.py --model FP16+Aug_Acc0.995_Epoch18.pth --ori_path ./layers/original/ --q_path ./layers/quantized/
```

**Notice**

Original model will be saved in `--ori_path` while quantized model will be saved in `--q_path`

After you get the result of quantization, copy them to corresponding path of `FPGA-proj-master`

## Visualization

You can use `visualizer.py` to make comparisons between your models.

# RTL Design



# Simulation



# IP Package



# Field Test