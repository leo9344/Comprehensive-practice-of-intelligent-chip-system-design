from ast import parse
import torch
from model import FPGANet
import numpy as np
import argparse
import os

def padorcut(num, is_frac, max_width=8):
    if is_frac == True:
        if len(num)>max_width:
            num = num[:max_width]
        else:
            num += "0"*(max_width-len(num))
    else:
        if len(num)>max_width:
            num = num[-max_width:]
        else:
            num = "0"*(max_width-len(num)) + num

    return num


def dectbin(num, max_digit=8):
    sig_flag = "0"
    if num < 0:
        sig_flag = "1"
    num = abs(num)
    # 判断是否为浮点数
    if num == int(num):
        # 若为整数
        integer = '{:b}'.format(int(num))
        return sig_flag + padorcut(integer, False) + padorcut("0", True)
    else:
        # 若为浮点数
        # 取整数部分
        integer_part = int(num)
        # 取小数部分
        decimal_part = num - integer_part
        # 整数部分进制转换
        integercom = '{:b}'.format(integer_part)  #{:b}.foemat中b是二进制
        # 小数部分进制转换
        tem = decimal_part
        tmpflo = []
        # for i in range(accuracy):
        A = True
        while A and len(tmpflo) <max_digit:
            tem *= 2
            tmpflo += str(int(tem))  #若整数部分为0则二进制部分为0，若整数部分为1则二进制部分为1 #将1或0放入列表
            if tem > 1 :   #若乘以2后为大于1的小数，则要减去整数部分
                tem -= int(tem)
            elif tem < 1:  #乘以2后若仍为小于1的小数，则继续使用这个数乘2变换进制
                pass
            else:    #当乘以2后正好为1，则进制变换停止
                break
        flocom = tmpflo
        return sig_flag + padorcut(integercom, False) + padorcut(''.join(flocom), True)

parser = argparse.ArgumentParser(description="Util for extracting parameters from your model and quantizing.")
parser.add_argument("--model", help="/path/to/your/model", default="FP16+Aug_Acc0.995_Epoch18.pth")
parser.add_argument("--ori_path", help="/path/to/save/original/results", default="./layers/original/")
parser.add_argument("--q_path", help="/path/to/quantized/results", default="./layers/quantized/")

if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.model
    original_path = args.ori_path
    quantized_path = args.q_path

    model = torch.load(model_path)
    print(f"--- Loading model from {model_path} ---")
    state_dict = model.state_dict()
    names = list(state_dict.keys())

    print(f"--- Saving original model to {original_path} ---")
    for i in range(len(names)):
        name = names[i]
        param = state_dict[name]
        name = name.replace(".", "_")
        param = param.cpu().view(-1).numpy()
        np.savetxt(os.path.join(original_path, f"{name}.txt"), param)

    print(f"--- Saving quantized model to {quantized_path} ---")
    for i in range(len(names)):
        name = names[i]
        param = state_dict[name]
        name = name.replace(".", "_")
        param = param.cpu().view(-1).numpy()
        bin_list = []
        for p in param:
            bin_list.append(dectbin(p))
        np.savetxt(os.path.join(quantized_path, f"{name}.txt"), bin_list, fmt="%s")


