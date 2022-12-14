#!/bin/sh

# 
# Vivado(TM)
# runme.sh: a Vivado-generated Runs Script for UNIX
# Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
# 

echo "This script was generated under a different operating system."
echo "Please update the PATH and LD_LIBRARY_PATH variables below, before executing this script"
exit

if [ -z "$PATH" ]; then
  PATH=F:/Xilinx/Vivado/2019.2/ids_lite/ISE/bin/nt64;F:/Xilinx/Vivado/2019.2/ids_lite/ISE/lib/nt64:F:/Xilinx/Vivado/2019.2/bin
else
  PATH=F:/Xilinx/Vivado/2019.2/ids_lite/ISE/bin/nt64;F:/Xilinx/Vivado/2019.2/ids_lite/ISE/lib/nt64:F:/Xilinx/Vivado/2019.2/bin:$PATH
fi
export PATH

if [ -z "$LD_LIBRARY_PATH" ]; then
  LD_LIBRARY_PATH=
else
  LD_LIBRARY_PATH=:$LD_LIBRARY_PATH
fi
export LD_LIBRARY_PATH

HD_PWD='E:/个人/学习/大四上/智能芯片与系统设计综合实践/Comprehensive-practice-of-intelligent-chip-system-design/FPGA-proj-master/CNN_minist/CNN_minist.runs/W_conv1_synth_1'
cd "$HD_PWD"

HD_LOG=runme.log
/bin/touch $HD_LOG

ISEStep="./ISEWrap.sh"
EAStep()
{
     $ISEStep $HD_LOG "$@" >> $HD_LOG 2>&1
     if [ $? -ne 0 ]
     then
         exit
     fi
}

EAStep vivado -log W_conv1.vds -m64 -product Vivado -mode batch -messageDb vivado.pb -notrace -source W_conv1.tcl
