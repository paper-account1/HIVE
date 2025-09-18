## Step1 
Please ensure you have configured the [accel-sim](https://github.com/accel-sim/accel-sim-framework) simulator running environment and successfully run the PTX test example.

To build the Accel-Sim simulator that uses the PTX mode, do the following:
```
source ./gpu-app-collection/src/setup_environment
source ./gpu-simulator/setup_environment.sh
make -j -C ./gpu-simulator/
./util/job_launching/run_simulations.py -B <benchmark> -C <gpu_config> -N <run_identifier>
# for example: ./util/job_launching/run_simulations.py -B polybench -C QV100-PTX -N myTest-PTX
```


## Step2
Replace the contents of the `gpu-simulator/gpgpu-sim/src/` directory with the contents of this repository and rerun the command in **step 1**. 

At this point, you will get the results of the modified architecture.

## Citation
If you find HIVE useful for your own work, please cite our paper as follows.
```
@INPROCEEDINGS{11133338,
  author={Tang, Yuhan and Zhang, Jianmin and Ma, Sheng and Li, Tiejun and Li, Hanqing and Luo, Shengbai and Tang, Jixuan and Wu, Lizhou},
  booktitle={2025 62nd ACM/IEEE Design Automation Conference (DAC)}, 
  title={HIVE: A High-Priority Victim Cache for Accelerating GPU Memory Accesses}, 
  year={2025},
  volume={},
  number={},
  pages={1-7},
  keywords={Design automation;Graphics processing units},
  doi={10.1109/DAC63849.2025.11133338}}

```
