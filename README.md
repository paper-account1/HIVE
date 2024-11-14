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
