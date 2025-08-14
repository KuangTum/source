#!/bin/bash
#SBATCH --account=sipv
#SBATCH --job-name=dftompx        # 作业名称
#SBATCH --output=optica%j.out       # 作业输出文件
#SBATCH --error=banddft%j.err        # 作业错误文件
#SBATCH --partition=debug #standard #standard #debug#short#ihighmem#gpu#shared
#SBATCH --time=1:00:00
#SBATCH --nodes=1                    # 需要的节点数量
#SBATCH --ntasks-per-node=1           # 每个节点的任务数量
#SBATCH --cpus-per-task=52 #totol 128
## 加载MPI模块
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load intel-oneapi
module load intel-oneapi-compilers
module load intel-oneapi-mkl
module load intel-oneapi-mpi
module list
# 运行MPI并行Python脚本
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nopt/nrel/apps/cpu_stack/libraries-intel/06-24/linux-rhel8-sapphirerapids/oneapi-2024.1.0/hdf5-1.14.3-i7q2i4oub3vqppdanjvy6flexieguxbk/lib
echo $SLURM_CPUS_PER_TASK
#cd $SLURM_SUBMIT_DIR/../work_dir/dataset/raw/${SLURM_ARRAY_TASK_ID}
echo $SLURM_SUBMIT_DIR
echo $SLURM_NTASKS
#cd "/kfs3/scratch/xkuang/dataset/MoS2/olp/120du_30A_5_5"
mpirun -np $SLURM_NTASKS /projects/sipv/xkuang/softwares/ompi_openmx/openmx3.9/source/openmx Cubulk.dat > openmx_scf.std
#mpirun -np $SLURM_NTASKS /projects/sipv/xkuang/softwares/Openmx/openmx3.9/source/openmx Cubulk.dat > originopenmx_scf.std
wait
# #bandgnu13 openmx.Band
# #get fermi energy
rm -r openmx_rst
# wait
# grep -m1 -i "Chemical Potential (Hartree)" openmx.out | awk '{print $NF*27.211386}'
#wait
#mpirun -np 1 analysis_example openmx_olpr.scfout > originHS.out
