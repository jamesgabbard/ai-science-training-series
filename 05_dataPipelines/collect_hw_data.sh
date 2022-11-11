# Set up software deps:
# qsub-gpu -I -A ALCFAITP --attrs filesystems=home,grand -t 60 -n 1 -q full-node
# module load cobalt/cobalt-gpu
module load conda/2022-07-01
conda activate
cd /home/jgabbard/Projects/ai-science-training-series/05_dataPipelines
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"

filename="homework_data.txt"

# Loop over prefetch and thread parameters
# Part 1 : plenty of resources (faster)
# for prf in 32 16 8 3 2 1
# do
#     for threads in 256 128 64 32 16 
#     do
#         python profile_resnet34.py ${filename} ${prf} ${threads}
#     done
# done

# Part 2 : way fewer threads (much slower)
# for prf in 32 16 8 3 2 1
# do
#     for threads in 8 4 2 1
#     do
#         python profile_resnet34.py ${filename} ${prf} ${threads}
#     done
# done

for prf in 1
do
    for threads in 8 4 2 1
    do
        python profile_resnet34.py ${filename} ${prf} ${threads}
    done
done