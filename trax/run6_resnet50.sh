#sh periodical_send.sh &
export FOLDER="resnet50_op_fusion_level_"$OP_FUSION"_tensor_fusion_threshold_"$TENSOR_FUSION_THRESHOLD
GIN_FILE=resnet50.gin
export PROC_ID=0
export PROC_NUM=6
export CUDA_VISIBLE_DEVICES=0,1
export TF_XLA_FLAGS="--tf_xla_max_cluster_size=1000000000 --tf_xla_auto_jit=2"
#export XLA_FLAGS="--xla_hlo_profile --xla_dump_to=hlo_module/$PROC_NUM/$FOLDER --xla_dump_hlo_as_proto --xla_dump_hlo_pass_re="all_reduce_combiner"" 
export XLA_FLAGS="--xla_dump_to=hlo_module/$PROC_NUM/$FOLDER --xla_dump_hlo_as_proto --xla_dump_hlo_pass_re="all_reduce_combiner"" 
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MIN_VLOG_LEVEL=0
#export TF_CPP_VMODULE="nccl_all_reduce_thunk=5"
if [ $1 = "profile" ]; then
  nsys nvprof -f -o hlo_module/$PROC_NUM/$FOLDER/$FOLDER --profile-from-start off python trainer.py --config_file=$GIN_FILE
elif [ $1 = "debug" ]; then
  gdb -ex r --args python trainer.py --config_file=$GIN_FILE
else
  python trainer.py --config_file=$GIN_FILE
fi
