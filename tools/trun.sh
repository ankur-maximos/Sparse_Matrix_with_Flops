dir=$1
dir_o=$2
mkdir -p $dir
bin=nGpuSpMM.x
for input in cant.mtx  consph.mtx  mc2depi.mtx filter3D.mtx cop20k_A.mtx  mac_econ_fwd500.mtx  mc2depi.mtx  pdb1HYS.mtx  pwtk.mtx rma10.mtx  scircuit.mtx  webbase-1M.mtx shipsec1.mtx 2cubes_sphere.mtx cage12.mtx hood.mtx m133-b3.mtx majorbasis.mtx mario002.mtx mono_500Hz.mtx offshore.mtx patents_main.mtx poisson3Da.mtx qcd5_4.mtx; do
#  nvprof ./$bin -cuda -spgemm /home/niuq/data/matrix_input/$input |& tee $dir/nvprof-liuw-$input-re
 # ./$bin -cuda -spgemm /home/niuq/data/matrix_input/$input |& tee $dir/liuw-$input-re

#for input in cant.mtx ;
#do
   mkdir -p $dir_o/prof-output-$input-re

   echo	-e "\n\n\nGeneral summary of nvprofiler-----------------------------------------------\n\n" >> $dir/nvprof-liuw-$input-re
   
   ## generating profiler output file
   nvprof -o $dir_o/prof-output-$input-re/profile_general.out ./$bin --input=/home/niuq/data/matrix_input/$input 

   ## appending profiler output to final profiling text file 
   nvprof ./$bin --input=/home/niuq/data/matrix_input/$input |& tee -a $dir/nvprof-liuw-$input-general-re

   echo -e "\n\nEvery counter is an aggregated value of the GPU i.e values are collected across all units of the GPU" >> $dir/nvprof-liuw-$input-re

   echo -e "\nInstructions Issued , Instructions Executed, gld_request, gst_request----------------" >> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - gld_request:  Number of executed load instructions where the state space is not specified and hence generic addressing is used, increments per warp on a multiprocessor. It can include the load operations from global,local and shared state space." >>  $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - gst_request:  Number of executed store instructions where the state space is not specified and hence generic addressing is used, increments per warp on a multiprocessor. It can include the store operations to global,local and shared state space." >> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - inst_issued1:  Number of single instruction issued per cycle" >> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - inst_issued2:  Number of dual instructions issued per cycle">> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - inst_executed:  Number of instructions executed, do not include replays.">> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - l1_global_load_miss:  Number of cache lines that miss in L1 cache for global memory load accesses. In case of perfect coalescing this increments by 1,2, and 4 for 32, 64 and 128 bit accesses by a warp respectively." >> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - l1_global_load_hit:  Number of cache lines that hit in L1 cache for global memory load accesses. In case of perfect coalescing this increments by 1,2, and 4 for 32, 64 and 128 bit accesses by a warp respectively.">> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - global_store_transaction:  Number of global store transactions. Incr    ements by 1 per transaction. Transaction can be 32/64/96/128B.">> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - uncached_global_load_transaction:  Number of uncached global load transactio    ns. Increments by 1 per transaction. Transaction can be 32/64/96/128B.">> $dir/nvprof-liuw-$input-re

  echo -e "\nEvent - shared_load_replay:  Replays caused due to shared load bank conflict (when the addresses for two or more shared memory load requests fall in the same memory bank) or when there is no conflict but the total number of words accessed by all threads in the warp executing that instruction exceed the number of words that can be loaded in one cycle (256 bytes).">> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - shared_store_replay:  Replays caused due to shared store bank conflict (when the addresses for two or more shared memory store requests fall in the same memory bank) or when there is no conflict but the total number of words accessed by all threads in the warp executing that instruction exceed the number of words that can be stored in one cycle.">> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - shared_load:  Number of executed load instructions where state space is specified as shared, increments per warp on a multiprocessor." >> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent - shared_store:  Number of executed store instructions where state space is specified as shared, increments per warp on a multiprocessor." >> $dir/nvprof-liuw-$input-re 
   echo -e "\nEvent : l1_local_load_miss:  Number of cache lines that miss in L1 cache for local memory load accesses. In case of perfect coalescing this increments by 1,2, and 4 for 32, 64 and 128 bit accesses by a warp respectively.">> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent : l1_local_load_hit:  Number of cache lines that hit in L1 cache for local memory load accesses. In case of perfect coalescing this increments by 1,2, and 4 for 32, 64 and 128 bit accesses by a warp respectively.">> $dir/nvprof-liuw-$input-re

   echo -e "\nEvent : local_store:  Number of executed store instructions where state space is specified as local, increments per warp on a multiprocessor.">> $dir/nvprof-liuw-$input-re

 
   ## generating profiler output file
   nvprof --events ld_request,gst_request,inst_issued1,inst_issued2,inst_executed,l1_global_load_miss,l1_global_load_hit,global_store_transaction,uncached_global_load_transaction,shared_load_replay,shared_store_replay,shared_load,shared_store,l1_local_load_miss,l1_local_load_hit,local_store -o $dir_o/prof-output-$input-re/profile_general.out ./$bin --input=/home/niuq/data/matrix_input/$input 

   ## appending profiler output to final profiling text file 
   nvprof --events ld_request,gst_request,inst_issued1,inst_issued2,inst_executed,l1_global_load_miss,l1_global_load_hit,global_store_transaction,uncached_global_load_transaction,shared_load_replay,shared_store_replay,shared_load,shared_store,l1_local_load_miss,l1_local_load_hit,local_store ./$bin --input=/home/niuq/data/matrix_input/$input |& tee -a $dir/nvprof-liuw-$input-re

   echo -e "\n\n\n GPU Tracing with the parameters">> $dir/nvprof-liuw-$input-gputrace-re

     ## generating profiler output file
   nvprof --events ld_request,gst_request,inst_issued1,inst_issued2,inst_executed --print-gpu-trace -o $dir_o/prof-output-$input-re/profile_general.out ./$bin --input=/home/niuq/data/matrix_input/$input 

   ## appending profiler output to final profiling text file 
   nvprof --events ld_request,gst_request,inst_issued1,inst_issued2,inst_executed --print-gpu-trace ./$bin --input=/home/niuq/data/matrix_input/$input |& tee -a $dir/nvprof-liuw-$input-gputrace-re

   echo -e "\n\n\n">> $dir/nvprof-liuw-$input-gputrace-re

     ## generating profiler output file
   nvprof --events l1_global_load_miss,l1_global_load_hit,global_store_transaction,uncached_global_load_transaction --print-gpu-trace -o $dir_o/prof-output-$input-re/profile_general.out ./$bin --input=/home/niuq/data/matrix_input/$input 

   ## appending profiler output to final profiling text file 
   nvprof --events l1_global_load_miss,l1_global_load_hit,global_store_transaction,uncached_global_load_transaction --print-gpu-trace ./$bin --input=/home/niuq/data/matrix_input/$input |& tee -a $dir/nvprof-liuw-$input-gputrace-re

   echo -e "\n\n\n">> $dir/nvprof-liuw-$input-gputrace-re

  ## generating profiler output file
   nvprof --events shared_load_replay,shared_store_replay,shared_load,shared_store --print-gpu-trace -o $dir_o/prof-output-$input-re/profile_general.out ./$bin --input=/home/niuq/data/matrix_input/$input 

   ## appending profiler output to final profiling text file 
   nvprof --events shared_load_replay,shared_store_replay,shared_load,shared_store --print-gpu-trace ./$bin --input=/home/niuq/data/matrix_input/$input |& tee -a $dir/nvprof-liuw-$input-gputrace-re

   echo -e "\n\n\n">> $dir/nvprof-liuw-$input-gputrace-re
 
   ## generating profiler output file
   nvprof --events l1_local_load_miss,l1_local_load_hit,local_store --print-gpu-trace -o $dir_o/prof-output-$input-re/profile_general.out ./$bin --input=/home/niuq/data/matrix_input/$input 

   ## appending profiler output to final profiling text file 
   nvprof --events l1_local_load_miss,l1_local_load_hit,local_store --print-gpu-trace ./$bin --input=/home/niuq/data/matrix_input/$input |& tee -a $dir/nvprof-liuw-$input-gputrace-re

   ## appending output to liuw file
   #./$bin --input=/home/niuq/data/matrix_input/$input |& tee $dir/liuw-$input-re

done

# done  > $dir/nvprof-output-$input-re
