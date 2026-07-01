// Explicit instantiation: kHeadDim=128, bf16, SM100, HasBlockSizes=false, HasVarBlockNums=false, UseClc=true
#include "../bsa_fwd_launch_template.h"

template void flash::run_bsa_fwd<128, false, false, true, false>(bsa_fwd_params const&, cudaStream_t);
