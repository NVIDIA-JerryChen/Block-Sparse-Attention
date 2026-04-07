// Explicit instantiation: HasVarBlockNums=false, HasBlockSizes=false, UseCLC=false
#include "../flash_fwd_launch_template.h"

template std::vector<torch::Tensor> flash::bsa_fused_fwd_blk64_launch<false, false, false>(
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, int, torch::Tensor, float, torch::Tensor);
