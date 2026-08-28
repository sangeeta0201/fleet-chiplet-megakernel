/* Copyright 2023-2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "mirage/kernel/task_register.h"
#include "mirage/kernel/operator.h"
#include "mirage/transpiler/utils.h"

// std::getenv / std::atoi, for re-reading MPK_WORKER_LDS_KB when sizing the
// attention kernel's LDS budget (see the cached_lds_limit lambda below).
#include <cstdlib>

#ifdef MIRAGE_BACKEND_USE_ROCM
// Forward declaration of mirage::utils::get_max_shared_mem(); the full
// header (rocm_helper.h) drags in device-only templates that don't compile
// in this host-only translation unit.
namespace mirage {
namespace utils {
size_t get_max_shared_mem();
}
} // namespace mirage
#endif

namespace mirage {
namespace runtime {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

TaskRegister *TaskRegister::singleton = nullptr;

TaskRegister::TaskRegister() {}

TaskRegister *TaskRegister::get_instance() {
  if (singleton == nullptr) {
    singleton = new TaskRegister();
  }
  return singleton;
}

int TaskRegister::register_task_variant(runtime::TaskType type,
                                        std::string const &code) {
  std::vector<std::string> &variants = all_task_variants[type];
  for (size_t i = 0; i < variants.size(); i++) {
    if (variants[i] == code) {
      return (int)(i);
    }
  }
  // Add a new variant
  variants.push_back(code);
  return (int)(variants.size() - 1);
}

int TaskRegister::register_embedding_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params) {
  assert(params.size() == 1);
  // params[0]: input source (0: tokens, 1: input_token)
  int batch_size = 0, output_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::embedding_kernel<bfloat16, $, $, $>(",
         batch_size,
         output_size,
         output_stride);
  if (params[0] == 0) {
    code.e("    runtime_config.tokens + runtime_config.step[0], ");
  } else if (params[0] == 1) {
    code.e("    task_desc->input_ptrs[0],");
  }
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_EMBEDDING, code.to_string());
}

int TaskRegister::register_rmsnorm_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params) {
  assert(params.size() <= 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int hidden_dim = output_ops[0]->output_tensors[0].dim[1];
  // actual_hidden_dim: if provided and > 0, use for RMS mean computation
  // (avoids bf16 rounding errors when padding hidden dim)
  int actual_hidden_dim =
      (params.size() > 0 && params[0] > 0) ? params[0] : hidden_dim;
  // Currently assume that each rmsnorm task processes one token
  assert(batch_size == 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
  assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::rms_norm_impl<bfloat16, $, $, $>(",
         batch_size,
         hidden_dim,
         actual_hidden_dim);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    1e-5f);");
  return register_task_variant(TASK_RMS_NORM, code.to_string());
}

int TaskRegister::register_rmsnorm_linear_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::norm_linear_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_RMS_NORM_LINEAR, code.to_string());
}

int TaskRegister::register_attention_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  assert(params.size() == 4);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::single_batch_decoding_kernel<bfloat16, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         head_dim,
         kv_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.step[0] + 1,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_ATTENTION_1, code.to_string());
}

int TaskRegister::register_paged_attention_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  assert(params.size() == 6);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  int max_tokens = input_ops[0]->dtensor.dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_task_impl<bfloat16, $, $, $, $, "
         "$, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         kv_stride,
         qkv_stride,
         output_size,
         head_dim,
         max_seq_len,
         page_size,
         max_tokens);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_PAGED_ATTENTION_1, code.to_string());
}

int TaskRegister::register_single_batch_extend_attention_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: extend_num
  // params[5]: output_stride
  assert(params.size() == 6);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int extend_num = params[4];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int output_stride = params[5];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::single_batch_extend_kernel<bfloat16, $, $, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         head_dim,
         kv_stride,
         output_stride,
         extend_num);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.step[0] + 1,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_SINGLE_BATCH_EXTEND_ATTENTION,
                               code.to_string());
}

int TaskRegister::register_silu_mul_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, input_stride, output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[1] == output_size * 2);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[1];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_SILU_MUL, code.to_string());
}

int TaskRegister::register_identity_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Both input and output tensors should be row major
  assert(input_ops[0]->dtensor.layout == layout::DmemRowMajor);
  assert(output_ops[0]->dtensor.layout == layout::DmemRowMajor);
  // Both input and output tensors should be INPUT OP
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  // Shape should be guranteed by higher-level APIs

  int outer_dim_size = 1, inner_dim_size, outer_dim_stride, output_size;
  for (int i = 0; i < input_ops[0]->dtensor.num_dims - 1; i++) {
    outer_dim_size *= input_ops[0]->dtensor.dim[i];
  }
  inner_dim_size =
      input_ops[0]->dtensor.dim[input_ops[0]->dtensor.num_dims - 1];
  outer_dim_stride = inner_dim_size;
  output_size = output_ops[0]
                    ->output_tensors[0]
                    .dim[output_ops[0]->output_tensors[0].num_dims - 1];
  // assert(output_size >= bgraph.block_dim.x);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::identity_task_impl<bfloat16, $, $, $, $>(",
         outer_dim_size,
         inner_dim_size,
         outer_dim_stride,
         output_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_IDENTITY, code.to_string());
}

int TaskRegister::register_silu_mul_linear_with_residual_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1] / 2;
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_linear_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.my_gpu_id == 0);");
  return register_task_variant(TASK_SILU_MUL_LINEAR_WITH_RESIDUAL,
                               code.to_string());
}

int TaskRegister::register_linear_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params,
                                       bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  // When batch_size > 16 and output_size ≤ 64, use small tile (16x64x256)
  // to avoid 50% N-dimension waste from NPerBlock=128 > output_size=64.
  bool force_small_tile = (batch_size > 16 && output_size <= 64);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  if (force_small_tile) {
    code.e("kernel::linear_kernel<bfloat16, $, $, true>(",
           batch_size,
           reduction_size);
  } else {
    code.e(
        "kernel::linear_kernel<bfloat16, $, $>(", batch_size, reduction_size);
  }
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  if (with_residual) {
    code.e("    task_desc->input_ptrs[2],");
  } else {
    code.e("    nullptr,");
  }
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  if (with_residual) {
    code.e("    runtime_config.my_gpu_id == 0,");
  } else {
    code.e("    false/*residual*/,");
  }
  code.e("    $, $);", output_size, output_stride);
  if (with_residual) {
    return register_task_variant(TASK_LINEAR_WITH_RESIDUAL, code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR, code.to_string());
  }
}

int TaskRegister::register_splitk_linear_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2; // input + weight (no separate residual)
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::splitk_linear_kernel<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_SPLITK_LINEAR_MI300, code.to_string());
}

// Gang linear with HipKittens Algorithm 1 windowed traversal.
// params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
//          n_tiles_per_xcd, wgm]
// Cross-XCD K-split GEMM: each XCD handles K/8, ALL N-tiles
int TaskRegister::register_gang_ksplit_gemm_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int ws_stride = params[0];
  int tile_n = params[1];
  int n_tiles = params[2];
  int k_splits = params[3];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < 2) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int reduction_size = input_ops[0]->dtensor.dim[1];
  int batch_size = input_ops[0]->dtensor.dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_ksplit_gemm_kernel<bfloat16, $, $, $>(",
         batch_size,
         reduction_size,
         k_splits);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", ws_stride);
  code.e("    $,", n_tiles);
  code.e("    (int)task_desc->task_metadata.n_tile_start,");
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_KSPLIT_GEMM_MI300, code.to_string());
}

// Cross-XCD K-split finalize: add residual + convert bf16
int TaskRegister::register_gang_ksplit_finalize_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 3);
  int o_stride = params[0];
  int n_cols_per_xcd = params[1];
  int finalize_tiles = params[2];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < 2) { // 2 inputs: workspace, residual
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops.size() == 1);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_ksplit_finalize_kernel<bfloat16, $>(", batch_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $,", o_stride);
  code.e("    $,", o_stride);
  code.e("    $,", n_cols_per_xcd);
  code.e("    (int)task_desc->task_metadata.n_tile_start,");
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_KSPLIT_FINALIZE_MI300,
                               code.to_string());
}

// Gang split-K linear with residual: splits K dimension within XCD for
// better worker utilization. Uses XCD-local atomics for merge.
int TaskRegister::register_gang_splitk_linear_res_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 5);
  int output_stride = params[0];
  int tile_n = params[1];
  int n_tiles_per_xcd = params[2];
  int k_splits = params[3];
  // 0 = derive the reduction from the input tensor's width, as before. A
  // non-zero value stops the reduction short of it, which is what lets this
  // task see a de-padded weight: the absorbed o_proj's input carries
  // num_heads_pad * kv_lora columns but only the leading num_heads * kv_lora
  // of them have non-zero weight rows behind them.
  int reduction_override = params[4];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4; // input, weight, residual, workspace+done_counter
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  if (reduction_override > 0) {
    assert(reduction_override <= reduction_size);
    reduction_size = reduction_override;
  }
  assert(reduction_size % k_splits == 0);
  int m_per_tile = input_ops[0]->dtensor.dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Done counters stored after the float32 workspace data.
  // Workspace is [batch, chunk_n] float32 partitioned per XCD.
  // Done counters: n_tiles_per_xcd int32 values after the float data.
  int ws_floats = m_per_tile * n_tiles_per_xcd * tile_n;
  code.e("kernel::gang_splitk_linear_res_kernel<bfloat16, $, $, $>(",
         m_per_tile,
         reduction_size,
         k_splits);
  code.e("    task_desc->input_ptrs[0],");  // input
  code.e("    task_desc->input_ptrs[1],");  // weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // workspace (float32)
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    (int*)((float*)task_desc->input_ptrs[3] + $),", ws_floats);
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_SPLITK_LINEAR_RES_MI300,
                               code.to_string());
}

// Gang RMSNorm: 8 tasks (1 per XCD), each computes the same RMSNorm
// redundantly. Enables XCD-local event counting to avoid cross-XCD barrier
// overhead.
int TaskRegister::register_gang_rmsnorm_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // input, weight, and optionally a chain_after edge the kernel never reads.
  int num_inputs = (int)bgraph.operators.size() - 1;
  int num_outputs = 1; // output
  assert(num_inputs == 2 || num_inputs == 3);

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int hidden_dim = input_ops[0]->dtensor.dim[1];
  int batch_size = input_ops[0]->dtensor.dim[0];
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
  assert(output_ops[0]->dtensor.dim[1] == hidden_dim);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Gang RMSNorm: only rank 0 (tile_idx == 0) on each XCD computes.
  // Other workers skip (they see the result via XCD-local L2).
  code.e("if (tile_idx == 0) {");
  // One call per row. rms_norm_impl static_asserts BATCH_SIZE == 1, so the
  // row loop has to live here -- without it a multi-row batch normalises row
  // 0 and silently leaves the rest holding whatever the buffer had, which is
  // the shape the MTP draft layer's hnorm runs in. At batch 1 this unrolls to
  // exactly the single call it always was.
  code.e("  for (int _b = 0; _b < $; _b++) {", batch_size);
  code.e("    kernel::rms_norm_impl<bfloat16, 1, $>(", hidden_dim);
  code.e("        (bfloat16 const *)task_desc->input_ptrs[0] + _b * $,",
         hidden_dim);                            // input row
  code.e("        task_desc->input_ptrs[1],");    // weight
  code.e("        (bfloat16 *)task_desc->output_ptrs[0] + _b * $,",
         hidden_dim);                            // output row
  code.e("        1e-6f);");
  // rms_norm_impl's last __syncthreads() precedes the read of reduce_smem[0],
  // so the next row's write to that same slot has to be fenced behind every
  // thread's read of it.
  code.e("    __syncthreads();");
  code.e("  }");
  code.e("}");
  return register_task_variant(TASK_GANG_RMS_NORM_MI300, code.to_string());
}

int TaskRegister::register_gang_linear_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // BATCH_SIZE = m_per_tile (small M-tile, not full batch)
  code.e("kernel::gang_linear_kernel<bfloat16, $, $>(",
         m_per_tile,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");  // full activation
  code.e("    task_desc->input_ptrs[1],");  // XCD's weight chunk
  code.e("    task_desc->output_ptrs[0],"); // XCD's output columns
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_LINEAR_MI300, code.to_string());
}

// Gang linear with residual + HipKittens Algorithm 1 windowed traversal.
// params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
//          n_tiles_per_xcd, wgm]
int TaskRegister::register_gang_linear_res_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 10);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];
  // Optional override; 0 means "reduce over the whole input row". A caller
  // passes this when the input is deliberately wider than the weight, so the
  // GEMM can stop short of a padded tail instead of multiplying by zeros.
  int reduction_override = params[7];
  // 0 = CK MFMA path. Non-zero swaps in the narrow-tile GEMV, which is the
  // same op with the N>=64 tile constraint lifted, so that a hidden-width
  // projection can spread over more than 32 of the 240 workers.
  int gemv_rows = params[8];
  // 0 = bf16 weight. 1 = MXFP8, i.e. the weight arrives workgroup-packed as
  // [n_tiles, gemv_rows * (K + K/32)] bytes instead of [N, K] bf16. GEMV-only:
  // the CK tile reads a plain row-major weight.
  int mxfp8 = params[9];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // 3 inputs: input, weight, residual; 1 output
  assert(bgraph.operators.size() == 4);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < 3) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = reduction_override > 0 ? reduction_override
                                          : input_ops[0]->dtensor.dim[1];
  assert(reduction_size <= input_ops[0]->dtensor.dim[1]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  if (gemv_rows > 0 && mxfp8) {
    assert(gemv_rows == tile_n &&
           "gemv_rows is the compile-time form of tile_n; they must agree");
    // The packing erases the logical N, so the only cross-check left is the
    // workgroup stride: data bytes plus one E8M0 per 32 K.
    assert(input_ops[1]->dtensor.dim[1] ==
               gemv_rows * (reduction_size + reduction_size / 32) &&
           "MXFP8 weight is not packed at this reduction and row count");
    // The last argument is INPUT_ROW_STRIDE: `reduction_override` lets the
    // GEMM stop short of a padded tail, so the reduction is not the input
    // row width and the second token's row does not sit one reduction in.
    code.e("kernel::gang_gemv_mxfp8_kernel<$, $, $, true, false, $>(",
           m_per_tile,
           reduction_size,
           gemv_rows,
           input_ops[0]->dtensor.dim[1]);
  } else if (gemv_rows > 0) {
    assert(gemv_rows == tile_n &&
           "gemv_rows is the compile-time form of tile_n; they must agree");
    assert(input_ops[1]->dtensor.dim[1] == reduction_size &&
           "the GEMV indexes weight rows at stride REDUCTION_SIZE, so a "
           "narrowed reduction needs a correspondingly narrow weight");
    code.e("kernel::gang_gemv_kernel<bfloat16, $, $, $, true>(",
           m_per_tile,
           reduction_size,
           gemv_rows);
  } else {
    assert(!mxfp8 && "MXFP8 gang linear+residual has no CK MFMA path");
    code.e("kernel::gang_linear_residual_kernel<bfloat16, $, $>(",
           m_per_tile,
           reduction_size);
  }
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_LINEAR_RES_MI300, code.to_string());
}

// Gang linear with fused bias_add into epilogue.
// params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
//          n_tiles_per_xcd, wgm]
// Inputs: [activation(replicate), weight(partition dim 0), bias(replicate)]
// Outputs: [output(partition dim 1)]
int TaskRegister::register_gang_linear_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3; // input, weight, bias
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_kernel<bfloat16, $, $>(",
         m_per_tile,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");  // activation
  code.e("    task_desc->input_ptrs[1],");  // weight
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx,");
  code.e("    task_desc->input_ptrs[2]);"); // bias
  return register_task_variant(TASK_GANG_LINEAR_BIAS_MI300, code.to_string());
}

// Gang split-K linear with residual + fused bias_add into epilogue.
// params: [output_stride, tile_n, n_tiles_per_xcd, k_splits]
// Inputs: [input, weight, residual, workspace, bias]
// Outputs: [output]
int TaskRegister::register_gang_splitk_linear_res_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int output_stride = params[0];
  int tile_n = params[1];
  int n_tiles_per_xcd = params[2];
  int k_splits = params[3];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5; // input, weight, residual, workspace, bias
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  int m_per_tile = input_ops[0]->dtensor.dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  int ws_floats = m_per_tile * n_tiles_per_xcd * tile_n;
  code.e("kernel::gang_splitk_linear_res_kernel<bfloat16, $, $, $>(",
         m_per_tile,
         reduction_size,
         k_splits);
  code.e("    task_desc->input_ptrs[0],");  // input
  code.e("    task_desc->input_ptrs[1],");  // weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // workspace (float32)
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    (int*)((float*)task_desc->input_ptrs[3] + $),", ws_floats);
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    tile_idx,");
  code.e("    task_desc->input_ptrs[4]);"); // bias
  return register_task_variant(TASK_GANG_SPLITK_LINEAR_RES_BIAS_MI300,
                               code.to_string());
}

// Fused RMSNorm + Gang Linear + Bias.
// params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
//          n_tiles_per_xcd, wgm, actual_hidden_dim]
// Inputs: [norm_input, norm_weight, norm_output_scratch, linear_weight, bias]
// Outputs: [linear_output]
int TaskRegister::register_gang_rmsnorm_linear_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() >= 8 && params.size() <= 10);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];
  int actual_hidden_dim = params[7];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs =
      5; // norm_input, norm_weight, norm_output, linear_weight, bias
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is norm_input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int reduction_size = input_ops[0]->dtensor.dim[1];
  // Optional 9th param: the leading span of the row the RMS sum runs over.
  // Defaults to the whole row; GLM's q_a_layernorm passes the padded q_lora
  // width because its input row is the fused [q_a | kv_latent] projection.
  int norm_span = params.size() >= 9 ? params[8] : reduction_size;
  assert(norm_span > 0 && norm_span <= reduction_size);
  assert(actual_hidden_dim <= norm_span);
  // Optional 10th param: shorten the reduction itself, dropping a zero-padded
  // tail from the GEMM instead of multiplying the weight against it. The
  // extent doubles as the row stride, so this is single-row only.
  if (params.size() == 10) {
    assert(params[9] > 0 && params[9] <= reduction_size);
    assert(norm_span <= params[9]);
    assert(m_per_tile == 1);
    reduction_size = params[9];
  }

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_bias_kernel<bfloat16, $, $, $, $>(",
         m_per_tile,
         reduction_size,
         actual_hidden_dim,
         norm_span);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch (writable)
  code.e("    task_desc->input_ptrs[3],");  // linear_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // linear_output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_BIAS_MI300,
                               code.to_string());
}

// Fused RMSNorm + Gang Linear + Bias + MLA KV cache update.
//
// Same shape as the plain variant above -- q_a_layernorm feeding the absorbed
// q_b_proj -- with what used to be the MLA_KV_CACHE_UPDATE task folded into
// its epilogue. `norm_span` and the narrowed reduction are mandatory here
// rather than optional; GLM's fused [q_a | kv_latent] projection is the only
// caller and always supplies both.
//
// kv_latent is not an input of its own: the latent is the tail of the very
// row this GEMM norms, so input_ptrs[0] serves both, read at kv_input_offset
// against the *tensor's* full row width rather than the narrowed reduction.
int TaskRegister::register_gang_rmsnorm_linear_bias_mla_kvupd_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 15);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];
  int actual_hidden_dim = params[7];
  int norm_span = params[8];
  int reduction_size = params[9];
  int kv_lora_rank = params[10];
  int qk_rope_head_dim = params[11];
  int kv_input_offset = params[12];
  int max_seq_len = params[13];
  int page_size = params[14];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // norm_input, norm_weight, norm_output, linear_weight, bias,
  // kv_a_layernorm weight, cos, sin, paged latent cache
  int num_inputs = 9;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  assert(input_ops[0]->dtensor.num_dims == 2);
  int kv_input_stride = input_ops[0]->dtensor.dim[1];
  assert(norm_span > 0 && norm_span <= reduction_size);
  assert(actual_hidden_dim <= norm_span);
  assert(reduction_size <= kv_input_stride);
  assert(m_per_tile == 1 &&
         "the narrowed reduction doubles as the row stride; needs one row");
  assert(kv_input_offset + kv_lora_rank + qk_rope_head_dim <= kv_input_stride);
  // As in the standalone task: one shared latent head, so the cache row
  // stride is just the last dim.
  int kv_cache_stride = input_ops[8]->output_tensors[0].dim[3];
  assert(kv_cache_stride >= kv_lora_rank + qk_rope_head_dim);
  // A head is (kv_lora_rank + qk_rope_head_dim) wide and its rope slice must
  // land in exactly one tile, or the in-place rotation would straddle
  // workgroups.
  assert(tile_n == qk_rope_head_dim &&
         (kv_lora_rank + qk_rope_head_dim) % tile_n == 0);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_bias_mla_kvupd_kernel<bfloat16, $, $, $, "
         "$, $, $, $, $, $, $, $>(",
         m_per_tile,        /* BATCH_SIZE */
         reduction_size,    /* REDUCTION_SIZE */
         actual_hidden_dim, /* ACTUAL_HIDDEN_DIM */
         norm_span,         /* NORM_SPAN */
         kv_lora_rank,      /* KV_LORA_RANK */
         qk_rope_head_dim,  /* QK_ROPE_HEAD_DIM */
         kv_input_stride,   /* KV_INPUT_STRIDE */
         kv_cache_stride,   /* KV_CACHE_STRIDE */
         max_seq_len,       /* MAX_SEQ_LEN */
         page_size,         /* PAGE_SIZE */
         kv_input_offset);  /* KV_INPUT_OFFSET */
  code.e("    task_desc->input_ptrs[0],");  // norm_input, also kv_latent
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch (writable)
  code.e("    task_desc->input_ptrs[3],");  // linear_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->input_ptrs[0],");  // kv_latent
  code.e("    task_desc->input_ptrs[5],");  // kv_a_layernorm weight
  code.e("    task_desc->input_ptrs[6],");  // cos
  code.e("    task_desc->input_ptrs[7],");  // sin
  code.e("    task_desc->output_ptrs[0],"); // q_workspace
  code.e("    task_desc->input_ptrs[8],");  // paged latent cache, written
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx,");
  // Matches the standalone task's epsilon, so this stays a pure scheduling
  // change.
  code.e("    1e-6f);");
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_BIAS_MI300,
                               code.to_string());
}

// Fused RMSNorm + Gang Linear + Bias + TopK Softmax.
// params: [output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd,
//          n_tiles_per_xcd, wgm, actual_hidden_dim, num_experts,
//          num_experts_per_tok, total_gang_tiles]
int TaskRegister::register_gang_rmsnorm_linear_bias_topk_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 11);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];
  int actual_hidden_dim = params[7];
  int num_experts = params[8];
  int num_experts_per_tok = params[9];
  int total_gang_tiles = params[10];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;  // norm_input, norm_weight, norm_output, linear_weight,
                       // bias, logits_scratch, gang_counter
  int num_outputs = 3; // topk_weight, routing_indices, active_expert_ids

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is norm_input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::gang_rmsnorm_linear_bias_topk_kernel<bfloat16, $, $, $, $, $>(",
      m_per_tile,
      reduction_size,
      actual_hidden_dim,
      num_experts,
      num_experts_per_tok);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[3],");  // linear_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->input_ptrs[5],");  // logits_scratch
  code.e("    task_desc->input_ptrs[6],");  // gang_counter
  code.e("    task_desc->output_ptrs[0],"); // topk_weight
  code.e("    task_desc->output_ptrs[1],"); // routing_indices
  code.e("    task_desc->output_ptrs[2],"); // active_expert_ids
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx,");
  code.e("    $);", total_gang_tiles);
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_BIAS_TOPK_MI300,
                               code.to_string());
}

// Same fusion as above, with the `noaux_tc` router tail (GLM / DeepSeek):
// sigmoid scores plus an additive selection bias instead of a softmax.
//
// Two shape differences follow from that tail and are asserted below. The bias
// is the full [num_experts] correction vector rather than an XCD-partitioned
// slice, because the tail needs all of it and no worker folds its own element
// into a logit. And the routing tables are sized for the *total* expert count:
// GLM's always-on shared expert rides along as one extra row and one extra
// routing slot, so it is inferred from the gap rather than passed in.
//
// params: the 11 of the softmax variant, plus
//         [11] routed_scaling_factor in 1/1000 units (GLM-5: 2500 => 2.5f)
//         [12] norm_topk_prob (0/1)
int TaskRegister::register_gang_rmsnorm_linear_bias_topk_sigmoid_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 13);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];
  int actual_hidden_dim = params[7];
  int num_experts = params[8];
  int num_experts_per_tok = params[9];
  int total_gang_tiles = params[10];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;  // norm_input, norm_weight, norm_output, linear_weight,
                       // bias, logits_scratch, gang_counter
  int num_outputs = 3; // topk_weight, routing_indices, active_expert_ids

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is norm_input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  // e_score_correction_bias: the whole [num_experts] vector, unpartitioned.
  assert(input_ops[4]->dtensor.num_dims == 1);
  assert(input_ops[4]->output_tensors[0].dim[0] == num_experts);

  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  assert(output_ops[1]->output_tensors[0].num_dims == 2);
  assert(output_ops[2]->output_tensors[0].num_dims == 1);
  int num_total_experts = output_ops[1]->output_tensors[0].dim[0];
  int num_shared_experts = num_total_experts - num_experts;
  assert(num_shared_experts == 0 || num_shared_experts == 1);
  assert(output_ops[1]->output_tensors[0].dim[1] == batch_size);
  assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(output_ops[0]->output_tensors[0].dim[1] ==
         num_experts_per_tok + num_shared_experts);
  assert(output_ops[2]->output_tensors[0].dim[0] == num_total_experts + 1);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_bias_topk_kernel<bfloat16, $, $, $, $, "
         "$, true>(",
         m_per_tile,
         reduction_size,
         actual_hidden_dim,
         num_experts,
         num_experts_per_tok);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[3],");  // linear_weight
  code.e("    task_desc->input_ptrs[4],");  // e_score_correction_bias
  code.e("    task_desc->input_ptrs[5],");  // logits_scratch
  code.e("    task_desc->input_ptrs[6],");  // gang_counter
  code.e("    task_desc->output_ptrs[0],"); // topk_weight
  code.e("    task_desc->output_ptrs[1],"); // routing_indices
  code.e("    task_desc->output_ptrs[2],"); // active_expert_ids
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx,");
  code.e("    $,", total_gang_tiles);
  code.e("    $,", params[12] != 0 ? "true" : "false");
  code.e("    $ / 1000.0f,", params[11]);
  code.e("    $);", num_shared_experts);
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_BIAS_TOPK_MI300,
                               code.to_string());
}

// Fused RMSNorm + MXFP4 Gang Linear + Bias.
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim]
// Inputs: [norm_input, norm_weight, norm_output_scratch, mxfp4_weight, bias]
// Outputs: [linear_output]
int TaskRegister::register_gang_rmsnorm_linear_mxfp4_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 5);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_tiles_per_xcd = params[3];
  int actual_hidden_dim = params[4];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs =
      5; // norm_input, norm_weight, norm_output, mxfp4_weight, bias
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is norm_input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_mxfp4_bias_kernel<$, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[3],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // linear_output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_MI300,
                               code.to_string());
}

// Fused RMSNorm + MXFP4 Gang Linear + Bias + Argmax (norm-once, internal tile
// loop). params: [output_stride, output_per_wg, n_wgs_per_xcd, workers_per_xcd,
//          actual_hidden_dim]
// total_tiles_per_xcd = workers_per_xcd (each worker enters once, loops
// internally). Inputs: [norm_input, norm_weight, norm_output, mxfp4_weight,
// bias] Outputs: [argmax_part_value (bf16), argmax_part_index (int64)]
//   plus an OPTIONAL third output [ppl_logits (f32)] used only by perplexity
//   mode. When absent the kernel gets a nullptr and skips the HBM logits
//   write entirely, so the serving path is byte-for-byte the previous code.
int TaskRegister::register_gang_rmsnorm_linear_mxfp4_bias_argmax_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 5);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int workers_per_xcd = params[3];
  int actual_hidden_dim = params[4];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  // 2 outputs normally, 3 when a perplexity logits sink is attached.
  int num_outputs = (int)bgraph.operators.size() - num_inputs;
  assert(num_outputs == 2 || num_outputs == 3);
  bool emit_logits = (num_outputs == 3);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int reduction_size = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_mxfp4_bias_argmax_kernel<$, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[3],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // argmax_part_value (bf16)
  code.e("    task_desc->output_ptrs[1],"); // argmax_part_index (int64)
  if (emit_logits) {
    code.e("    task_desc->output_ptrs[2],"); // ppl_logits (f32)
  } else {
    code.e("    nullptr,"); // no logits sink
  }
  // Row to write in the logits buffer: the position being scored. step[0] is
  // the last consumed position, so the token produced here lands at step+1.
  code.e("    runtime_config.step[0] + 1,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", workers_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_ARGMAX_MI300,
                               code.to_string());
}

// Fused MulSumAdd + RMSNorm + MXFP4 Gang Linear + Bias.
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim, num_topk, input_stride]
// Inputs: [mlp_out, routing_weight, residual, norm_weight, norm_scratch,
//          mxfp4_weight, bias]
// Outputs: [x_output, qkv_output]
int TaskRegister::register_gang_mulsumradd_rmsnorm_linear_mxfp4_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_tiles_per_xcd = params[3];
  int actual_hidden_dim = params[4];
  int num_topk = params[5];
  int input_stride = params[6];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;  // mlp_out, routing_weight, residual, norm_weight,
                       // norm_scratch, mxfp4_weight, bias
  int num_outputs = 2; // x_output, qkv_output

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[2] is residual [batch, reduction_size]
  assert(input_ops[2]->dtensor.num_dims == 2);
  int batch_size = input_ops[2]->dtensor.dim[0];
  int reduction_size = input_ops[2]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kernel<$, $, $, $, "
         "$, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         num_topk,
         input_stride);
  code.e("    task_desc->input_ptrs[0],");  // mlp_out
  code.e("    task_desc->input_ptrs[1],");  // routing_weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // norm_weight
  code.e("    task_desc->input_ptrs[4],");  // norm_scratch
  code.e("    task_desc->input_ptrs[5],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[6],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // x_output (MulSumAdd result)
  code.e("    task_desc->output_ptrs[1],"); // qkv_output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx);");
  return register_task_variant(
      TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_MI300, code.to_string());
}

// Fused RMSNorm + MXFP4 Gang Linear + KV Cache Update (layer 0).
// params: [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim, head_dim, num_q_per_kv, page_size,
//          kv_stride, q_ws_stride]
// Inputs: [norm_input, norm_weight, norm_output, mxfp4_weight, bias]
// Outputs: [k_cache, v_cache, q_workspace]
int TaskRegister::register_gang_rmsnorm_linear_mxfp4_bias_kvupd_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 9);
  int output_per_wg = params[0];
  int n_wgs_per_xcd = params[1];
  int actual_hidden_dim = params[3];
  int head_dim = params[4];
  int num_q_per_kv = params[5];
  int page_size = params[6];
  int kv_stride = params[7];
  int q_ws_stride = params[8];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 3;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_mxfp4_bias_kvupd_kernel<$, $, $, $, $, "
         "$, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size);
  code.e("    task_desc->input_ptrs[0],");  // norm_input
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[3],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // k_cache
  code.e("    task_desc->output_ptrs[1],"); // v_cache
  code.e("    task_desc->output_ptrs[2],"); // q_workspace
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300,
                               code.to_string());
}

// Fused MulSumAdd + RMSNorm + MXFP4 Gang Linear + KV Cache Update (layers 1+).
// params: [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim, num_topk, input_stride,
//          head_dim, num_q_per_kv, page_size,
//          kv_stride, q_ws_stride]
// Inputs: [mlp_out, routing_weight, residual, norm_weight, norm_scratch,
//          mxfp4_weight, bias]
// Outputs: [x_output, k_cache, v_cache, q_workspace]
int TaskRegister::
    register_gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kvupd_mi300_task(
        threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 11);
  int output_per_wg = params[0];
  int n_wgs_per_xcd = params[1];
  int actual_hidden_dim = params[3];
  int num_topk = params[4];
  int input_stride = params[5];
  int head_dim = params[6];
  int num_q_per_kv = params[7];
  int page_size = params[8];
  int kv_stride = params[9];
  int q_ws_stride = params[10];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 4;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int batch_size = input_ops[2]->dtensor.dim[0];
  int reduction_size = input_ops[2]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_mulsumradd_rmsnorm_linear_mxfp4_bias_kvupd_kernel<$, $, "
         "$, $, $, $, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size,
         num_topk,
         input_stride);
  code.e("    task_desc->input_ptrs[0],");  // mlp_out
  code.e("    task_desc->input_ptrs[1],");  // routing_weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // norm_weight
  code.e("    task_desc->input_ptrs[4],");  // norm_scratch
  code.e("    task_desc->input_ptrs[5],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[6],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // x_output
  code.e("    task_desc->output_ptrs[1],"); // k_cache
  code.e("    task_desc->output_ptrs[2],"); // v_cache
  code.e("    task_desc->output_ptrs[3],"); // q_workspace
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    tile_idx);");
  return register_task_variant(
      TASK_GANG_MULSUMRADD_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300,
      code.to_string());
}

// Fused ResAddF32 + RMSNorm + MXFP4 Gang Linear + Bias (layers 1+).
// Reads from f32 workspace (pre-accumulated by W2 atomicAdd) instead of
// MulSumAdd. params: [output_stride, output_per_wg, n_wgs_per_xcd,
// total_tiles_per_xcd, actual_hidden_dim] Inputs: [workspace_f32, residual,
// norm_weight, norm_scratch, mxfp4_weight, bias] Outputs: [x_output,
// qkv_output]
int TaskRegister::register_gang_resaddf32_rmsnorm_linear_mxfp4_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 5);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_tiles_per_xcd = params[3];
  int actual_hidden_dim = params[4];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 6;  // workspace_f32, residual, norm_weight, norm_scratch,
                       // mxfp4_weight, bias
  int num_outputs = 2; // x_output, qkv_output

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[1] is residual [batch, reduction_size]
  assert(input_ops[1]->dtensor.num_dims == 2);
  int batch_size = input_ops[1]->dtensor.dim[0];
  int reduction_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_resaddf32_rmsnorm_linear_mxfp4_bias_kernel<$, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim);
  code.e("    task_desc->input_ptrs[0],");  // workspace_f32
  code.e("    task_desc->input_ptrs[1],");  // residual
  code.e("    task_desc->input_ptrs[2],");  // norm_weight
  code.e("    task_desc->input_ptrs[3],");  // norm_scratch
  code.e("    task_desc->input_ptrs[4],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[5],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // x_output
  code.e("    task_desc->output_ptrs[1],"); // qkv_output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx);");
  return register_task_variant(
      TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_MI300, code.to_string());
}

// Fused ResAddF32 + RMSNorm + MXFP4 Gang Linear + KV Cache Update (layers 1+).
// params: [output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim, head_dim, num_q_per_kv, page_size,
//          kv_stride, q_ws_stride]
// Inputs: [workspace_f32, residual, norm_weight, norm_scratch, mxfp4_weight,
// bias] Outputs: [x_output, k_cache, v_cache, q_workspace]
int TaskRegister::
    register_gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_mi300_task(
        threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 9);
  int output_per_wg = params[0];
  int n_wgs_per_xcd = params[1];
  int actual_hidden_dim = params[3];
  int head_dim = params[4];
  int num_q_per_kv = params[5];
  int page_size = params[6];
  int kv_stride = params[7];
  int q_ws_stride = params[8];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 6;
  int num_outputs = 4;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  int batch_size = input_ops[1]->dtensor.dim[0];
  int reduction_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_resaddf32_rmsnorm_linear_mxfp4_bias_kvupd_kernel<$, $, "
         "$, $, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size);
  code.e("    task_desc->input_ptrs[0],");  // workspace_f32
  code.e("    task_desc->input_ptrs[1],");  // residual
  code.e("    task_desc->input_ptrs[2],");  // norm_weight
  code.e("    task_desc->input_ptrs[3],");  // norm_scratch
  code.e("    task_desc->input_ptrs[4],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[5],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // x_output
  code.e("    task_desc->output_ptrs[1],"); // k_cache
  code.e("    task_desc->output_ptrs[2],"); // v_cache
  code.e("    task_desc->output_ptrs[3],"); // q_workspace
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    tile_idx);");
  return register_task_variant(
      TASK_GANG_RESADDF32_RMSNORM_LINEAR_MXFP4_BIAS_KVUPD_MI300,
      code.to_string());
}

// Fused QKV + Attention gang task.
// Phase 1: ResAddF32+RMSNorm+QKV+KVUpdate (all workers, gang tiles)
// Phase 2: CK FMHA attention (1 worker per XCD, after hierarchical barrier)
// params: [output_per_wg, n_wgs_per_xcd, total_qkv_tiles_per_xcd,
//          actual_hidden_dim, head_dim, num_q_per_kv, page_size,
//          kv_stride, q_ws_stride,
//          max_seq_len, num_kv_chunks, q_workspace_stride, kv_cache_stride,
//          num_kv_heads, sliding_window, has_sinks, total_tiles_per_xcd]
// Inputs (9): workspace_f32, residual, norm_weight, norm_scratch, weight, bias,
//             sinks, barrier, lse_acc
// Outputs (5): x_output, k_cache, v_cache, q_workspace, o_acc
int TaskRegister::register_gang_qkv_attn_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 16);
  int output_per_wg = params[0];
  int n_wgs_per_xcd = params[1];
  int total_qkv_tiles_per_xcd = params[2];
  int actual_hidden_dim = params[3];
  int head_dim = params[4];
  int num_q_per_kv = params[5];
  int page_size = params[6];
  int kv_stride = params[7];
  int q_ws_stride = params[8];
  int max_seq_len = params[9];
  int num_kv_chunks = params[10];
  int q_workspace_stride = params[11];
  int kv_cache_stride = params[12];
  int num_kv_heads = params[13];
  int sliding_window = params[14];
  int has_sinks = params[15];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 9;
  int num_outputs = 5;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // batch_size from residual tensor dim[0], reduction_size from weight tensor
  // dim[1]
  int batch_size = input_ops[1]->dtensor.dim[0];
  int reduction_size = input_ops[1]->dtensor.dim[1];
  float scale_s = 1.0f / sqrtf((float)head_dim) * 1.44269504088896340736f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_qkv_attn_fused_kernel_mi300<$, $, $, $, $, $, $, $, $, "
         "$, $, $, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size,
         max_seq_len,
         num_kv_chunks,
         q_workspace_stride,
         kv_cache_stride,
         num_kv_heads,
         sliding_window,
         has_sinks);
  code.e("    task_desc->input_ptrs[0],");  // workspace_f32
  code.e("    task_desc->input_ptrs[1],");  // residual
  code.e("    task_desc->input_ptrs[2],");  // norm_weight
  code.e("    task_desc->input_ptrs[3],");  // norm_scratch
  code.e("    task_desc->input_ptrs[4],");  // weight
  code.e("    task_desc->input_ptrs[5],");  // bias
  code.e("    task_desc->input_ptrs[6],");  // sinks (nullable)
  code.e("    task_desc->input_ptrs[7],");  // barrier
  code.e("    task_desc->input_ptrs[8],");  // lse_acc
  code.e("    task_desc->output_ptrs[0],"); // x_output
  code.e("    task_desc->output_ptrs[1],"); // k_cache
  code.e("    task_desc->output_ptrs[2],"); // v_cache
  code.e("    task_desc->output_ptrs[3],"); // q_workspace
  code.e("    task_desc->output_ptrs[4],"); // o_acc
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    $f,", scale_s);
  code.e("    $,", total_qkv_tiles_per_xcd);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_QKV_ATTN_FUSED_MI300,
                               code.to_string());
}

// MoE residual add from f32 workspace (last layer).
// params: [output_stride]
// Inputs: [workspace_f32, residual]
// Outputs: [output]
int TaskRegister::register_moe_residual_add_f32_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 1);
  int output_stride = params[0];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[1] is residual [batch, hidden_size] bf16
  assert(input_ops[1]->dtensor.num_dims == 2);
  int batch_size = input_ops[1]->dtensor.dim[0];
  int hidden_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_residual_add_f32_mi300_impl<$, $, $>(",
         batch_size,
         hidden_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");   // workspace_f32
  code.e("    task_desc->input_ptrs[1],");   // residual
  code.e("    task_desc->output_ptrs[0]);"); // output
  return register_task_variant(TASK_MOE_RESIDUAL_ADD_F32_MI300,
                               code.to_string());
}

// MXFP4 Gang Linear with Residual + Bias.
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd]
// Inputs: [input, mxfp4_weight, residual, bias]
// Outputs: [output]
int TaskRegister::register_gang_linear_mxfp4_res_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_tiles_per_xcd = params[3];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4; // input, mxfp4_weight, residual, bias
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_mxfp4_res_bias_kernel<$, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");  // input
  code.e("    task_desc->input_ptrs[1],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_LINEAR_MXFP4_RES_BIAS_MI300,
                               code.to_string());
}

// Fused O-PROJ + RMSNorm + Router + TopK.
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_oproj_tiles,
//          actual_hidden_dim, num_experts, topk_k, router_tile_n,
//          total_topk_tiles, total_tiles_per_xcd]
// Inputs: [input, mxfp4_weight, residual, bias, norm_weight, norm_output,
//          router_weight, router_bias, logits_scratch, counters]
// Outputs: [output, topk_weight, routing_indices, active_expert_ids]
int TaskRegister::register_gang_linear_mxfp4_res_bias_rmsnorm_topk_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 10);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_oproj_tiles = params[3];
  int actual_hidden_dim = params[4];
  int num_experts = params[5];
  int topk_k = params[6];
  int router_tile_n = params[7];
  int total_topk_tiles = params[8];
  int total_tiles_per_xcd = params[9];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 10;
  int num_outputs = 4;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is input [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_mxfp4_res_bias_rmsnorm_topk_kernel<$, $, $, $, "
         "$, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         num_experts,
         topk_k);
  code.e("    task_desc->input_ptrs[0],");  // input (attn_out)
  code.e("    task_desc->input_ptrs[1],");  // mxfp4_weight
  code.e("    task_desc->input_ptrs[2],");  // residual
  code.e("    task_desc->input_ptrs[3],");  // oproj_bias
  code.e("    task_desc->input_ptrs[4],");  // norm_weight
  code.e("    task_desc->input_ptrs[5],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[6],");  // router_weight
  code.e("    task_desc->input_ptrs[7],");  // router_bias
  code.e("    task_desc->input_ptrs[8],");  // logits_scratch
  code.e("    task_desc->input_ptrs[9],");  // counters (int32[2])
  code.e("    task_desc->output_ptrs[0],"); // output (attn_proj_out)
  code.e("    task_desc->output_ptrs[1],"); // topk_weight
  code.e("    task_desc->output_ptrs[2],"); // routing_indices
  code.e("    task_desc->output_ptrs[3],"); // active_expert_ids
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    $,", router_tile_n);
  code.e("    $,", total_oproj_tiles);
  code.e("    $,", total_topk_tiles);
  code.e("    $,", total_tiles_per_xcd);
  code.e("    tile_idx);");
  return register_task_variant(
      TASK_GANG_LINEAR_MXFP4_RES_BIAS_RMSNORM_TOPK_MI300, code.to_string());
}

// Fused absorbed-o_proj + post-attention RMSNorm + sigmoid/bias router + TopK.
//
// GLM's counterpart of the MXFP4 fused O-proj above, but assembled instead of
// rewritten: the two halves stay in their own kernels and the wrapper only
// supplies the cross-XCD barrier the dispatch used to provide. It reuses
// TASK_GANG_OPROJ_TOPK_MOE_FUSED_MI300's task id -- variants are what
// distinguish the two -- so nothing in runtime.cc has to learn a new type,
// and the global tile_idx that id already carries is exactly what the
// wrapper needs to recover its XCD.
//
// params: [hidden_size, oproj_rows_per_wg, oproj_tiles_per_xcd,
//          tiles_per_xcd, total_barrier_arrivals, actual_hidden_dim,
//          num_experts, topk_k, router_tile_n, total_router_tiles,
//          oproj_reduction_size, scaling_milli, norm_topk_prob, batch_size,
//          moe_intermediate, moe_w13_opw, moe_w2_opw,
//          moe_w13_tiles_per_xcd, moe_w2_tiles_per_xcd]
// Inputs (15): [attn_out, oproj_weight, residual, norm_weight, norm_output,
//               router_weight, router_bias, logits_scratch, router_counter,
//               oproj_counters, moe_gate_up_weight, moe_down_weight,
//               moe_w13_bias, moe_w2_bias, moe_swiglu_out]
// Outputs (5): [hidden, topk_weight, routing_indices, active_expert_ids,
//               moe_workspace_f32]
int TaskRegister::register_gang_oproj_router_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 19);
  int hidden_size = params[0];
  int oproj_rows_per_wg = params[1];
  int oproj_tiles_per_xcd = params[2];
  int tiles_per_xcd = params[3];
  int total_barrier_arrivals = params[4];
  int actual_hidden_dim = params[5];
  int num_experts = params[6];
  int topk_k = params[7];
  int router_tile_n = params[8];
  int total_router_tiles = params[9];
  int oproj_reduction_size = params[10];
  int scaling_milli = params[11];
  int norm_topk_prob = params[12];
  int batch_size = params[13];
  int moe_intermediate = params[14];
  int moe_w13_opw = params[15];
  int moe_w2_opw = params[16];
  int moe_w13_tiles_per_xcd = params[17];
  int moe_w2_tiles_per_xcd = params[18];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 15;
  int num_outputs = 5;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // The MXFP8 weight's row is the only cross-check the packing leaves: it
  // erases the logical output width, so shape agreement has to be asserted
  // against the byte stride instead.
  assert(input_ops[1]->dtensor.dim[1] ==
             oproj_rows_per_wg *
                 (oproj_reduction_size + oproj_reduction_size / 32) &&
         "o_proj MXFP8 weight is not packed at this reduction and row count");
  assert(oproj_tiles_per_xcd * oproj_rows_per_wg * 8 == hidden_size &&
         "the packed o_proj weight does not cover the hidden row exactly");
  // The o_proj barrier is sized to the workers that actually run o_proj or the
  // router, not to every dispatched worker: the MoE-only workers wait on
  // `routing_ready` instead, and decoupling the two barriers is the whole
  // point of having both. Clipped by the dispatch width, because a phase with
  // more tiles than workers grid-strides -- the extra tiles are extra rounds
  // on the same workers, not extra arrivals.
  {
    int const participants = oproj_tiles_per_xcd > router_tile_n
                                 ? oproj_tiles_per_xcd
                                 : router_tile_n;
    assert(total_barrier_arrivals ==
           (participants < tiles_per_xcd ? participants : tiles_per_xcd) * 8);
  }

  // e_score_correction_bias arrives whole, as the sigmoid tail reads all of it.
  assert(input_ops[6]->dtensor.num_dims == 1);
  assert(input_ops[6]->output_tensors[0].dim[0] == num_experts);

  int num_total_experts = output_ops[2]->output_tensors[0].dim[0];
  int num_shared_experts = num_total_experts - num_experts;
  assert(num_shared_experts == 0 || num_shared_experts == 1);
  assert(output_ops[1]->output_tensors[0].dim[1] ==
         topk_k + num_shared_experts);
  assert(output_ops[3]->output_tensors[0].dim[0] == num_total_experts + 1);

  // ── MoE geometry ──
  // The packed MXFP8 weight erases N and K, so num_experts is all it carries;
  // the widths come from the biases, exactly as the standalone MoE registrar
  // reads them. Both weights must agree on the expert count, and it is the
  // routed experts plus the shared one -- the MoE tile decoder reads the
  // activated count at active_expert_ids[MOE_NUM_EXPERTS].
  assert(input_ops[10]->output_tensors[0].num_dims == 3);
  assert(input_ops[11]->output_tensors[0].num_dims == 3);
  int moe_num_experts = input_ops[10]->output_tensors[0].dim[0];
  assert(input_ops[11]->output_tensors[0].dim[0] == moe_num_experts);
  assert(moe_num_experts == num_experts + num_shared_experts);
  assert(input_ops[12]->output_tensors[0].num_dims == 2);
  assert(input_ops[12]->output_tensors[0].dim[0] == moe_num_experts);
  assert(input_ops[12]->output_tensors[0].dim[1] == 2 * moe_intermediate);
  assert(input_ops[13]->output_tensors[0].num_dims == 2);
  assert(input_ops[13]->output_tensors[0].dim[0] == moe_num_experts);
  assert(input_ops[13]->output_tensors[0].dim[1] == hidden_size);
  // The SwiGLU activation is the half-width [batch, topk_total, intermediate]
  // slab W13 writes and W2 reduces over.
  assert(input_ops[14]->output_tensors[0].num_dims == 3);
  assert(input_ops[14]->output_tensors[0].dim[0] == batch_size);
  int moe_num_topk = input_ops[14]->output_tensors[0].dim[1];
  assert(moe_num_topk == topk_k + num_shared_experts);
  assert(input_ops[14]->output_tensors[0].dim[2] == moe_intermediate);
  // Both MoE kernels index [batch, topk, stride] with a compile-time stride,
  // so the tensor's own row stride has to be exactly that.
  assert(input_ops[14]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  assert(static_cast<int>(static_cast<kn::KNInputOp *>(
             input_ops[14]->dtensor.owner_op)->input_strides[1]) ==
         moe_intermediate);
  // W2's fused epilogue atomically accumulates into an f32 [batch, hidden].
  assert(output_ops[4]->output_tensors[0].num_dims == 2);
  assert(output_ops[4]->output_tensors[0].dim[0] == batch_size);
  assert(output_ops[4]->output_tensors[0].dim[1] == hidden_size);

  // Which element width the experts were packed at. The packed layout erases
  // N and K, so the workgroup stride is the only place the width shows: it is
  // OPW*(K + K/32) at MXFP8 and OPW*(K/2 + K/32) at MXFP4. The two stacks have
  // to agree -- the kernel carries one flag for the pair -- and everything
  // else about them, the expert count and the workgroup count, is identical
  // between the formats, so nothing above this needed to know.
  int const w13_fp8_bytes = moe_w13_opw * (hidden_size + hidden_size / 32);
  int const w13_fp4_bytes = moe_w13_opw * (hidden_size / 2 + hidden_size / 32);
  int const w2_fp8_bytes = moe_w2_opw * (moe_intermediate + moe_intermediate / 32);
  int const w2_fp4_bytes =
      moe_w2_opw * (moe_intermediate / 2 + moe_intermediate / 32);
  int const w13_bytes = input_ops[10]->output_tensors[0].dim[2];
  int const w2_bytes = input_ops[11]->output_tensors[0].dim[2];
  bool const moe_fp4 = (w13_bytes == w13_fp4_bytes);
  assert((moe_fp4 ? w13_fp4_bytes : w13_fp8_bytes) == w13_bytes &&
         "the packed W13 weight matches neither the MXFP8 nor the MXFP4 "
         "workgroup stride at this output_per_wg and hidden size");
  assert((moe_fp4 ? w2_fp4_bytes : w2_fp8_bytes) == w2_bytes &&
         "W13 and W2 are packed at different element widths");

  assert(2 * moe_intermediate % moe_w13_opw == 0);
  assert(hidden_size % moe_w2_opw == 0);
  int moe_w13_tiles_per_expert = batch_size * (2 * moe_intermediate / moe_w13_opw);
  int moe_w2_tiles_per_expert = batch_size * (hidden_size / moe_w2_opw);
  // Both MoE phases grid-stride by tiles_per_xcd, so a tile count above the
  // worker count costs extra rounds rather than dropping work. Every
  // dispatched worker still arrives at the W13->W2 barrier exactly once,
  // whichever side of the width the tile count falls on.
  assert(moe_w13_tiles_per_xcd > 0 && moe_w2_tiles_per_xcd > 0);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_oproj_router_fused_kernel_mi300<$, $, $, $, $, $, $, $, "
         "$, $, $, $, $, $, $>(",
         batch_size,
         oproj_reduction_size,
         oproj_rows_per_wg,
         hidden_size,
         actual_hidden_dim,
         num_experts,
         topk_k,
         moe_intermediate,
         moe_num_experts,
         moe_num_topk,
         moe_w13_tiles_per_expert,
         moe_w2_tiles_per_expert,
         moe_w13_opw,
         moe_w2_opw,
         moe_fp4 ? "true" : "false");
  for (int i = 0; i < num_inputs; i++) {
    code.e("    task_desc->input_ptrs[$],", i);
  }
  for (int i = 0; i < num_outputs; i++) {
    code.e("    task_desc->output_ptrs[$],", i);
  }
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", oproj_tiles_per_xcd);
  code.e("    $,", router_tile_n);
  code.e("    $,", tiles_per_xcd);
  code.e("    $,", total_barrier_arrivals);
  code.e("    $,", total_router_tiles);
  code.e("    $,", norm_topk_prob != 0 ? "true" : "false");
  code.e("    $ / 1000.0f,", scaling_milli);
  code.e("    $,", num_shared_experts);
  code.e("    $,", moe_w13_tiles_per_xcd);
  code.e("    $,", moe_w2_tiles_per_xcd);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_OPROJ_TOPK_MOE_FUSED_MI300,
                               code.to_string());
}

// Fused O-PROJ+TopK+MoE (combines tasks 213 and 187 into one gang task).
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_oproj_tiles,
//          actual_hidden_dim, num_experts, topk_k, router_tile_n,
//          total_topk_tiles, oproj_tiles_per_xcd,
//          moe_intermediate_size, moe_hidden_size,
//          w13_output_per_wg, w2_output_per_wg,
//          moe_total_tiles_per_xcd]
// Inputs (16): [attn_out, oproj_weight, residual, oproj_bias, norm_weight,
//               norm_output, router_weight, router_bias, logits_scratch,
//               hier_counters, gate_up_weight, down_weight, w13_bias,
//               w2_bias, moe_barrier, swiglu_out]
// Outputs (6): [oproj_output, topk_weight, routing_indices,
//               active_expert_ids, routing_weight_moe, workspace_f32]
int TaskRegister::register_gang_oproj_topk_moe_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 16);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_oproj_tiles = params[3];
  int actual_hidden_dim = params[4];
  int num_experts = params[5];
  int topk_k = params[6];
  int router_tile_n = params[7];
  int total_topk_tiles = params[8];
  int oproj_tiles_per_xcd = params[9];
  int moe_intermediate_size = params[10];
  int moe_hidden_size = params[11];
  int w13_output_per_wg = params[12];
  int w2_output_per_wg = params[13];
  int moe_total_tiles_per_xcd = params[14];
  int workers_per_xcd = params[15];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 16;
  int num_outputs = 6;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is attn_out [batch, reduction_size]
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_oproj_topk_moe_fused_kernel_mi300<$, $, $, $, $, $, $, "
         "$, $, $>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         num_experts,
         topk_k,
         moe_intermediate_size,
         moe_hidden_size,
         w13_output_per_wg,
         w2_output_per_wg);
  // O-PROJ+TopK inputs (same as task 213)
  code.e("    task_desc->input_ptrs[0],"); // attn_out
  code.e("    task_desc->input_ptrs[1],"); // oproj_weight
  code.e("    task_desc->input_ptrs[2],"); // residual
  code.e("    task_desc->input_ptrs[3],"); // oproj_bias
  code.e("    task_desc->input_ptrs[4],"); // norm_weight
  code.e("    task_desc->input_ptrs[5],"); // norm_output scratch
  code.e("    task_desc->input_ptrs[6],"); // router_weight
  code.e("    task_desc->input_ptrs[7],"); // router_bias
  code.e("    task_desc->input_ptrs[8],"); // logits_scratch
  code.e("    task_desc->input_ptrs[9],"); // hier_counters
  // MoE inputs (same as task 187)
  code.e("    task_desc->input_ptrs[10],"); // gate_up_weight
  code.e("    task_desc->input_ptrs[11],"); // down_weight
  code.e("    task_desc->input_ptrs[12],"); // w13_bias
  code.e("    task_desc->input_ptrs[13],"); // w2_bias
  code.e("    task_desc->input_ptrs[14],"); // moe_barrier
  code.e("    task_desc->input_ptrs[15],"); // swiglu_out
  // Outputs
  code.e("    task_desc->output_ptrs[0],"); // oproj_output
  code.e("    task_desc->output_ptrs[1],"); // topk_weight
  code.e("    task_desc->output_ptrs[2],"); // routing_indices
  code.e("    task_desc->output_ptrs[3],"); // active_expert_ids
  code.e("    task_desc->output_ptrs[4],"); // routing_weight (for MoE W2)
  code.e("    task_desc->output_ptrs[5],"); // workspace_f32
  // O-PROJ parameters
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    $,", router_tile_n);
  code.e("    $,", total_oproj_tiles);
  code.e("    $,", total_topk_tiles);
  code.e("    $,", oproj_tiles_per_xcd);
  // MoE parameters
  code.e("    $,", moe_total_tiles_per_xcd);
  code.e("    $,", workers_per_xcd);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_OPROJ_TOPK_MOE_FUSED_MI300,
                               code.to_string());
}

// Full-layer fused gang task (QKV+Attn+O-proj+TopK+MoE in one dispatch).
// params: [qkv_output_per_wg, qkv_n_wgs_per_xcd, total_qkv_tiles_per_xcd,
//          actual_hidden_dim, head_dim, num_q_per_kv, page_size,
//          kv_stride, q_ws_stride, max_seq_len, num_kv_chunks,
//          q_workspace_stride, kv_cache_stride, num_kv_heads,
//          sliding_window, has_sinks,
//          oproj_output_per_wg, oproj_output_stride, total_oproj_tiles,
//          num_experts, topk_k, router_tile_n, total_topk_tiles,
//          oproj_tiles_per_xcd, moe_total_tiles_per_xcd,
//          w13_output_per_wg, w2_output_per_wg,
//          moe_intermediate_size, workers_per_xcd]
// Inputs (23): [workspace_f32, residual, norm_weight_pre, norm_scratch_pre,
//               qkv_weight, qkv_bias, attn_sinks, qkv_barrier, lse_acc,
//               oproj_weight, oproj_bias, norm_weight_post, norm_scratch_post,
//               router_weight, router_bias, logits_scratch, oproj_counters,
//               moe_gate_up_weight, moe_down_weight, moe_w13_bias, moe_w2_bias,
//               moe_barrier, moe_swiglu_out]
// Outputs (11): [x_output, k_cache, v_cache, q_workspace, o_acc,
//                attn_proj_out, topk_weight, routing_indices,
//                active_expert_ids, moe_routing_weight, moe_workspace_f32]
int TaskRegister::register_gang_full_layer_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 38);
  int qkv_output_per_wg = params[0];
  int qkv_n_wgs_per_xcd = params[1];
  int total_qkv_tiles_per_xcd = params[2];
  int actual_hidden_dim = params[3];
  int head_dim = params[4];
  int num_q_per_kv = params[5];
  int page_size = params[6];
  int kv_stride = params[7];
  int q_ws_stride = params[8];
  int max_seq_len = params[9];
  int num_kv_chunks = params[10];
  int q_workspace_stride = params[11];
  int kv_cache_stride = params[12];
  int num_kv_heads = params[13];
  int sliding_window = params[14];
  int has_sinks = params[15];
  int oproj_output_per_wg = params[16];
  int oproj_output_stride = params[17];
  int total_oproj_tiles = params[18];
  int num_experts = params[19];
  int topk_k = params[20];
  int router_tile_n = params[21];
  int total_topk_tiles = params[22];
  int oproj_tiles_per_xcd = params[23];
  int moe_total_tiles_per_xcd = params[24];
  int w13_output_per_wg = params[25];
  int w2_output_per_wg = params[26];
  int moe_intermediate_size = params[27];
  int workers_per_xcd = params[28];
  // Expert-parallel: this rank owns experts
  // [moe_expert_base, moe_expert_base + moe_num_local_experts). Single-GPU and
  // replicated-MoE pass 0 / num_experts, which the kernel treats as identity.
  int moe_expert_base = params[29];
  int moe_num_local_experts = params[30];
  // Inline EP combine (Phase 9). ep_world_size == 1 compiles the phase out and
  // the task keeps its 24/11 tensor arity; > 1 adds the gather buffer, the
  // signal array, and the combined output.
  int ep_world_size = params[31];
  int ep_my_pe = params[32];
  int ep_fold_pe = params[33];
  bool ep_inline = ep_world_size > 1;
  // Slot-parallel expert split: own activated-list slots congruent to
  // ep_slot_me mod ep_slot_ws rather than an id range. 1/0 = disabled.
  int ep_slot_ws = params[34];
  int ep_slot_me = params[35];
  // The EP reduce, dissolved into this layer's QKV prologue: > 1 means
  // input[26] is the PREVIOUS layer's gather buffer and Phase 1 sums its slots
  // instead of reading a combined residual. ep_write_combined marks the one
  // layer (the last) whose consumer is a separate task and therefore still
  // needs the sum materialized into output[11].
  int ep_prev_slots = params[36];
  bool ep_write_combined = params[37] != 0;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = ep_inline ? (ep_prev_slots > 1 ? 27 : 26) : 24;
  int num_outputs = ep_inline ? 12 : 11;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // input[1] = residual [batch, hidden_dim]
  int batch_size = input_ops[1]->dtensor.dim[0];
  // input[1] = residual dim[1] = QKV reduction_size (padded hidden)
  int qkv_reduction_size = input_ops[1]->dtensor.dim[1];
  // output[4] = o_acc [batch, attn_dim] — O-proj reduction size
  int oproj_reduction_size = output_ops[4]->dtensor.dim[1];
  // MoE hidden size: same as intermediate
  int moe_hidden_size = moe_intermediate_size;

  float scale_s = 1.0f / sqrtf((float)head_dim) * 1.44269504088896340736f;
  int oproj_n_wgs_per_xcd = oproj_tiles_per_xcd / batch_size;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // 22 template parameters + DECODE_ONLY=true (compile out prefill attention
  // path) + the 2 expert-parallel ownership bounds + the 3 inline-combine ones
  // + the 2 slot-split ones + the 2 dissolved-reduce ones
  code.e("kernel::gang_full_layer_fused_kernel_mi300<$, $, $, $, $, $, $, $, "
         "$, $, $, $, $, $, $, $, $, $, $, $, $, $, true, $, $, $, $, $, $, "
         "$, $, $>(",
         batch_size,
         qkv_output_per_wg,
         qkv_reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size,
         max_seq_len,
         num_kv_chunks,
         q_workspace_stride,
         kv_cache_stride,
         num_kv_heads,
         sliding_window,
         has_sinks,
         oproj_output_per_wg,
         oproj_reduction_size,
         num_experts,
         topk_k,
         moe_intermediate_size,
         moe_hidden_size,
         w13_output_per_wg,
         w2_output_per_wg,
         moe_expert_base,
         moe_num_local_experts,
         ep_world_size,
         ep_my_pe,
         ep_fold_pe,
         ep_slot_ws,
         ep_slot_me,
         ep_prev_slots,
         ep_write_combined);
  // Pass input/output pointer arrays directly (2 params instead of 34)
  code.e("    task_desc->input_ptrs,");
  code.e("    task_desc->output_ptrs,");
  // 6 runtime config pointers
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  // 14 runtime parameters
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", qkv_n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    $f,", scale_s);
  code.e("    $,", total_qkv_tiles_per_xcd);
  code.e("    $,", oproj_n_wgs_per_xcd);
  code.e("    $,", oproj_output_stride);
  code.e("    $,", router_tile_n);
  code.e("    $,", total_oproj_tiles);
  code.e("    $,", total_topk_tiles);
  code.e("    $,", oproj_tiles_per_xcd);
  code.e("    $,", moe_total_tiles_per_xcd);
  code.e("    $,", workers_per_xcd);
  code.e("    tile_idx,");
  // Deterministic layer counter, published by the ml loop into the free int32
  // of the n_tile union member. The task derives its barrier release values
  // from this instead of snapshotting a shared counter -- see the
  // layer_counter comment in gang_full_layer_fused_mi300.cuh.
  code.e("    (int)task_desc->task_metadata._linear_reserved);");
  return register_task_variant(TASK_GANG_FULL_LAYER_FUSED_MI300,
                               code.to_string());
}

int TaskRegister::register_gang_full_layer_with_lmhead_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 33);
  int qkv_output_per_wg = params[0];
  int qkv_n_wgs_per_xcd = params[1];
  int total_qkv_tiles_per_xcd = params[2];
  int actual_hidden_dim = params[3];
  int head_dim = params[4];
  int num_q_per_kv = params[5];
  int page_size = params[6];
  int kv_stride = params[7];
  int q_ws_stride = params[8];
  int max_seq_len = params[9];
  int num_kv_chunks = params[10];
  int q_workspace_stride = params[11];
  int kv_cache_stride = params[12];
  int num_kv_heads = params[13];
  int sliding_window = params[14];
  int has_sinks = params[15];
  int oproj_output_per_wg = params[16];
  int oproj_output_stride = params[17];
  int total_oproj_tiles = params[18];
  int num_experts = params[19];
  int topk_k = params[20];
  int router_tile_n = params[21];
  int total_topk_tiles = params[22];
  int oproj_tiles_per_xcd = params[23];
  int moe_total_tiles_per_xcd = params[24];
  int w13_output_per_wg = params[25];
  int w2_output_per_wg = params[26];
  int moe_intermediate_size = params[27];
  int workers_per_xcd = params[28];
  int lm_output_per_wg = params[29];
  int lm_n_wgs_per_xcd = params[30];
  int lm_output_stride = params[31];
  int lm_actual_hidden_dim = params[32];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 28;
  int num_outputs = 13;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int batch_size = input_ops[1]->dtensor.dim[0];
  int qkv_reduction_size = input_ops[1]->dtensor.dim[1];
  int oproj_reduction_size = output_ops[4]->dtensor.dim[1];
  // MoE hidden size: same as intermediate
  int moe_hidden_size = moe_intermediate_size;
  int lm_reduction_size = input_ops[24]->dtensor.dim[0];

  float scale_s = 1.0f / sqrtf((float)head_dim) * 1.44269504088896340736f;
  int oproj_n_wgs_per_xcd = oproj_tiles_per_xcd / batch_size;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_full_layer_with_lmhead_fused_kernel_mi300<$, $, $, $, "
         "$, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $>(",
         batch_size,
         qkv_output_per_wg,
         qkv_reduction_size,
         actual_hidden_dim,
         head_dim,
         num_q_per_kv,
         page_size,
         max_seq_len,
         num_kv_chunks,
         q_workspace_stride,
         kv_cache_stride,
         num_kv_heads,
         sliding_window,
         has_sinks,
         oproj_output_per_wg,
         oproj_reduction_size,
         num_experts,
         topk_k,
         moe_intermediate_size,
         moe_hidden_size,
         w13_output_per_wg,
         w2_output_per_wg,
         lm_output_per_wg,
         lm_reduction_size);
  code.e("    task_desc->input_ptrs,");
  code.e("    task_desc->output_ptrs,");
  code.e("    runtime_config.rope_cos_ptr,");
  code.e("    runtime_config.rope_sin_ptr,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", qkv_n_wgs_per_xcd);
  code.e("    $,", kv_stride);
  code.e("    $,", q_ws_stride);
  code.e("    $f,", scale_s);
  code.e("    $,", total_qkv_tiles_per_xcd);
  code.e("    $,", oproj_n_wgs_per_xcd);
  code.e("    $,", oproj_output_stride);
  code.e("    $,", router_tile_n);
  code.e("    $,", total_oproj_tiles);
  code.e("    $,", total_topk_tiles);
  code.e("    $,", oproj_tiles_per_xcd);
  code.e("    $,", moe_total_tiles_per_xcd);
  code.e("    $,", workers_per_xcd);
  code.e("    $,", lm_n_wgs_per_xcd);
  code.e("    $,", lm_output_stride);
  code.e("    $,", lm_actual_hidden_dim);
  code.e("    tile_idx,");
  // See the matching comment in the non-LM-head variant above.
  code.e("    (int)task_desc->task_metadata._linear_reserved);");
  return register_task_variant(TASK_GANG_FULL_LAYER_WITH_LMHEAD_FUSED_MI300,
                               code.to_string());
}

// Gang linear with fused SiLU+mul.
// params: [output_stride, tile_n, m_tiles, m_per_tile,
// total_tile_pairs_per_xcd,
//          n_tile_pairs_per_xcd, wgm]
// Inputs: [activation(replicate), gate_up_weight(partition dim 0)]
// Outputs: [silu_mul_output(partition dim 1)]
int TaskRegister::register_gang_linear_silu_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0];
  int tile_n = params[1];
  int m_tiles = params[2];
  int m_per_tile = params[3];
  int total_tiles_per_xcd = params[4];
  int n_tiles_per_xcd = params[5];
  int wgm = params[6];

  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_silu_kernel<bfloat16, $, $>(",
         m_per_tile,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");  // full activation
  code.e("    task_desc->input_ptrs[1],");  // XCD's interleaved gate+up weight
  code.e("    task_desc->output_ptrs[0],"); // XCD's output columns
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles_per_xcd);
  code.e("    $,", wgm);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_LINEAR_SILU_MI300, code.to_string());
}

int TaskRegister::register_linear_silu_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int output_stride = params[0], tile_n = params[1], m_tiles = params[2];
  int m_per_tile = params[3], n_tiles = params[5], wgm = params[6];
  int reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  int num_inputs = 2, num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  reduction_size = input_ops[0]->dtensor.dim[1];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_linear_silu_kernel<bfloat16, $, $>(",
         m_per_tile,
         reduction_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", tile_n);
  code.e("    $,", output_stride);
  code.e("    $,", m_tiles);
  code.e("    $,", n_tiles);
  code.e("    $,", wgm);
  code.e("    task_desc->task_metadata.expert_offset);");
  return register_task_variant(TASK_LINEAR_SILU_MI300, code.to_string());
}

// Gang MoE W13 linear: 8 tasks (1 per XCD), tile_idx = expert_local *
// tiles_per_expert + tile params: [tiles_per_expert, max_experts_per_xcd,
// total_tiles_per_xcd]
int TaskRegister::register_gang_moe_w13_linear_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[3]: fuse_swiglu (epilogue SiLU-mul; needs pairwise-interleaved
  // gate/up weight rows and a half-width [batch, topk, intermediate] output)
  assert(params.size() == 4);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  bool fuse_swiglu = params[3] != 0;
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [batch, topk, output_size], or [batch, topk, output_size / 2]
  // when the SwiGLU is fused in.
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  // Input: [batch, reduction_size]
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  reduction_size = input_ops[0]->output_tensors[0].dim[1];
  // Weight: [num_experts, output_size, reduction_size]. The GEMM's N comes
  // from the weight rather than the output, since the fused epilogue emits
  // half as many columns as it computes.
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  output_size = input_ops[1]->output_tensors[0].dim[1];
  // Bias: [num_experts, output_stride]
  assert(input_ops[4]->output_tensors[0].num_dims == 2);
  output_stride = input_ops[4]->output_tensors[0].dim[1];
  // Activation row stride of the output tensor
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  int act_stride = static_cast<int>(kn_input_op->input_strides[1]);
  if (fuse_swiglu) {
    assert(2 * act_stride == output_stride);
  } else {
    assert(act_stride == output_stride);
  }

  int n_tiles = output_size / 64;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_w13_linear_kernel<bfloat16, $, $, $, $, $, $, $, "
         "$, $, $>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         tiles_per_expert,
         n_tiles,
         fuse_swiglu ? "true" : "false",
         act_stride);
  code.e("    task_desc->input_ptrs[0],");  // input activation
  code.e("    task_desc->input_ptrs[1],");  // expert weights
  code.e("    task_desc->input_ptrs[2],");  // routing indices
  code.e("    task_desc->input_ptrs[3],");  // mask
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MOE_W13_LINEAR_MI300,
                               code.to_string());
}

// Gang MoE W2 linear: 8 tasks (1 per XCD), tile_idx = expert_local *
// tiles_per_expert + tile params: [tiles_per_expert, max_experts_per_xcd,
// total_tiles_per_xcd]
int TaskRegister::register_gang_moe_w2_linear_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[3]: fuse_mulsumadd (epilogue topk-weight + f32 atomicAdd into a
  // [batch, hidden] workspace, replacing the [batch, topk, hidden] slab and
  // the moe_mul_sum_add pass over it). Adds routing_weight as input 5.
  assert(params.size() == 4);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  bool fuse_mulsumadd = params[3] != 0;
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = fuse_mulsumadd ? 6 : 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Input: [batch, topk, reduction_size]
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = input_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = input_ops[0]->output_tensors[0].dim[1];
  reduction_size = input_ops[0]->output_tensors[0].dim[2];
  // Weight: [num_experts, output_size, reduction_size]
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  output_size = input_ops[1]->output_tensors[0].dim[1];
  // Bias: [num_experts, output_stride]
  assert(input_ops[4]->output_tensors[0].num_dims == 2);
  output_stride = input_ops[4]->output_tensors[0].dim[1];
  if (fuse_mulsumadd) {
    // Output is the f32 workspace [batch, hidden]; row stride is hidden.
    assert(output_ops[0]->output_tensors[0].num_dims == 2);
    assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
    assert(output_ops[0]->output_tensors[0].dim[1] == output_size);
    assert(output_stride == output_size);
    // Routing weight: [batch, topk] float32
    assert(input_ops[5]->output_tensors[0].num_dims == 2);
    assert(input_ops[5]->output_tensors[0].dim[1] == num_experts_per_tok);
  } else {
    // Output: [batch, topk, output_size]
    assert(output_ops[0]->output_tensors[0].num_dims == 3);
    assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
    assert(output_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
    assert(output_ops[0]->output_tensors[0].dim[2] == output_size);
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    assert(static_cast<int>(kn_input_op->input_strides[1]) == output_stride);
  }

  int n_tiles = output_size / 64;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::gang_moe_w2_linear_kernel<bfloat16, $, $, $, $, $, $, $, $, $>(",
      batch_size,
      output_size,
      output_stride,
      reduction_size,
      num_experts,
      num_experts_per_tok,
      tiles_per_expert,
      n_tiles,
      fuse_mulsumadd ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");  // input activation
  code.e("    task_desc->input_ptrs[1],");  // expert weights
  code.e("    task_desc->input_ptrs[2],");  // routing indices
  code.e("    task_desc->input_ptrs[3],");  // mask
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // output / f32 workspace
  if (fuse_mulsumadd) {
    code.e("    tile_idx,");
    code.e("    task_desc->input_ptrs[5]);"); // routing weight
  } else {
    code.e("    tile_idx);");
  }
  return register_task_variant(TASK_GANG_MOE_W2_LINEAR_MI300, code.to_string());
}

// Gang MoE MXFP4 linear: 8 tasks (1 per XCD), MXFP4 weight dequant + MFMA
// params: [tiles_per_expert, max_experts_per_xcd, total_tiles_per_xcd,
// output_per_wg]
int TaskRegister::register_gang_moe_linear_mxfp4_mi300_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 4);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  int output_per_wg = params[3];
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [batch, topk, output_size]
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  // Input
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
  }
  // Weight: [num_experts, expert_wgs, wg_bytes]
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  // Bias: [num_experts, output_stride]
  assert(input_ops[4]->output_tensors[0].num_dims == 2);
  // Output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_pipelined_mxfp4_kernel_mi300<$, $, $, $, $, $, $, "
         "$, $>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         tiles_per_expert,
         output_per_wg,
         w13_linear ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");  // input activation
  code.e("    task_desc->input_ptrs[1],");  // expert weights (MXFP4 packed)
  code.e("    task_desc->input_ptrs[2],");  // routing indices
  code.e("    task_desc->input_ptrs[3],");  // mask
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    tile_idx);");
  if (w13_linear) {
    return register_task_variant(TASK_GANG_MOE_W13_LINEAR_MXFP4_MI300,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_GANG_MOE_W2_LINEAR_MXFP4_MI300,
                                 code.to_string());
  }
}

// Fused RMSNorm + MXFP8 Gang Linear + Bias. Same shape of registrar as the
// MXFP4 one above -- only the emitted kernel name differs, because the two
// kernels take identical parameters and differ solely in weight width.
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim]
int TaskRegister::register_gang_rmsnorm_linear_mxfp8_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 6);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_tiles_per_xcd = params[3];
  int actual_hidden_dim = params[4];
  // Expert parallelism. 0 or 1 is the single-GPU identity. > 1 makes the
  // residual fold the cross-rank sum as well, and re-reads input[0] as a
  // symmetric gather buffer of ep_peer_slots [batch, reduction] planes.
  int ep_peer_slots = params[5];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // The residual fold is optional and detected from the operator count, so the
  // plain five-input form keeps registering exactly as it did. With the fold
  // the caller adds the previous layer's f32 MoE accumulator and the residual
  // it belongs to, plus one output for the resolved row.
  bool const fuse_resadd = bgraph.operators.size() == 8;
  assert((bgraph.operators.size() == 6 || fuse_resadd) &&
         "gang_rmsnorm_linear_mxfp8_bias takes 5 inputs + 1 output, or 6 + 2 "
         "with the residual fold");
  int num_inputs = fuse_resadd ? 6 : 5; // norm_input, norm_weight, norm_output,
                                        // mxfp8_weight, bias, [resadd_ws]
  int num_outputs = fuse_resadd ? 2 : 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // input[0] is norm_input [batch, reduction_size], or the symmetric gather
  // buffer [ep_peer_slots, batch, reduction_size] under EP.
  int batch_size, reduction_size;
  if (ep_peer_slots > 1) {
    assert(fuse_resadd &&
           "EP reduces inside the residual fold; there is nowhere else to "
           "put it");
    assert(input_ops[0]->dtensor.num_dims == 3);
    assert(input_ops[0]->dtensor.dim[0] == ep_peer_slots);
    batch_size = input_ops[0]->dtensor.dim[1];
    reduction_size = input_ops[0]->dtensor.dim[2];
  } else {
    assert(input_ops[0]->dtensor.num_dims == 2);
    batch_size = input_ops[0]->dtensor.dim[0];
    reduction_size = input_ops[0]->dtensor.dim[1];
  }

  if (fuse_resadd) {
    // The fold reads the workspace and the residual as rows of exactly the
    // reduction width, so it has no padded-row variant.
    assert(actual_hidden_dim == reduction_size &&
           "the residual fold has no padded-row variant");
    assert(input_ops[5]->dtensor.num_dims == 2);
    assert(input_ops[5]->dtensor.dim[0] == batch_size);
    assert(input_ops[5]->dtensor.dim[1] == reduction_size);
    assert(output_ops[1]->dtensor.num_dims == 2);
    assert(output_ops[1]->dtensor.dim[0] == batch_size);
    assert(output_ops[1]->dtensor.dim[1] == reduction_size);
  }

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_mxfp8_bias_kernel<$, $, $, $, false, $, "
         "$>(",
         batch_size,
         output_per_wg,
         reduction_size,
         actual_hidden_dim,
         fuse_resadd ? "true" : "false",
         ep_peer_slots);
  code.e("    task_desc->input_ptrs[0],");  // norm_input == residual, folding
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch
  code.e("    task_desc->input_ptrs[3],");  // mxfp8_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // linear_output
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  if (fuse_resadd) {
    code.e("    tile_idx,");
    code.e("    task_desc->input_ptrs[5],");   // previous layer's MoE f32 ws
    code.e("    task_desc->output_ptrs[1]);"); // resolved residual stream
  } else {
    code.e("    tile_idx);");
  }
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_MXFP8_BIAS_MI300,
                               code.to_string());
}

// Fused q_a_layernorm + MXFP8 absorbed q_b_proj + MLA KV cache update.
// Registers as a variant of the plain MXFP8 rmsnorm+linear task type, exactly
// as the bf16 kvupd task is a variant of its own bf16 base -- the runtime
// bookkeeping is identical and only the emitted body differs.
// params: [output_stride, output_per_wg, n_wgs_per_xcd, total_tiles_per_xcd,
//          actual_hidden_dim, reduction_size, kv_lora_rank, qk_rope_head_dim,
//          kv_input_offset, max_seq_len, page_size]
int TaskRegister::register_gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 11);
  int output_stride = params[0];
  int output_per_wg = params[1];
  int n_wgs_per_xcd = params[2];
  int total_tiles_per_xcd = params[3];
  int actual_hidden_dim = params[4];
  int reduction_size = params[5];
  int kv_lora_rank = params[6];
  int qk_rope_head_dim = params[7];
  int kv_input_offset = params[8];
  int max_seq_len = params[9];
  int page_size = params[10];
  (void)total_tiles_per_xcd;

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // norm_input, norm_weight, norm_output, mxfp8_weight, bias,
  // kv_a_layernorm weight, cos, sin, paged latent cache
  int num_inputs = 9;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int kv_input_stride = input_ops[0]->dtensor.dim[1];
  assert(actual_hidden_dim <= reduction_size);
  assert(reduction_size <= kv_input_stride);
  // The kernel takes KV_INPUT_STRIDE as the GEMM's input row stride now, so
  // the narrowed reduction no longer doubles as one and batch_size is free.
  assert(kv_input_offset + kv_lora_rank + qk_rope_head_dim <= kv_input_stride);
  // One shared latent head, so the cache row stride is just the last dim.
  int kv_cache_stride = input_ops[8]->output_tensors[0].dim[3];
  assert(kv_cache_stride >= kv_lora_rank + qk_rope_head_dim);
  assert(output_per_wg == qk_rope_head_dim &&
         (kv_lora_rank + qk_rope_head_dim) % output_per_wg == 0);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_rmsnorm_linear_mxfp8_bias_mla_kvupd_kernel<$, $, $, $, "
         "$, $, $, $, $, $, $>(",
         batch_size,        /* BATCH_SIZE */
         output_per_wg,     /* OUTPUT_PER_WG */
         reduction_size,    /* REDUCTION_SIZE */
         actual_hidden_dim, /* ACTUAL_HIDDEN_DIM */
         kv_lora_rank,      /* KV_LORA_RANK */
         qk_rope_head_dim,  /* QK_ROPE_HEAD_DIM */
         kv_input_stride,   /* KV_INPUT_STRIDE */
         kv_cache_stride,   /* KV_CACHE_STRIDE */
         max_seq_len,       /* MAX_SEQ_LEN */
         page_size,         /* PAGE_SIZE */
         kv_input_offset);  /* KV_INPUT_OFFSET */
  code.e("    task_desc->input_ptrs[0],");  // norm_input, also kv_latent
  code.e("    task_desc->input_ptrs[1],");  // norm_weight
  code.e("    task_desc->input_ptrs[2],");  // norm_output scratch (writable)
  code.e("    task_desc->input_ptrs[3],");  // mxfp8_weight
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->input_ptrs[0],");  // kv_latent
  code.e("    task_desc->input_ptrs[5],");  // kv_a_layernorm weight
  code.e("    task_desc->input_ptrs[6],");  // cos
  code.e("    task_desc->input_ptrs[7],");  // sin
  code.e("    task_desc->output_ptrs[0],"); // q_workspace
  code.e("    task_desc->input_ptrs[8],");  // paged latent cache, written
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", n_wgs_per_xcd);
  code.e("    $,", output_stride);
  code.e("    tile_idx,");
  // Matches the standalone task's epsilon, as the bf16 variant does.
  code.e("    1e-6f);");
  return register_task_variant(TASK_GANG_RMSNORM_LINEAR_MXFP8_BIAS_MI300,
                               code.to_string());
}

// Gang MoE MXFP8 linear: 8 tasks (1 per XCD), FP8 weight x FP8 activation MFMA.
// params: [tiles_per_expert, max_experts_per_xcd, total_tiles_per_xcd,
// output_per_wg, fuse_epilogue]
//
// Five params rather than the MXFP4 version's four: these kernels carry GLM's
// fused epilogues (SwiGLU on W13, topk-weight + f32 atomicAdd on W2), so the
// registrar needs the same fuse flag the bf16 pair takes.
int TaskRegister::register_gang_moe_linear_mxfp8_mi300_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 5);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  int output_per_wg = params[3];
  bool fuse_epilogue = params[4] != 0;
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0,
      output_stride = 0, reduction_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // W2's fused epilogue reads routing_weight as input 5, exactly as the bf16
  // path does. W13's fused epilogue needs no extra input.
  int num_inputs = (!w13_linear && fuse_epilogue) ? 6 : 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Input: [batch, K] for W13, [batch, topk, K] for W2.
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    batch_size = input_ops[0]->output_tensors[0].dim[0];
    num_experts_per_tok = input_ops[0]->output_tensors[0].dim[1];
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
  }
  // Weight: [num_experts, expert_wgs, wg_bytes]. The packing erases the logical
  // N and K, so only num_experts survives here.
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  // Bias: [num_experts, output_stride]. This is where N comes from -- the
  // packed weight covers exactly output_stride rows per expert, so OUTPUT_SIZE
  // and OUTPUT_STRIDE coincide and padded rows are not representable.
  assert(input_ops[4]->output_tensors[0].num_dims == 2);
  assert(input_ops[4]->output_tensors[0].dim[0] == num_experts);
  output_stride = input_ops[4]->output_tensors[0].dim[1];
  assert(output_stride % output_per_wg == 0);

  // Output, and the batch/topk that W13 cannot read off its 2-D input.
  if (w13_linear) {
    assert(output_ops[0]->output_tensors[0].num_dims == 3);
    batch_size = output_ops[0]->output_tensors[0].dim[0];
    num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
    if (fuse_epilogue) {
      // Half-width [batch, topk, output_stride / 2] activation.
      assert(2 * output_ops[0]->output_tensors[0].dim[2] == output_stride);
    } else {
      assert(output_ops[0]->output_tensors[0].dim[2] == output_stride);
    }
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    assert(static_cast<int>(kn_input_op->input_strides[1]) ==
           output_ops[0]->output_tensors[0].dim[2]);
  } else if (fuse_epilogue) {
    // Output is the f32 workspace [batch, hidden].
    assert(output_ops[0]->output_tensors[0].num_dims == 2);
    assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
    assert(output_ops[0]->output_tensors[0].dim[1] == output_stride);
    // Routing weight: [batch, topk] float32
    assert(input_ops[5]->output_tensors[0].num_dims == 2);
    assert(input_ops[5]->output_tensors[0].dim[1] == num_experts_per_tok);
  } else {
    assert(output_ops[0]->output_tensors[0].num_dims == 3);
    assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
    assert(output_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
    assert(output_ops[0]->output_tensors[0].dim[2] == output_stride);
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    assert(static_cast<int>(kn_input_op->input_strides[1]) == output_stride);
  }
  // Tile space per expert: one tile per (token, workgroup) pair.
  assert(tiles_per_expert == batch_size * (output_stride / output_per_wg));

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_$_linear_mxfp8_kernel<$, $, $, $, $, $, $, $, $>(",
         w13_linear ? "w13" : "w2",
         batch_size,
         output_stride, // OUTPUT_SIZE
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         tiles_per_expert,
         output_per_wg,
         fuse_epilogue ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");  // input activation
  code.e("    task_desc->input_ptrs[1],");  // expert weights (MXFP8 packed)
  code.e("    task_desc->input_ptrs[2],");  // routing indices
  code.e("    task_desc->input_ptrs[3],");  // mask
  code.e("    task_desc->input_ptrs[4],");  // bias
  code.e("    task_desc->output_ptrs[0],"); // output / f32 workspace
  if (!w13_linear && fuse_epilogue) {
    code.e("    tile_idx,");
    code.e("    task_desc->input_ptrs[5]);"); // routing weight
  } else {
    code.e("    tile_idx);");
  }
  if (w13_linear) {
    return register_task_variant(TASK_GANG_MOE_W13_LINEAR_MXFP8_MI300,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_GANG_MOE_W2_LINEAR_MXFP8_MI300,
                                 code.to_string());
  }
}

// Gang fused W13+SwiGLU+W2 MXFP4 with per-expert pipelining.
// params: [tiles_per_expert, w13_output_per_wg, total_tiles_per_xcd,
// w2_output_per_wg]
int TaskRegister::register_gang_moe_fused_mxfp4_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 6);
  int tiles_per_expert = params[0];
  int w13_output_per_wg = params[1];
  int total_tiles_per_xcd = params[2];
  int w2_output_per_wg = params[3];
  // Expert-parallel: rank owns experts [expert_base, expert_base +
  // num_local_experts). For single-GPU these are 0 and the global expert
  // count, so the kernel's ownership test is a no-op.
  int expert_base = params[4];
  int num_local_experts = params[5];
  (void)tiles_per_expert;
  (void)total_tiles_per_xcd;

  // 8 inputs + 3 outputs
  // inputs: input, gate_up_weight, down_weight, routing, mask, w13_bias,
  // w2_bias, routing_weight outputs: swiglu_out, workspace_f32, barrier
  int num_inputs = 8;
  int num_outputs = 3;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // input[0]: [batch, hidden_size] BF16
  assert(input_ops[0]->dtensor.num_dims == 2);
  int batch_size = input_ops[0]->dtensor.dim[0];
  int hidden_size = input_ops[0]->dtensor.dim[1];

  // input[1]: gate_up weights [E_local, expert_wgs, wg_bytes]
  // Under expert-parallel this is the *local* expert count (sliced per rank);
  // single-GPU it equals the global count.
  assert(input_ops[1]->dtensor.num_dims == 3);
  int num_local_experts_weight = input_ops[1]->dtensor.dim[0];
  (void)num_local_experts_weight;

  // input[3]: routing [E_global, batch] -- routing/mask/barrier stay
  // replicated global structures, so the kernel's NUM_EXPERTS must be the
  // global count, not the sliced weight count.
  assert(input_ops[3]->dtensor.num_dims == 2);
  int num_experts = input_ops[3]->dtensor.dim[0];

  // output[0]: swiglu_out [batch, topk, intermediate_size]
  assert(output_ops[0]->dtensor.num_dims == 3);
  int num_topk = output_ops[0]->dtensor.dim[1];
  int intermediate_size = output_ops[0]->dtensor.dim[2];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_fused_mxfp4_kernel_mi300<$, $, $, $, $, $, $, $, "
         "$>(",
         batch_size,
         intermediate_size,
         hidden_size,
         num_experts,
         num_topk,
         w13_output_per_wg,
         w2_output_per_wg,
         expert_base,
         num_local_experts);
  code.e("    task_desc->input_ptrs[0],"); // input [batch, hidden]
  code.e("    task_desc->input_ptrs[1],"); // gate_up weights [E, W13_WGS,
                                           // wg_bytes]
  code.e("    task_desc->input_ptrs[2],"); // down weights [E, W2_WGS, wg_bytes]
  code.e("    task_desc->input_ptrs[3],"); // routing [E, batch]
  code.e("    task_desc->input_ptrs[4],"); // mask [E+1]
  code.e("    task_desc->input_ptrs[5],"); // w13_bias [E, 2*intermediate]
  code.e("    task_desc->input_ptrs[6],"); // w2_bias [E, hidden]
  code.e("    task_desc->input_ptrs[7],"); // routing_weight [batch, topk] f32
  code.e("    task_desc->output_ptrs[0],"); // swiglu_out [batch, topk,
                                            // intermediate]
  code.e("    task_desc->output_ptrs[1],"); // workspace_f32 [batch, hidden] f32
  code.e("    task_desc->output_ptrs[2],"); // barrier [2*E]
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MOE_FUSED_MXFP4_MI300,
                               code.to_string());
}

// Gang fused SwiGLU+W2 MXFP4 (no barrier, fused activation)
// Reads interleaved gate/up from W13 output, applies SwiGLU during FP8 quant,
// feeds into W2 MFMA. Same 5+1 signature as W2 but different kernel call.
int TaskRegister::register_gang_moe_swiglu_w2_mxfp4_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  int output_per_wg = params[3];
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Output: [batch, topk, hidden_size]
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  int hidden_size = output_ops[0]->output_tensors[0].dim[2];

  // Input: [batch, topk, 2*intermediate] (interleaved gate/up from W13)
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  int w13_out_dim = input_ops[0]->output_tensors[0].dim[2];
  int intermediate_size = w13_out_dim / 2;

  // Weight: [num_experts, expert_wgs, wg_bytes] (W2 down weights)
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_swiglu_w2_mxfp4_kernel_mi300<$, $, $, $, $, $, $>(",
         batch_size,
         intermediate_size,
         hidden_size,
         num_experts,
         num_experts_per_tok,
         tiles_per_expert,
         output_per_wg);
  code.e("    task_desc->input_ptrs[0],");  // w13 output (interleaved gate/up)
  code.e("    task_desc->input_ptrs[1],");  // W2 weights (MXFP4)
  code.e("    task_desc->input_ptrs[2],");  // routing indices
  code.e("    task_desc->input_ptrs[3],");  // mask
  code.e("    task_desc->input_ptrs[4],");  // W2 bias
  code.e("    task_desc->output_ptrs[0],"); // output
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MOE_SWIGLU_W2_MXFP4_MI300,
                               code.to_string());
}

// Gang MoE W13 MXFP4 with SwiGLU fused into epilogue.
// Same MFMA as W13 but epilogue pairs gate/up outputs, applies SwiGLU,
// and writes half-sized output: [batch, topk, intermediate] instead of
// [batch, topk, 2*intermediate].
// params: [tiles_per_expert, max_experts_per_xcd, total_tiles_per_xcd,
// output_per_wg]
int TaskRegister::register_gang_moe_w13_swiglu_mxfp4_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 4);
  int tiles_per_expert = params[0];
  int max_experts_per_xcd = params[1];
  int total_tiles_per_xcd = params[2];
  int output_per_wg = params[3];
  (void)max_experts_per_xcd;
  (void)total_tiles_per_xcd;

  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Output: [batch, topk, intermediate] (half of W13's 2*intermediate)
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  int intermediate_size = output_ops[0]->output_tensors[0].dim[2];

  // Input: [batch, hidden] (W13_LINEAR=true, 2D input)
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int reduction_size = input_ops[0]->output_tensors[0].dim[1];

  // Weight: [num_experts, expert_wgs, wg_bytes] — still the full interleaved
  // gate_up weights
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];

  // The MFMA output size is 2*intermediate (full interleaved gate/up)
  int output_size = 2 * intermediate_size;
  int output_stride = output_size;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_pipelined_mxfp4_kernel_mi300<$, $, $, $, $, $, $, "
         "$, true, true>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         tiles_per_expert,
         output_per_wg);
  code.e("    task_desc->input_ptrs[0],"); // input activation [batch, hidden]
  code.e("    task_desc->input_ptrs[1],"); // expert weights (MXFP4 packed,
                                           // interleaved gate/up)
  code.e("    task_desc->input_ptrs[2],"); // routing indices
  code.e("    task_desc->input_ptrs[3],"); // mask
  code.e("    task_desc->input_ptrs[4],"); // bias (interleaved gate/up bias)
  code.e(
      "    task_desc->output_ptrs[0],"); // output [batch, topk, intermediate]
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MOE_W13_SWIGLU_MXFP4_MI300,
                               code.to_string());
}

// Gang CK FMHA attention: 8 tasks (1 per XCD), tile_idx → (request_id, kv_head)
// Fuses KV cache update + CK FMHA attention into one gang task.
// params: [num_q_heads, num_kv_heads, qk_norm, rotary_embed, max_seq_len,
//          page_size, num_kv_chunks, total_work_items_per_xcd,
//          total_work_items, q_workspace_stride]
int TaskRegister::register_gang_attn_split_kv_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 10);
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  int total_work_items = params[8];
  int q_workspace_stride = params[9];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 3;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_qo_per_kv = num_q_heads / num_kv_heads;
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int kv_stride = head_dim * num_kv_heads;
  int max_tokens = input_ops[0]->dtensor.dim[0];

  // Cap MAX_TOKENS to fit in LDS (same as kv_cache_update)
  {
    constexpr int LDS_LIMIT = 58368; // 60KB - 3KB reserved
    int per_token_bytes = (num_qo_per_kv + 1) * head_dim * 2; // bf16
    int overhead = 256;
    int max_tokens_lds = (LDS_LIMIT - overhead) / per_token_bytes;
    if (max_tokens_lds < 1) {
      max_tokens_lds = 1;
    }
    if (max_tokens > max_tokens_lds) {
      max_tokens = max_tokens_lds;
    }
  }

  // Per-kv-head pointer offsets
  int qkv_head_offset = qkv_stride / num_kv_heads;
  int kv_cache_head_offset = head_dim;

  float scale_s = 1.0f / sqrtf((float)head_dim) * 1.44269504088896340736f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_attention_split_kv_kernel<bfloat16,");
  code.e("    $, $, $, $, $, $, $, $, $, $, $, $>(",
         num_qo_per_kv,                     // NUM_QO_HEADS_PER_KV
         num_kv_heads,                      // NUM_KV_HEADS_ACTUAL
         kv_stride,                         // KV_CACHE_STRIDE
         qkv_stride,                        // QKV_STRIDE
         head_dim,                          // HEAD_DIM
         max_seq_len,                       // MAX_SEQ_LEN
         page_size,                         // PAGE_SIZE
         max_tokens,                        // MAX_TOKENS
         num_kv_chunks,                     // NUM_KV_CHUNKS
         qkv_head_offset,                   // QKV_HEAD_OFFSET
         kv_cache_head_offset,              // KV_CACHE_HEAD_OFFSET
         q_workspace_stride);               // Q_WORKSPACE_STRIDE
  code.e("    task_desc->input_ptrs[0],");  // qkv (full, un-partitioned)
  code.e("    task_desc->input_ptrs[1],");  // k_cache (full)
  code.e("    task_desc->input_ptrs[2],");  // v_cache (full)
  code.e("    task_desc->output_ptrs[1],"); // output (full)
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    $,", params[2] > 0);         // qk_norm
  code.e("    $,", params[3] > 0);         // rope
  code.e("    task_desc->input_ptrs[3],"); // q_norm
  code.e("    task_desc->input_ptrs[4],"); // k_norm
  code.e("    task_desc->input_ptrs[5],"); // cos
  code.e("    task_desc->input_ptrs[6],"); // sin
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0],"); // lse (full)
  code.e("    task_desc->output_ptrs[2],"); // q_workspace (full)
  code.e("    $,", num_kv_heads);
  code.e("    $,", total_work_items);
  code.e("    tile_idx,");
  code.e("    $f);", scale_s);
  return register_task_variant(TASK_GANG_ATTN_SPLIT_KV_MI300, code.to_string());
}

// Gang absorbed MLA decode (GLM-5): tile_idx → (request_id, q_head_group,
// kv_chunk). MLA has a single shared latent head, so the GQA gang's
// kv_head == xcd_id mapping is replaced by a q-head-group x sequence-chunk
// split; see gang_mla_decode_mi300.cuh.
// params: [num_q_heads, kv_lora_rank, qk_rope_head_dim, qk_head_dim,
//          max_seq_len, page_size, num_kv_chunks, total_work_items_per_xcd,
//          total_work_items, q_workspace_stride]
int TaskRegister::register_gang_mla_decode_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 11);
  int num_q_heads = params[0];
  int kv_lora_rank = params[1];
  int qk_rope_head_dim = params[2];
  int qk_head_dim = params[3];
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  int total_work_items = params[8];
  int q_workspace_stride = params[9];
  int batch_size = params[10];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;  // q_workspace, kv_cache
  int num_outputs = 2; // lse, output

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Latent cache rows are [c_kv | k_rope], one shared head per token.
  int kv_cache_stride = kv_lora_rank + qk_rope_head_dim;
  assert(input_ops[1]->dtensor.dim[input_ops[1]->dtensor.num_dims - 1] ==
         kv_cache_stride);

  // MLA scales by the *unabsorbed* qk head dim (qk_nope + qk_rope), not the
  // absorbed reduction width.
  float scale_s = 1.0f / sqrtf((float)qk_head_dim) * 1.44269504088896340736f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // WRITE_THROUGH stays false here -- the standalone decode always feeds the
  // split-KV merge, so its o_acc is read back on the same CU and there is
  // nothing to flush past L2. BATCH_SIZE is what makes the tile decomposition
  // recover a token index; see the token_idx line in gang_mla_decode_kernel.
  code.e("kernel::gang_mla_decode_kernel<bfloat16,");
  code.e("    $, $, $, $, $, $, $, $, false, $>(",
         num_q_heads,        // NUM_Q_HEADS
         kv_lora_rank,       // KV_LORA_RANK
         qk_rope_head_dim,   // QK_ROPE_HEAD_DIM
         page_size,          // PAGE_SIZE
         max_seq_len,        // MAX_SEQ_LEN
         num_kv_chunks,      // NUM_KV_CHUNKS
         q_workspace_stride, // Q_WORKSPACE_STRIDE
         kv_cache_stride,    // KV_CACHE_STRIDE
         batch_size);        // BATCH_SIZE
  code.e("    task_desc->input_ptrs[0],");  // q_workspace (full)
  code.e("    task_desc->input_ptrs[1],");  // latent kv cache (full)
  code.e("    task_desc->output_ptrs[1],"); // output (full)
  code.e("    task_desc->output_ptrs[0],"); // lse (full)
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    $,", total_work_items);
  code.e("    tile_idx,");
  code.e("    $f);", scale_s);
  return register_task_variant(TASK_GANG_MLA_DECODE_MI300, code.to_string());
}

// The attention half of a GLM decoder layer in one gang task: qkv_a, q_b +
// latent append, MLA decode, split-KV merge. Registers as a variant of the
// MLA decode task type, which is what puts it in runtime.cc's global-tile_idx
// list -- the wrapper needs tile_idx = xcd_id * tiles_per_xcd + xcd_rank and
// synthesizes each sub-kernel's own index from it.
// params: [batch_size, qkv_opw, qkv_actual_hidden, qkv_n_wgs_per_xcd,
//          qkv_output_stride, qb_opw, qb_reduction, qb_actual_hidden,
//          qb_n_wgs_per_xcd, qb_output_stride, kv_lora_rank,
//          qk_rope_head_dim, kv_input_offset, max_seq_len, page_size,
//          num_q_heads, qk_head_dim, num_kv_chunks, q_workspace_stride,
//          mla_total_work_items, mla_tiles_per_xcd, merge_dim_splits,
//          merge_write_through, merge_tiles_per_xcd, tiles_per_xcd]
int TaskRegister::register_gang_mla_attn_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 25);
  int batch_size = params[0];
  int qkv_opw = params[1];
  int qkv_actual_hidden = params[2];
  int qkv_n_wgs_per_xcd = params[3];
  int qkv_output_stride = params[4];
  int qb_opw = params[5];
  int qb_reduction = params[6];
  int qb_actual_hidden = params[7];
  int qb_n_wgs_per_xcd = params[8];
  int qb_output_stride = params[9];
  int kv_lora_rank = params[10];
  int qk_rope_head_dim = params[11];
  int kv_input_offset = params[12];
  int max_seq_len = params[13];
  int page_size = params[14];
  int num_q_heads = params[15];
  int qk_head_dim = params[16];
  int num_kv_chunks = params[17];
  int q_workspace_stride = params[18];
  int mla_total_work_items = params[19];
  int mla_tiles_per_xcd = params[20];
  int merge_dim_splits = params[21];
  bool merge_write_through = params[22] != 0;
  int merge_tiles_per_xcd = params[23];
  int tiles_per_xcd = params[24];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 15;
  int num_outputs = 6;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // x is [batch, hidden]; the qkv_a GEMM reduces over the whole row.
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->dtensor.dim[0] == batch_size);
  int qkv_reduction = input_ops[0]->dtensor.dim[1];
  // Both GEMMs take their input row stride explicitly now (qkv_a's equals its
  // reduction, q_b's is kv_input_stride), so batch_size is free.

  // qkv_a_out carries [q_a | latent] and is the q_b GEMM's input row.
  assert(output_ops[0]->dtensor.num_dims == 2);
  int kv_input_stride = output_ops[0]->dtensor.dim[1];
  assert(qkv_output_stride == kv_input_stride);
  assert(qkv_n_wgs_per_xcd * qkv_opw * 8 == kv_input_stride &&
         "the packed qkv_a weight does not cover the row exactly");
  assert(qb_actual_hidden <= qb_reduction && qb_reduction <= kv_input_stride);
  assert(kv_input_offset + kv_lora_rank + qk_rope_head_dim <= kv_input_stride);

  // The packed MXFP8 weights erase N and K, so the byte stride is the only
  // cross-check left.
  assert(input_ops[3]->dtensor.dim[1] ==
             qkv_opw * (qkv_reduction + qkv_reduction / 32) &&
         "qkv_a MXFP8 weight is not packed at this reduction and row count");
  assert(input_ops[7]->dtensor.dim[1] ==
             qb_opw * (qb_reduction + qb_reduction / 32) &&
         "q_b MXFP8 weight is not packed at this reduction and row count");

  // One shared latent head, so the cache row stride is just the last dim.
  assert(input_ops[12]->dtensor.num_dims == 4);
  assert(input_ops[12]->dtensor.dim[2] == 1);
  int kv_cache_stride = input_ops[12]->dtensor.dim[3];
  assert(kv_cache_stride == kv_lora_rank + qk_rope_head_dim);
  assert(qb_opw >= qk_rope_head_dim &&
         kv_cache_stride % qb_opw == 0 &&
         "a head's rope slice has to fit inside one q_b workgroup");
  assert((qb_n_wgs_per_xcd * qb_opw) % kv_cache_stride == 0 &&
         "each XCD's q_b column chunk must hold whole heads");
  assert(qb_n_wgs_per_xcd * qb_opw * 8 == qb_output_stride);

  // 29 cache-line-strided int32 slots, three monotonic barriers: qkv_a->q_b
  // at [0..8], q_b->decode at [10..18], decode->merge at [20..28].
  assert(input_ops[13]->dtensor.num_dims == 1);
  assert(input_ops[13]->dtensor.dim[0] >= 29 * 16);

  // The MoE accumulator this task resolves into the residual stream, and the
  // row it writes it to. Both are exactly the qkv_a reduction wide -- the
  // fused prologue has no padded-row variant.
  assert(input_ops[14]->dtensor.num_dims == 2);
  assert(input_ops[14]->dtensor.dim[0] == batch_size);
  assert(input_ops[14]->dtensor.dim[1] == qkv_reduction);
  assert(qkv_actual_hidden == qkv_reduction &&
         "the residual resolve reads workspace and residual rows of exactly "
         "the reduction width, so a padded qkv_a row has no meaning here");
  assert(output_ops[5]->dtensor.num_dims == 2);
  assert(output_ops[5]->dtensor.dim[0] == batch_size);
  assert(output_ops[5]->dtensor.dim[1] == qkv_reduction);

  // Phases wider than the dispatch width grid-stride by it; the width itself
  // is the caller's clamp against the resident worker count.
  assert(tiles_per_xcd > 0);
  assert(num_q_heads % 16 == 0);
  int num_q_groups = num_q_heads / 16;
  // mla_total_work_items counts (query row, q group, kv chunk) triples, so it
  // carries batch_size as well as the request count: gang_mla_decode_kernel
  // decomposes tile_idx into (q_group, kv_chunk, token, request), with the
  // token dimension inside the request one because a request's rows share its
  // page list. request_id is still a literal 0 here (see the note below), so
  // the quotient IS batch_size rather than merely bounded by it.
  assert(mla_total_work_items % (num_q_groups * num_kv_chunks) == 0);
  assert(mla_total_work_items / (num_q_groups * num_kv_chunks) == batch_size);
  assert(merge_dim_splits >= 1 && kv_lora_rank % merge_dim_splits == 0);
  assert(num_kv_chunks > 1 &&
         "with one chunk the decode writes attn_out directly and there is no "
         "merge phase; use the unfused path");
  assert(q_workspace_stride == num_q_heads * kv_cache_stride);

  // MLA scales by the *unabsorbed* qk head dim (qk_nope + qk_rope), not the
  // absorbed reduction width. 1/sqrt(d) folded with log2(e), as the decode
  // kernel exponentiates base 2.
  float scale_s = 1.0f / sqrtf((float)qk_head_dim) * 1.44269504088896340736f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_mla_attn_fused_kernel_mi300<$, $, $, $, $, $, $, $, $, "
         "$, $, $, $, $, $, $, $, $, $>(",
         batch_size,
         qkv_opw,
         qkv_reduction,
         qkv_actual_hidden,
         qb_opw,
         qb_reduction,
         qb_actual_hidden,
         kv_lora_rank,
         qk_rope_head_dim,
         kv_input_stride,
         kv_cache_stride,
         max_seq_len,
         page_size,
         kv_input_offset,
         num_q_heads,
         num_kv_chunks,
         q_workspace_stride,
         merge_dim_splits,
         merge_write_through ? "true" : "false");
  for (int i = 0; i < num_inputs; i++) {
    code.e("    task_desc->input_ptrs[$],", i);
  }
  for (int i = 0; i < num_outputs; i++) {
    code.e("    task_desc->output_ptrs[$],", i);
  }
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  // NOT task_metadata.request_id: TaskMetadata is a union, and this task is
  // in the gang group that sets n_tile_start = bid.x * tiles_per_xcd
  // (runtime.cc:483), which aliases request_id. Reading it back here yields
  // 0, 28, 56, ... 196 -- one garbage index per XCD into a two-entry
  // qo_indptr buffer, which the merge then multiplies by the output row
  // stride. Gang tasks own the whole GPU for one request by construction (the
  // standalone kvupd task lands on 0 only because its task type takes the
  // n_tile_start = 0 branch), so the request index is a literal zero. The
  // python layer asserts max_num_batched_requests == 1 to keep it honest.
  code.e("    (int16_t)0,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", qkv_n_wgs_per_xcd);
  code.e("    $,", qkv_output_stride);
  code.e("    $,", qb_n_wgs_per_xcd);
  code.e("    $,", qb_output_stride);
  code.e("    $,", mla_tiles_per_xcd);
  code.e("    $,", mla_total_work_items);
  code.e("    $,", merge_tiles_per_xcd);
  code.e("    $,", tiles_per_xcd);
  code.e("    $f,", scale_s);
  // Matches the standalone kvupd task's epsilon.
  code.e("    1e-6f,");
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MLA_DECODE_MI300, code.to_string());
}

// Whole-layer fused GLM task: the attention half and the MoE half of one
// decoder layer in a single gang dispatch, with an in-kernel cross-XCD
// barrier where the task graph used to put an event. Six dispatches per layer
// become one.
//
// This is the union of register_gang_mla_attn_fused_mi300_task and
// register_gang_oproj_router_fused_mi300_task; the assertions below are those
// two sets, re-indexed onto the merged pointer map and with the checks that
// only the fusion can make added. Kept as one registrar rather than two calls
// because the two halves disagree about tiles_per_xcd -- the fused dispatch
// width is the max over every phase in the layer, and both halves have to be
// told the same one or xcd_id = tile_idx / tiles_per_xcd decodes differently
// on either side of Phase 8.
//
// Inputs (27):  0 x, 1 pre_norm_weight, 2 pre_norm_scratch, 3 qkv_weight,
//               4 qkv_bias, 5 q_a_norm_weight, 6 q_a_norm_scratch,
//               7 qb_weight, 8 qb_bias, 9 kv_norm_weight, 10 cos, 11 sin,
//               12 kv_cache, 13 moe_workspace_f32 (read), 14 counters,
//               15 oproj_weight, 16 residual, 17 post_norm_weight,
//               18 post_norm_output, 19 router_weight, 20 router_bias,
//               21 logits_scratch, 22 moe_gate_up_weight,
//               23 moe_down_weight, 24 moe_w13_bias, 25 moe_w2_bias,
//               26 moe_swiglu_out
//               [27, 28 under EP: ep_gather, ep_signal]
//               last: wuv_weight, un-absorbed kv_b_v only (index 29 under EP,
//               27 otherwise; absent when absorbed)
// Outputs (12): 0 qkv_a_out, 1 q_workspace, 2 lse, 3 o_acc, 4 attn_out,
//               5 x_out, 6 hidden, 7 topk_weight, 8 routing_indices,
//               9 active_expert_ids, 10 moe_workspace_f32 (written),
//               11 v_out (un-absorbed kv_b_v only)
int TaskRegister::register_gang_mla_full_layer_fused_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 54);
  // ── attention half ──
  int batch_size = params[0];
  int qkv_opw = params[1];
  int qkv_actual_hidden = params[2];
  int qkv_n_wgs_per_xcd = params[3];
  int qkv_output_stride = params[4];
  int qb_opw = params[5];
  int qb_reduction = params[6];
  int qb_actual_hidden = params[7];
  int qb_n_wgs_per_xcd = params[8];
  int qb_output_stride = params[9];
  int kv_lora_rank = params[10];
  int qk_rope_head_dim = params[11];
  int kv_input_offset = params[12];
  int max_seq_len = params[13];
  int page_size = params[14];
  int num_q_heads = params[15];
  int qk_head_dim = params[16];
  int num_kv_chunks = params[17];
  int q_workspace_stride = params[18];
  int mla_total_work_items = params[19];
  int mla_tiles_per_xcd = params[20];
  int merge_dim_splits = params[21];
  bool merge_write_through = params[22] != 0;
  int merge_tiles_per_xcd = params[23];
  int tiles_per_xcd = params[24];
  // ── MoE half ──
  int hidden_size = params[25];
  int oproj_rows_per_wg = params[26];
  int oproj_tiles_per_xcd = params[27];
  int total_barrier_arrivals = params[28];
  int actual_hidden_dim = params[29];
  int num_experts = params[30];
  int topk_k = params[31];
  int router_tile_n = params[32];
  int total_router_tiles = params[33];
  int oproj_reduction_size = params[34];
  int scaling_milli = params[35];
  int norm_topk_prob = params[36];
  int moe_intermediate = params[37];
  int moe_w13_opw = params[38];
  int moe_w2_opw = params[39];
  int moe_w13_tiles_per_xcd = params[40];
  int moe_w2_tiles_per_xcd = params[41];
  // Expert parallelism. 1 / 0 / 0 / 0 is the single-GPU identity and compiles
  // the whole EP block, the tail variant and the expert remap out; > 1 adds
  // the symmetric gather buffer and the signal array as inputs [27] and [28].
  int ep_world_size = params[42];
  int ep_my_pe = params[43];
  int ep_fold_pe = params[44];
  bool ep_tail_only = params[45] != 0;
  // ── un-absorbed kv_b_v ──
  // 0 keeps W_UV folded into o_proj, which is the shape every assertion below
  // already describes. Non-zero narrows oproj_reduction_size from
  // num_q_heads * kv_lora_rank to num_q_heads * wuv_v_head_dim and adds the
  // block-diagonal GEMV that produces o_proj's input.
  int wuv_rows_per_wg = params[46];
  int wuv_v_head_dim = params[47];
  int wuv_tiles_per_xcd = params[48];
  bool const unabsorb_v = wuv_rows_per_wg > 0;
  // ── un-absorbed kv_b_k ──
  // Same trade on the Q side: q_b's per-head output narrows from
  // kv_lora_rank + qk_rope to qk_nope + qk_rope, and Phase 3b's block-diagonal
  // GEMV applies W_UK. qb_output_stride then addresses the nope scratch, not
  // the query row, so the head-span assertions below have to follow.
  int qk_nope_head_dim = params[49];
  int wuk_rows_per_wg = params[50];
  int wuk_tiles_per_xcd = params[51];
  bool const unabsorb_k = wuk_rows_per_wg > 0;
  // ── router tile width ──
  // Experts per router call. router_tile_n above is the TILE count, so the
  // two multiply back to num_experts / 8. Appended last so the 52 indices
  // above keep their numbering.
  int router_experts_per_tile = params[52];
  assert(router_experts_per_tile >= 1);
  // ── the router fold ──
  // Non-zero moves the router's gate GEMV and the RMSNorm's sum-of-squares
  // into the o_proj epilogue, reducing both across the ranks on the o_proj
  // all-gather's rendezvous instead of after it. Needs the column shard that
  // all-gather imposes, so it is EP-only, and it adds two trailing inputs.
  int router_fold = params[53];
  assert((!router_fold || ep_world_size > 1) &&
         "the router fold reduces over the o_proj column shard, which only "
         "exists under EP");
  assert(router_tile_n * router_experts_per_tile * 8 == num_experts &&
         "router_tile_n counts tiles, not experts");
  assert(total_router_tiles == router_tile_n * 8);
  bool ep_inline = ep_world_size > 1;
  assert(!ep_tail_only || ep_inline);

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // W_UV's weight and the V row it produces exist only when kv_b_v is
  // un-absorbed. The codegen still emits input_ptrs[29] / output_ptrs[11]
  // unconditionally -- both are inside the fixed task_desc arrays, and the
  // kernel discards them under `if constexpr` -- so the absorbed build passes
  // an unread pointer rather than needing a dummy tensor.
  // The fold's pair is the one trailing group the kernel cannot read
  // unconditionally: it sits past the end of the declared list rather than at
  // a slot the codegen always emits, so ROUTER_FOLD gates the read.
  int num_inputs = (ep_inline ? 29 : 27) + (unabsorb_v ? 1 : 0) +
                   (unabsorb_k ? 1 : 0) + (router_fold ? 2 : 0);
  int num_outputs = 11 + (unabsorb_v ? 1 : 0) + (unabsorb_k ? 1 : 0);
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // ══ attention geometry ══
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->dtensor.dim[0] == batch_size);
  int qkv_reduction = input_ops[0]->dtensor.dim[1];
  // See the same note in the attn-fused registrar: the input row stride is a
  // template argument now, so the narrowed reduction no longer stands in.
  assert(output_ops[0]->dtensor.num_dims == 2);
  int kv_input_stride = output_ops[0]->dtensor.dim[1];
  assert(qkv_output_stride == kv_input_stride);
  assert(qkv_n_wgs_per_xcd * qkv_opw * 8 == kv_input_stride &&
         "the packed qkv_a weight does not cover the row exactly");
  assert(qb_actual_hidden <= qb_reduction && qb_reduction <= kv_input_stride);
  assert(kv_input_offset + kv_lora_rank + qk_rope_head_dim <= kv_input_stride);
  assert(input_ops[3]->dtensor.dim[1] ==
             qkv_opw * (qkv_reduction + qkv_reduction / 32) &&
         "qkv_a MXFP8 weight is not packed at this reduction and row count");
  assert(input_ops[7]->dtensor.dim[1] ==
             qb_opw * (qb_reduction + qb_reduction / 32) &&
         "q_b MXFP8 weight is not packed at this reduction and row count");
  assert(input_ops[12]->dtensor.num_dims == 4);
  assert(input_ops[12]->dtensor.dim[2] == 1);
  int kv_cache_stride = input_ops[12]->dtensor.dim[3];
  assert(kv_cache_stride == kv_lora_rank + qk_rope_head_dim);
  // q_b's per-head output span: the query row itself when W_UK is absorbed,
  // the [nope | rope] scratch when it is not. Everything below is stated
  // against that span rather than against kv_cache_stride, because the two
  // stop coinciding under un-absorption.
  int const qb_head_span =
      unabsorb_k ? (qk_nope_head_dim + qk_rope_head_dim) : kv_cache_stride;
  // Only inside one, not equal to it: the slice is the tail of a head's span,
  // so the kernel offsets by (qb_opw - qk_rope_head_dim). Lets q_b size its
  // tile for the grid-stride makespan instead of inheriting the rope width.
  //
  // Un-absorbed the floor lifts entirely -- the kernel defers the rotation to
  // the far side of Phase 3b's XCD-local W_UK release, which is the barrier a
  // split slice needs, and that switch is compile-time off exactly this
  // comparison (QB_DEFER_ROPE), so the two sides cannot disagree.
  assert((unabsorb_k || qb_opw >= qk_rope_head_dim) &&
         qb_head_span % qb_opw == 0 &&
         "a head's rope slice has to fit inside one q_b workgroup");
  assert((qb_n_wgs_per_xcd * qb_opw) % qb_head_span == 0 &&
         "each XCD's q_b column chunk must hold whole heads");
  // Either the whole nope row, or this rank's 1/world-th of it under the
  // head-sharded q_b. The kernel reads the shard off exactly this ratio -- see
  // qb_tp in gang_mla_attn_fused_mi300.cuh -- so nothing between the two is
  // legal, and qb_output_stride stays the FULL row either way: it is the
  // scratch's declared width, which every rank allocates whole.
  assert((qb_n_wgs_per_xcd * qb_opw * 8 == qb_output_stride ||
          (ep_inline && qb_n_wgs_per_xcd * qb_opw * 8 * ep_world_size ==
                            qb_output_stride)) &&
         "the packed q_b weight covers neither the nope row nor this rank's "
         "1/world-th of it");
  assert(num_q_heads % 16 == 0);
  int num_q_groups = num_q_heads / 16;
  // mla_total_work_items counts (query row, q group, kv chunk) triples, so it
  // carries batch_size as well as the request count: gang_mla_decode_kernel
  // decomposes tile_idx into (q_group, kv_chunk, token, request), with the
  // token dimension inside the request one because a request's rows share its
  // page list. request_id is still a literal 0 here (see the note below), so
  // the quotient IS batch_size rather than merely bounded by it.
  assert(mla_total_work_items % (num_q_groups * num_kv_chunks) == 0);
  assert(mla_total_work_items / (num_q_groups * num_kv_chunks) == batch_size);
  assert(merge_dim_splits >= 1 && kv_lora_rank % merge_dim_splits == 0);
  assert(num_kv_chunks > 1 &&
         "with one chunk the decode writes attn_out directly and there is no "
         "merge phase; the fused layer has no such variant");
  assert(q_workspace_stride == num_q_heads * kv_cache_stride);
  // The merge's consumer is now Phase 9's o_proj, on the other side of an
  // in-kernel barrier rather than an event, and it reduces over the whole row
  // while each XCD writes only its own tiles. A plain store would sit in the
  // producing XCD's L2 where the consumer's buffer_inv cannot reach it.
  assert(merge_write_through &&
         "the fused layer requires MERGE_WRITE_THROUGH; see the header");

  // The residual resolve reads and writes exactly the reduction width.
  assert(input_ops[13]->dtensor.num_dims == 2);
  assert(input_ops[13]->dtensor.dim[0] == batch_size);
  assert(input_ops[13]->dtensor.dim[1] == qkv_reduction);
  assert(qkv_actual_hidden == qkv_reduction &&
         "the residual resolve has no padded-row variant");
  assert(output_ops[5]->dtensor.num_dims == 2);
  assert(output_ops[5]->dtensor.dim[0] == batch_size);
  assert(output_ops[5]->dtensor.dim[1] == qkv_reduction);

  // ══ the merged counter buffer ══
  // 71 cache-line-strided int32 slots, 106 once Phase 8b's barrier is live,
  // 114 once Phase 3b's is too; see the kernel header for the map.
  assert(input_ops[14]->dtensor.num_dims == 1);
  assert(input_ops[14]->dtensor.dim[0] >=
         (unabsorb_k ? 114 : unabsorb_v ? 106 : 71) * 16);

  // ══ MoE geometry ══
  // Two legal row widths, not one: MPK_OPROJ_MXFP4 packs the data half as
  // E2M1 nibbles, halving it, while the E8M0 scale half is one byte per 32
  // elements either way and does not shrink. The layout is otherwise
  // identical, so the format is not recoverable from the shape alone -- but
  // this registration only needs the row to be one of the two, and the
  // compile-time flag that picks the kernel body also picks the packer.
  assert((input_ops[15]->dtensor.dim[1] ==
              oproj_rows_per_wg *
                  (oproj_reduction_size + oproj_reduction_size / 32) ||
          input_ops[15]->dtensor.dim[1] ==
              oproj_rows_per_wg *
                  (oproj_reduction_size / 2 + oproj_reduction_size / 32)) &&
         "o_proj weight is not packed at this reduction and row count");
  // Either the whole row, or this rank's 1/world-th of it under output-wise
  // sharded o_proj followed by the in-kernel all-gather. The kernel detects
  // the shard off exactly this ratio -- there is no flag -- so nothing between
  // the two is legal, and hidden_size stays the FULL row either way: it is the
  // router's K and the all-gather's target width.
  assert((oproj_tiles_per_xcd * oproj_rows_per_wg * 8 == hidden_size ||
          (ep_inline &&
           oproj_tiles_per_xcd * oproj_rows_per_wg * 8 * ep_world_size ==
               hidden_size)) &&
         "the packed o_proj weight covers neither the hidden row nor this "
         "rank's 1/world-th of it");
  // ── W_UV, un-absorbed kv_b_v only ──
  // A block-diagonal GEMV over num_q_heads independent [kv_lora, v_head]
  // blocks, packed as one [num_q_heads * v_head, kv_lora] MXFP8 stack, so the
  // reduction is kv_lora_rank and the output row is oproj_reduction_size.
  if (unabsorb_v) {
    int const wuv_in = ep_inline ? 29 : 27;
    assert(oproj_reduction_size == num_q_heads * wuv_v_head_dim &&
           "un-absorbed o_proj reduces over num_q_heads * v_head_dim");
    assert(wuv_v_head_dim % wuv_rows_per_wg == 0 &&
           "a head's V slice must be a whole number of W_UV workgroups");
    assert(input_ops[wuv_in]->dtensor.dim[1] ==
               wuv_rows_per_wg * (kv_lora_rank + kv_lora_rank / 32) &&
           "W_UV MXFP8 weight is not packed at kv_lora_rank and this row "
           "count");
    // Either the whole V row, or this rank's 1/world-th of it under
    // output-wise sharded W_UV followed by the in-kernel all-gather -- the
    // same relaxation, for the same reason, as the o_proj weight above. The
    // kernel detects the shard off exactly this ratio, so nothing between the
    // two is legal, and oproj_reduction_size stays the FULL row either way.
    assert((wuv_tiles_per_xcd * wuv_rows_per_wg * 8 == oproj_reduction_size ||
            (ep_inline &&
             wuv_tiles_per_xcd * wuv_rows_per_wg * 8 * ep_world_size ==
                 oproj_reduction_size)) &&
           "the packed W_UV weight covers neither the V row nor this rank's "
           "1/world-th of it");
    assert(output_ops[11]->dtensor.num_dims == 2);
    assert(output_ops[11]->dtensor.dim[0] == batch_size);
    assert(output_ops[11]->dtensor.dim[1] == oproj_reduction_size);
  } else {
    assert(wuv_tiles_per_xcd == 0 && wuv_v_head_dim == 0);
    assert(oproj_reduction_size == num_q_heads * kv_lora_rank &&
           "absorbed o_proj reduces over num_q_heads * kv_lora_rank");
  }
  // ── W_UK, un-absorbed kv_b_k only ──
  // The mirror image of the block above: num_q_heads independent
  // [qk_nope, kv_lora] blocks packed as one [num_q_heads * kv_lora, qk_nope]
  // MXFP8 stack, reducing over qk_nope into the query row's latent columns.
  if (unabsorb_k) {
    int const wuk_in = (ep_inline ? 29 : 27) + (unabsorb_v ? 1 : 0);
    int const qnope_out = 11 + (unabsorb_v ? 1 : 0);
    assert(kv_lora_rank % wuk_rows_per_wg == 0 &&
           "a head's absorbed rows must fill whole W_UK workgroups");
    assert(input_ops[wuk_in]->dtensor.dim[1] ==
               wuk_rows_per_wg * (qk_nope_head_dim + qk_nope_head_dim / 32) &&
           "W_UK MXFP8 weight is not packed at qk_nope_head_dim and this row "
           "count");
    // Either every head's latent columns, or this rank's 1/world-th of them
    // under the head shard. The kernel detects the shard off q_b's ratio
    // below -- there is no flag -- and W_UK follows q_b head for head, which
    // the XCD-local-barrier assert further down re-checks.
    assert((wuk_tiles_per_xcd * wuk_rows_per_wg * 8 ==
                num_q_heads * kv_lora_rank ||
            (ep_inline && wuk_tiles_per_xcd * wuk_rows_per_wg * 8 *
                                  ep_world_size ==
                              num_q_heads * kv_lora_rank)) &&
           "the packed W_UK weight covers neither the query row's latent "
           "columns nor this rank's 1/world-th of them");
    // Phase 3b's barrier is XCD-local because the producer and the consumer
    // land on the same heads: q_b's per-XCD chunk and W_UK's per-XCD tile run
    // must describe the same head count.
    assert(wuk_tiles_per_xcd / (kv_lora_rank / wuk_rows_per_wg) ==
               (qb_n_wgs_per_xcd * qb_opw) / qb_head_span &&
           "q_b and W_UK disagree on how many heads live on an XCD");
    assert(output_ops[qnope_out]->dtensor.num_dims == 2);
    assert(output_ops[qnope_out]->dtensor.dim[0] == batch_size);
    assert(output_ops[qnope_out]->dtensor.dim[1] == num_q_heads * qb_head_span);
    assert(qb_output_stride == num_q_heads * qb_head_span &&
           "un-absorbed q_b writes the nope scratch, not the query row");
  } else {
    assert(wuk_tiles_per_xcd == 0 && qk_nope_head_dim == 0);
  }
  assert(hidden_size == qkv_reduction &&
         "o_proj's N is the next layer's qkv_a K; they are one row");
  {
    int const participants = oproj_tiles_per_xcd > router_tile_n
                                 ? oproj_tiles_per_xcd
                                 : router_tile_n;
    assert(total_barrier_arrivals ==
           (participants < tiles_per_xcd ? participants : tiles_per_xcd) * 8);
  }
  assert(input_ops[20]->dtensor.num_dims == 1);
  assert(input_ops[20]->output_tensors[0].dim[0] == num_experts);

  int num_total_experts = output_ops[8]->output_tensors[0].dim[0];
  int num_shared_experts = num_total_experts - num_experts;
  assert(num_shared_experts == 0 || num_shared_experts == 1);
  assert(output_ops[7]->output_tensors[0].dim[1] ==
         topk_k + num_shared_experts);
  assert(output_ops[9]->output_tensors[0].dim[0] == num_total_experts + 1);

  assert(input_ops[22]->output_tensors[0].num_dims == 3);
  assert(input_ops[23]->output_tensors[0].num_dims == 3);
  // What the WEIGHT tensors hold, which under EP is a slice: this rank's
  // num_experts / world_size routed experts, then the shared expert, which
  // every rank stores and exactly one computes. What the KERNEL is templated
  // on is the GLOBAL count, because the activated list and the routing table
  // are replicated and keyed by the global expert id -- d_mask[MOE_NUM_EXPERTS]
  // is the activated-slot count and would read past the list at the local one.
  int moe_num_local_experts = input_ops[22]->output_tensors[0].dim[0];
  int moe_num_experts = num_experts + num_shared_experts;
  assert(num_experts % ep_world_size == 0 &&
         "ep_slice needs the routed expert count to divide by the world size");
  assert(moe_num_local_experts ==
             num_experts / ep_world_size + num_shared_experts &&
         "the expert weight tensor is not this rank's ep_slice");
  assert(input_ops[23]->output_tensors[0].dim[0] == moe_num_local_experts);
  assert(input_ops[24]->output_tensors[0].num_dims == 2);
  assert(input_ops[24]->output_tensors[0].dim[0] == moe_num_local_experts);
  assert(input_ops[24]->output_tensors[0].dim[1] == 2 * moe_intermediate);
  assert(input_ops[25]->output_tensors[0].num_dims == 2);
  assert(input_ops[25]->output_tensors[0].dim[0] == moe_num_local_experts);
  assert(input_ops[25]->output_tensors[0].dim[1] == hidden_size);
  assert(input_ops[26]->output_tensors[0].num_dims == 3);
  assert(input_ops[26]->output_tensors[0].dim[0] == batch_size);
  int moe_num_topk = input_ops[26]->output_tensors[0].dim[1];
  assert(moe_num_topk == topk_k + num_shared_experts);
  assert(input_ops[26]->output_tensors[0].dim[2] == moe_intermediate);
  assert(input_ops[26]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  assert(static_cast<int>(
             static_cast<kn::KNInputOp *>(input_ops[26]->dtensor.owner_op)
                 ->input_strides[1]) == moe_intermediate);
  assert(output_ops[10]->output_tensors[0].num_dims == 2);
  assert(output_ops[10]->output_tensors[0].dim[0] == batch_size);
  assert(output_ops[10]->output_tensors[0].dim[1] == hidden_size);

  // Expert element width, read off the packed workgroup stride: the packing
  // erases N and K, so OPW*(K + K/32) against OPW*(K/2 + K/32) is the only
  // place MXFP8 and MXFP4 differ.
  int const w13_fp8_bytes = moe_w13_opw * (hidden_size + hidden_size / 32);
  int const w13_fp4_bytes = moe_w13_opw * (hidden_size / 2 + hidden_size / 32);
  int const w2_fp8_bytes =
      moe_w2_opw * (moe_intermediate + moe_intermediate / 32);
  int const w2_fp4_bytes =
      moe_w2_opw * (moe_intermediate / 2 + moe_intermediate / 32);
  int const w13_bytes = input_ops[22]->output_tensors[0].dim[2];
  int const w2_bytes = input_ops[23]->output_tensors[0].dim[2];
  bool const moe_fp4 = (w13_bytes == w13_fp4_bytes);
  assert((moe_fp4 ? w13_fp4_bytes : w13_fp8_bytes) == w13_bytes &&
         "the packed W13 weight matches neither the MXFP8 nor the MXFP4 "
         "workgroup stride at this output_per_wg and hidden size");
  assert((moe_fp4 ? w2_fp4_bytes : w2_fp8_bytes) == w2_bytes &&
         "W13 and W2 are packed at different element widths");

  assert(2 * moe_intermediate % moe_w13_opw == 0);
  assert(hidden_size % moe_w2_opw == 0);
  int moe_w13_tiles_per_expert =
      batch_size * (2 * moe_intermediate / moe_w13_opw);
  int moe_w2_tiles_per_expert = batch_size * (hidden_size / moe_w2_opw);

  // ══ the one dispatch width ══
  // Both halves decode xcd_id from tile_idx / tiles_per_xcd, so this is the
  // single number they must agree on. A phase wider than it is not an error:
  // every phase grid-strides by tiles_per_xcd and simply takes more rounds,
  // which is how GLM-5 runs 108 W2 tiles per XCD on 30 workers. What would be
  // an error is a width above the resident worker count, since the barriers
  // count dispatched workers -- but that is the caller's clamp to enforce,
  // and it is not visible from here.
  assert(tiles_per_xcd > 0);

  float scale_s = 1.0f / sqrtf((float)qk_head_dim) * 1.44269504088896340736f;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_mla_full_layer_fused_kernel_mi300<$, $, $, $, $, $, $, "
         "$, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, $, "
         "$, $, $, $, $, $, $, $, $, $, $, $, $, $>(",
         batch_size,
         qkv_opw,
         qkv_reduction,
         qkv_actual_hidden,
         qb_opw,
         qb_reduction,
         qb_actual_hidden,
         kv_lora_rank,
         qk_rope_head_dim,
         kv_input_stride,
         kv_cache_stride,
         max_seq_len,
         page_size,
         kv_input_offset,
         num_q_heads,
         num_kv_chunks,
         q_workspace_stride,
         merge_dim_splits,
         merge_write_through ? "true" : "false",
         oproj_reduction_size,
         oproj_rows_per_wg,
         hidden_size,
         actual_hidden_dim,
         num_experts,
         topk_k,
         moe_intermediate,
         moe_num_experts,
         moe_num_topk,
         moe_w13_tiles_per_expert,
         moe_w2_tiles_per_expert,
         moe_w13_opw,
         moe_w2_opw,
         moe_fp4 ? "true" : "false",
         ep_world_size,
         ep_my_pe,
         ep_fold_pe,
         ep_tail_only ? "true" : "false",
         wuv_rows_per_wg,
         wuv_v_head_dim,
         qk_nope_head_dim,
         wuk_rows_per_wg,
         router_experts_per_tile,
         router_fold ? "true" : "false");
  code.e("    task_desc->input_ptrs,");
  code.e("    task_desc->output_ptrs,");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $,", qkv_n_wgs_per_xcd);
  code.e("    $,", qkv_output_stride);
  code.e("    $,", qb_n_wgs_per_xcd);
  code.e("    $,", qb_output_stride);
  code.e("    $,", mla_tiles_per_xcd);
  code.e("    $,", mla_total_work_items);
  code.e("    $,", merge_tiles_per_xcd);
  code.e("    $f,", scale_s);
  // Matches the standalone kvupd task's epsilon.
  code.e("    1e-6f,");
  code.e("    $,", oproj_tiles_per_xcd);
  code.e("    $,", router_tile_n);
  code.e("    $,", total_barrier_arrivals);
  code.e("    $,", total_router_tiles);
  code.e("    $,", norm_topk_prob != 0 ? "true" : "false");
  code.e("    $ / 1000.0f,", scaling_milli);
  code.e("    $,", num_shared_experts);
  code.e("    $,", moe_w13_tiles_per_xcd);
  code.e("    $,", moe_w2_tiles_per_xcd);
  code.e("    $,", wuv_tiles_per_xcd);
  code.e("    $,", wuk_tiles_per_xcd);
  code.e("    $,", tiles_per_xcd);
  code.e("    tile_idx,");
  // Multi-layer mode (task #14). ml_num_layers is 0 whenever the scheduler is
  // dispatching one task per layer, which is what the kernel's snapshot path
  // keys off; _linear_reserved carries the monotonic layer index the batched
  // loop publishes before each layer. Both live behind MPK_PRECOMPUTED_DISPATCH
  // -- RuntimeConfig does not declare ml_num_layers without it -- so the
  // fallback has to be a literal.
  code.e("#if defined(MPK_PRECOMPUTED_DISPATCH) && "
         "defined(MPK_FUSED_LAYER_BATCHING)");
  code.e("    runtime_config.ml_num_layers,");
  code.e("#else");
  code.e("    0,");
  code.e("#endif");
  code.e("    (int)task_desc->task_metadata._linear_reserved);");
  return register_task_variant(TASK_GANG_MLA_FULL_LAYER_FUSED_MI300,
                               code.to_string());
}

// Gang merge split-KV: 8 tasks (1 per XCD), tile_idx → (request_id, kv_head)
// params: [num_qo_heads_per_kv, head_dim, max_seq_len, page_size,
//          num_kv_heads, total_work_items_per_xcd, total_work_items]
int TaskRegister::register_gang_attn_merge_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int num_qo_heads_per_kv = params[0];
  int head_dim = params[1];
  int max_seq_len = params[2];
  int page_size = params[3];
  int num_kv_heads = params[4];
  int total_work_items = params[6];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 128;
  int merge_output_head_offset = num_qo_heads_per_kv * head_dim;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_attention_merge_kernel<bfloat16,");
  code.e("    $, $, $, $, $, $, $, $>(",
         num_qo_heads_per_kv,
         num_kv_heads,
         head_dim,
         max_tokens,
         ((max_seq_len + SEQ_LEN_PER_BLOCK - 1) /
          SEQ_LEN_PER_BLOCK), // NUM_KV_CHUNKS
         SEQ_LEN_PER_BLOCK,   // KV_CHUNK_SIZE
         page_size,
         merge_output_head_offset);
  code.e("    task_desc->input_ptrs[0],"); // lse (full)
  code.e("    task_desc->input_ptrs[1],"); // output_tmp (full)
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->output_ptrs[0],"); // output (full)
  code.e("    $,", num_kv_heads);
  code.e("    $,", total_work_items);
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_ATTN_MERGE_MI300, code.to_string());
}

int TaskRegister::register_splitk_reduce_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 1); // params[0] = K_SPLITS
  int k_splits = params[0];
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;  // workspace (float32) + residual (bf16)
  int num_outputs = 1; // output (bf16)

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [BS, N_per_task]
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int output_size = output_ops[0]->output_tensors[0].dim[1];

  // Output stride (bf16)
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_output_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  int output_stride = static_cast<int>(kn_output_op->input_strides[0]);

  // Workspace stride (float32) - row stride in float elements
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_ws_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  int ws_stride = static_cast<int>(kn_ws_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::splitk_reduce<bfloat16, $, $, $, $, $>(",
         batch_size,
         output_size,
         k_splits,
         ws_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_SPLITK_REDUCE_MI300, code.to_string());
}

int TaskRegister::register_splitk_linear_res_atomic_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0] = K_SPLITS
  assert(params.size() == 1);
  int k_splits = params[0];

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // Inputs: 0=input, 1=weight, 2=residual, 3=workspace(float32),
  // 4=done_counter(int32) Outputs: 5=output(bf16)
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // input[0]: [batch, K/K_SPLITS] — reduction_size per K-split
  assert(input_ops[0]->dtensor.num_dims == 2);
  int reduction_size = input_ops[0]->dtensor.dim[1];

  // output: [batch, NPerBlock]
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int n_per_block = output_ops[0]->output_tensors[0].dim[1];

  // Output stride (bf16)
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_output_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  int output_stride = static_cast<int>(kn_output_op->input_strides[0]);

  // Workspace stride (float32)
  assert(input_ops[3]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_ws_op =
      static_cast<kn::KNInputOp *>(input_ops[3]->dtensor.owner_op);
  int ws_stride = static_cast<int>(kn_ws_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::splitk_linear_res_atomic<bfloat16, $, $, $, $>(",
         batch_size,
         n_per_block,
         reduction_size,
         k_splits);
  code.e("    task_desc->input_ptrs[0],");       // input
  code.e("    task_desc->input_ptrs[1],");       // weight
  code.e("    task_desc->input_ptrs[2],");       // residual
  code.e("    task_desc->input_ptrs[3],");       // workspace (float32)
  code.e("    task_desc->output_ptrs[0],");      // output (bf16)
  code.e("    (int*)task_desc->input_ptrs[4],"); // done_counter (int32)
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    $, $);", ws_stride, output_stride);
  return register_task_variant(TASK_SPLITK_LINEAR_RES_ATOMIC_MI300,
                               code.to_string());
}

int TaskRegister::register_argmax_partial_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  // params[0]: num_partial_tasks
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_elements = input_ops[0]->output_tensors[0].dim[1];
  int num_partial_tasks = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_partial_kernel<bfloat16, $, $, $>(",
         batch_size,
         num_elements,
         num_partial_tasks);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->output_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_ARGMAX_PARTIAL, code.to_string());
}

int TaskRegister::register_argmax_reduce_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params) {
  // params[0]: output size
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_reduce_kernel<bfloat16, $, $, $>(",
         batch_size,
         params[0],
         num_parts);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_ARGMAX_REDUCE, code.to_string());
}

// Cross-rank variant of the above, for the vocab-sharded LM head. Same local
// reduce, then a four-way peer exchange of one 64-bit (value, index) word.
// Registered under TASK_ARGMAX_REDUCE as a second variant -- the dispatch is
// (task_type, variant_id), and nothing else keys off the type.
int TaskRegister::register_argmax_reduce_xrank_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: chunk size (the partial task's per-task span)
  // params[1]: EP world size
  // params[2]: this rank's PE id
  // params[3]: vocab rows owned by one rank
  assert(params.size() == 4);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_reduce_xrank_kernel<bfloat16, $, $, $, $, $, $>(",
         batch_size,
         params[0],
         num_parts,
         params[1],
         params[2],
         params[3]);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    runtime_config.step[0]);");
  return register_task_variant(TASK_ARGMAX_REDUCE, code.to_string());
}

// Cross-rank sum of one hidden-width partial, plus a residual add. Used by the
// intermediate-sharded dense MLP: every rank owns a slice of the intermediate
// dim, so down_proj is K-parallel and each rank holds a partial hidden vector.
// Also rides TASK_ARGMAX_REDUCE -- the dispatch is (task_type, variant_id).
int TaskRegister::register_xrank_sum_add_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params) {
  // params[0]: hidden size
  // params[1]: EP world size
  // params[2]: this rank's PE id
  assert(params.size() == 3);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_outputs = 1;
  // A 4th input is a DEPENDENCY-ONLY tensor and is never read by the kernel.
  // The task graph is a linear chain and every edge is proved by a shared
  // tensor guid (runtime.cc's `num_shared_tensors >= 1`), so a producer that
  // writes this rank's COLUMN SLICE of `partial` -- a different DTensor over
  // the same storage -- has no guid in common with us. Passing the slice as a
  // trailing input is what makes that edge visible.
  assert(bgraph.operators.size() == 4 || bgraph.operators.size() == 5);
  int num_inputs = (int)bgraph.operators.size() - num_outputs;

  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::xrank_sum_add_kernel<bfloat16, $, $, $, $>(",
         batch_size,
         params[0],
         params[1],
         params[2]);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS],");
  code.e("    runtime_config.step[0]);");
  return register_task_variant(TASK_ARGMAX_REDUCE, code.to_string());
}

int TaskRegister::register_reduce_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params) {
  // Currently, allreduce task is split to two sub-tasks: allgather + reduce
  // params[0]: num_gpus
  // params[1]: my_gpu_id
  assert(params.size() == 2);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // For now, the memory partition of the input[0] results in a strided
  // 2D tensor, which cannot be directly transferred by a single nvshmem
  // memput. So we use for loop to iterate over the first dim and transfer each
  // row. If the upperlayer changes this layout, this "for-loop" method can
  // fail. So we assert it here just in case.
  assert(input_ops[0]->input_map.x == 1 && input_ops[0]->input_map.y == -1 &&
         input_ops[0]->input_map.z == -1);
  // Currently support 2D reduction, buffer has an extra world_size dim
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int output_size = input_ops[0]->output_tensors[0].dim[1];
  // get output stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  int input_stride = static_cast<int>(kn_input_op->input_strides[0]);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  int output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  assert(input_stride == output_stride);
  // Register nvshmem copy task (allgather)
  mirage::transpiler::CodeKeeper c;
  c.inc_indent();
  c.e("size_t event_index = "
      "get_event_position_index(task_desc->trigger_event);");
  c.inc_indent();
  c.e("int gpu_id = "
      "static_cast<int>(get_event_gpu_id(task_desc->trigger_event));");
  c.e("assert(gpu_id < runtime_config.num_gpus);");
  c.e("assert(gpu_id != runtime_config.my_gpu_id);");
  c.e("for (int i = 0; i < $; i++) {", batch_size);
  c.e("  mpk_putmem_signal_block(");
  c.e("      reinterpret_cast<char*>(task_desc->output_ptrs[0]) + i * $ * "
      "sizeof(bfloat16),",
      input_stride);
  c.e("      reinterpret_cast<char*>(task_desc->input_ptrs[0]) + i * $ * "
      "sizeof(bfloat16),",
      output_stride);
  c.e("      task_desc->task_metadata.xfer_size_in_bytes / $,", batch_size);
  c.e("      reinterpret_cast<uint64_t "
      "*>(&runtime_config.all_event_counters[event_index]),");
  c.e("      1 /*signal*/,");
  c.e("      MPK_SIGNAL_ADD,");
  c.e("      gpu_id);");
  c.e("}");
  register_task_variant(TASK_NVSHMEM_COPY, c.to_string());
  // Register reduction kernel
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::reduction_kernel<bfloat16, $, $, $, $, $>(",
         params[0],
         params[1],
         batch_size,
         output_size,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_REDUCE, code.to_string());
}

int TaskRegister::register_find_ngram_partial_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: ngram size
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int num_parts = output_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::find_ngram_partial_kernel<$, $>(", params[0], num_parts);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.step[0] + 1);");

  return register_task_variant(TASK_FIND_NGRAM_PARTIAL, code.to_string());
}

int TaskRegister::register_find_ngram_global_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: ngram size
  // params[1]: spec length
  assert(params.size() == 2);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::find_ngram_global_kernel<$, $, $>(",
         params[0],
         params[1],
         num_parts);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.step[0]);");
  return register_task_variant(TASK_FIND_NGRAM_GLOBAL, code.to_string());
}

int TaskRegister::register_target_verify_greedy_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int num_spec_tokens = input_ops[0]->output_tensors[0].dim[1] - 1;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::target_verify_greedy_kernel<$>(", num_spec_tokens);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    (void*)(runtime_config.new_token_nums),"); // int pointer
  code.e("    (void*)(runtime_config.tokens + runtime_config.step[0] + 1));");
  return register_task_variant(TASK_TARGET_VERIFY_GREEDY, code.to_string());
}

int TaskRegister::register_linear_hopper_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params,
                                              bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 128;
  int const Kstages = output_size >= 256 ? 3 : 6;
  int const SMEM_M_SIZE = batch_size;
  // int const SMEM_M_SIZE = 64;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = (output_size >= 256)   ? 256
                               : (output_size >= 128) ? 128
                               : (output_size >= 64)  ? 64
                               : (output_size >= 32)  ? 32
                                                      : 16;
  code.e("using TMA_A = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_B = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  if (with_residual) {
    code.e(
        "using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, "
        "$, $, $, $, $, $, true>;",
        B,
        M,
        S,
        batch_size,         /*GMEM_ROW_*/
        output_size,        /*GMEM_COL_*/
        batch_size,         /*SMEM_ROW_*/
        output_tma_cp_size, /*SMEM_COL_*/
        output_stride,      /*GMEM_STRIDE_ROW_*/
        1,                  /*GMEM_STRIDE_COL_*/
        1,                  /*SMEM_REPEAT_ROW_*/
        (output_atom_size + output_tma_cp_size - 1) /
            output_tma_cp_size,         /*SMEM_REPEAT_COL_*/
        SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
    );
  }

  code.e("using TMA_OUT = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, "
         "$, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,         /*GMEM_ROW_*/
         output_size,        /*GMEM_COL_*/
         batch_size,         /*SMEM_ROW_*/
         output_tma_cp_size, /*SMEM_COL_*/
         output_stride,      /*GMEM_STRIDE_ROW_*/
         1,                  /*GMEM_STRIDE_COL_*/
         1,                  /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size,         /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  if (with_residual) {
    code.e("TMA_RESIDUAL "
           "tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_"
           "ptrs[2][0]));");
  }
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");
  // code.e("printf(\"linear_kernel_hopper start\");");

  code.e("kernel::linear_kernel_hopper<bfloat16, $, $, $, $, TMA_A, TMA_B, "
         "TMA_OUT, $, $>(",
         batch_size,
         output_size,
         reduction_size,
         Kstages,
         with_residual ? "TMA_RESIDUAL" : "void",
         output_stride);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    tma_out, ");
  if (with_residual) {
    code.e("    &tma_residual");
  } else {
    code.e("    nullptr");
  }
  code.e(");");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_WITH_RESIDUAL_HOPPER,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_HOPPER, code.to_string());
  }
}
int TaskRegister::register_paged_attention_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  assert(params.size() == 6);

  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if ((int)input_ops.size() < num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  // Shapes/strides
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int num_q_heads_per_kv = num_q_heads / num_kv_heads;
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int max_tokens = input_ops[0]->dtensor.dim[0];

  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();

  constexpr int B = 3, M = 3, S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int KV_TILE_SIZE = 64;
  int const qkv_rows = num_q_heads_per_kv + 2;
  int const smem_repeat_col =
      (head_dim + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  int const q_smem_stride = max_tokens * num_q_heads_per_kv * TMA_CP_ASYNC_SIZE;
  int const kv_smem_stride = KV_TILE_SIZE * TMA_CP_ASYNC_SIZE;
  int const non_cached_kv_smem_stride = max_tokens * TMA_CP_ASYNC_SIZE;
  int const num_pages = (max_seq_len + page_size - 1) / page_size;
  int const num_head_group = qkv_stride / head_dim / (num_q_heads_per_kv + 2);

  // code.e("using TMA_Q = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $, $, $,
  // "
  //        "$, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        max_tokens,         /* GMEM_DEPTH */
  //        qkv_rows,           /* GMEM_ROW   */
  //        head_dim,           /* GMEM_COL   */
  //        max_tokens,         /* SMEM_DEPTH */
  //        num_q_heads_per_kv, /* SMEM_ROW   */
  //        TMA_CP_ASYNC_SIZE,  /* SMEM_COL   */
  //        qkv_stride,         /* GMEM_STRIDE_DEPTH */
  //        head_dim,           /* GMEM_STRIDE_ROW   */
  //        1,                  /* GMEM_STRIDE_COL   */
  //        1,                  /* SMEM_REPEAT_ROW   */
  //        smem_repeat_col,    /* SMEM_REPEAT_COL   */
  //        q_smem_stride       /* SMEM_STRIDE       */
  // );

  // code.e("using TMA_KV = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $, $,
  // $, "
  //        "$, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        max_tokens,               /* GMEM_DEPTH */
  //        qkv_rows,                 /* GMEM_ROW   */
  //        head_dim,                 /* GMEM_COL   */
  //        max_tokens,               /* SMEM_DEPTH */
  //        1,                        /* SMEM_ROW   */
  //        TMA_CP_ASYNC_SIZE,        /* SMEM_COL   */
  //        qkv_stride,               /* GMEM_STRIDE_DEPTH */
  //        head_dim,                 /* GMEM_STRIDE_ROW   */
  //        1,                        /* GMEM_STRIDE_COL   */
  //        1,                        /* SMEM_REPEAT_ROW   */
  //        smem_repeat_col,          /* SMEM_REPEAT_COL   */
  //        non_cached_kv_smem_stride /* SMEM_STRIDE       */
  // );

  // code.e("using TMA_PAGED_KV_CACHE = kernel::tma::tma_4d<bfloat16, $, $, $,
  // $, "
  //        "$, $, $, $, $, $, $, $, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        num_pages,                             /* GMEM_OUTERMOST_ */
  //        page_size,                             /* GMEM_DEPTH   */
  //        num_head_group,                        /* GMEM_ROW   */
  //        head_dim,                              /* GMEM_COL   */
  //        1,                                     /* SMEM_OUTERMOST_ */
  //        KV_TILE_SIZE,                          /* SMEM_DEPTH   */
  //        num_q_heads_per_kv,                    /* SMEM_ROW   */
  //        TMA_CP_ASYNC_SIZE,                     /* SMEM_COL   */
  //        page_size * head_dim * num_head_group, /* GMEM_STRIDE_OUTERMOST_ */
  //        page_size * head_dim,                  /* GMEM_STRIDE_DEPTH */
  //        head_dim,                              /* GMEM_STRIDE_ROW   */
  //        1,                                     /* GMEM_STRIDE_COL   */
  //        1,                                     /* SMEM_REPEAT_ROW   */
  //        smem_repeat_col,                       /* SMEM_REPEAT_COL   */
  //        kv_smem_stride                         /* SMEM_STRIDE       */
  // );

  // code.e("using TMA_OUTPUT = kernel::tma::tma_3d<bfloat16, $, $, $, $, $, $,
  // "
  //        "$, $, $, $, $, $, $, $, $, true>;",
  //        B,
  //        M,
  //        S,
  //        max_tokens,
  //        num_q_heads_per_kv * num_head_group,
  //        head_dim,
  //        max_tokens,
  //        num_q_heads_per_kv,
  //        TMA_CP_ASYNC_SIZE,
  //        head_dim * num_head_group * num_head_group,
  //        head_dim,
  //        1,
  //        1,
  //        smem_repeat_col,
  //        max_tokens * num_q_heads_per_kv * TMA_CP_ASYNC_SIZE);

  // code.e("TMA_Q  tma_q "
  //        "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0]));");
  // code.e("TMA_KV tma_k "
  //        "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][1]));");
  // code.e("TMA_KV tma_v "
  //        "(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][2]));");

  // code.e("TMA_PAGED_KV_CACHE "
  //        "tma_paged_k_cache(static_cast<CUtensorMap*>(task_desc->input_tma_"
  //        "desc_ptrs[1][0]));");
  // code.e("TMA_PAGED_KV_CACHE "
  //        "tma_paged_v_cache(static_cast<CUtensorMap*>(task_desc->input_tma_"
  //        "desc_ptrs[2][0]));");

  // code.e("TMA_OUTPUT "
  //        "tma_output(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs["
  //        "0][0]));");

  code.e("kernel::multitoken_paged_attention_hopper_impl<bfloat16, $, $, $, $, "
         "$, $, $, $, $, "
         "$, $, $, $>(",
         num_q_heads_per_kv, /* NUM_QO_HEADS               */
         1,                  /* NUM_KV_HEADS               */
         num_kv_heads,       /* NUM_QO_GROUPS              */
         kv_stride,          /* KV_CACHE_STRIDE            */
         qkv_stride,         /* QKV_STRIDE                 */
         output_size,        /* O_STRIDE (= num_q_heads*head_dim) */
         head_dim,           /* HEAD_DIM                   */
         -1,          /* SEQ_LEN (not used for non-split KV tasks)          */
         max_seq_len, /* MAX_SEQ_LEN                */
         page_size,   /* PAGE_SIZE                  */
         max_tokens,  /* MAX_TOKENS                 */
         "false",     /* PARTITION_KV               */
         1            /* NUM_KV_CHUNKS              */
  );
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0); // qk_norm
  code.e("    $,", params[3] > 0); // rope
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    nullptr,"); // lse, not used for non-split KV tasks
  code.e("    0);");      // kv_idx, not used for non-split KV tasks

  return register_task_variant(TASK_PAGED_ATTENTION_HOPPER, code.to_string());
}

int TaskRegister::register_rmsnorm_hopper_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int hidden_dim = output_ops[0]->output_tensors[0].dim[1];

  // Currently assume that each rmsnorm task processes one token
  // assert(batch_size == 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
  assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::rms_norm_hopper_impl<bfloat16, $, $>(", batch_size, hidden_dim);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    1e-6f);");
  return register_task_variant(TASK_RMS_NORM_HOPPER, code.to_string());
}

int TaskRegister::register_linear_swapAB_hopper_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 128;
  constexpr int Kstages = 5;
  assert(batch_size <= 16);
  int const SMEM_M_SIZE = batch_size <= 8 ? 8 : 16;
  // int const SMEM_M_SIZE = 16;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = 64;
  code.e("using TMA_B = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_A = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  if (with_residual) {
    code.e(
        "using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, "
        "$, $, $, $, $, $, true>;",
        0,
        0,
        0,
        batch_size,                      /*GMEM_ROW_*/
        output_size,                     /*GMEM_COL_*/
        batch_size,                      /*SMEM_ROW_*/
        output_tma_cp_size,              /*SMEM_COL_*/
        output_stride,                   /*GMEM_STRIDE_ROW_*/
        1,                               /*GMEM_STRIDE_COL_*/
        1,                               /*SMEM_REPEAT_ROW_*/
        1,                               /*SMEM_REPEAT_COL_*/
        SMEM_M_SIZE * output_tma_cp_size /*SMEM_STRIDE_*/
    );
  }

  code.e("using TMA_OUT = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, "
         "$, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,                      /*GMEM_ROW_*/
         output_size,                     /*GMEM_COL_*/
         batch_size,                      /*SMEM_ROW_*/
         output_tma_cp_size,              /*SMEM_COL_*/
         output_stride,                   /*GMEM_STRIDE_ROW_*/
         1,                               /*GMEM_STRIDE_COL_*/
         1,                               /*SMEM_REPEAT_ROW_*/
         1,                               /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * output_tma_cp_size /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  if (with_residual) {
    code.e("TMA_RESIDUAL "
           "tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_"
           "ptrs[2][0]));");
  }
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");

  code.e(
      "kernel::linear_swapAB_kernel_hopper<bfloat16, $, $, $, $, TMA_A, TMA_B, "
      "TMA_OUT, $, $, $>(",
      batch_size,
      output_size,
      reduction_size,
      Kstages,
      with_residual ? "TMA_RESIDUAL" : "void",
      output_stride,
      "false" /*SplitK*/);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    tma_out, ");
  if (with_residual) {
    code.e("    &tma_residual,");
    code.e("    runtime_config.my_gpu_id == 0");
  } else {
    code.e("    nullptr,");
    code.e("    false/*residual*/");
  }

  code.e(");");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_SWAPAB_HOPPER, code.to_string());
  }
}

int TaskRegister::register_linear_cutlass_hopper_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;
  constexpr int KSTAGES = 4;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  constexpr int TILE_SIZE = 128;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // NOTE: output_size and batch_size are swapped here
  code.e("auto problem_shape = cute::Shape<cute::Int<$>, cute::Int<$>, "
         "cute::Int<$>>{};",
         output_size,
         batch_size,
         reduction_size);
  // NOTE: output_size and batch_size are swapped here
  code.e("using KernelTraits = kernel::MMAKernelTraits<cutlass::bfloat16_t, $, "
         "$, $, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, "
         "cutlass::layout::RowMajor, cutlass::layout::RowMajor, $, $, $, $, "
         "decltype(problem_shape), $, $>;",
         output_size,
         batch_size,
         reduction_size,
         8,
         64,
         batch_size,
         TILE_SIZE,
         batch_size,
         KSTAGES);
  code.e("using Mainloop = kernel::CollectiveMainloop<KernelTraits>;");
  code.e("using Epilogue = kernel::CollectiveEpilogue<KernelTraits>;");
  // code.e("using StrideA = typename KernelTraits::StrideA;");
  // code.e("using StrideB = typename KernelTraits::StrideB;");
  // code.e("using StrideC = typename KernelTraits::StrideC;");
  // code.e("using StrideD = typename KernelTraits::StrideD;");
  // code.e("StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, "
  //        "{KernelTraits::OUTPUT_SIZE, KernelTraits::REDUCTION_SIZE, 1});");
  // code.e("StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, "
  //        "{KernelTraits::BATCH_SIZE, KernelTraits::REDUCTION_SIZE, 1});");
  // code.e("StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, "
  //        "{KernelTraits::BATCH_SIZE, KernelTraits::OUTPUT_SIZE, 1});");
  // code.e("StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, "
  //        "{KernelTraits::BATCH_SIZE, KernelTraits::OUTPUT_SIZE, 1});");
  // code.e("typename Mainloop::Arguments mainloop_args{");
  // code.e("    static_cast<cutlass::bfloat16_t const "
  //        "*>(task_desc.inputs[1].base_ptr),");
  // code.e("    stride_A,");
  // code.e("    static_cast<cutlass::bfloat16_t const "
  //        "*>(task_desc.inputs[0].base_ptr),");
  // code.e("    stride_B,");
  // code.e("};");
  // code.e("typename Epilogue::Arguments epilogue_args{");
  // code.e("    static_cast<cutlass::bfloat16_t const "
  //        "*>(task_desc.inputs[2].base_ptr),");
  // code.e("    stride_C,");
  // code.e(
  //     "    static_cast<cutlass::bfloat16_t
  //     *>(task_desc.outputs[0].base_ptr),");
  // code.e("    stride_C,");
  // code.e("    {1.0f, 1.0f},");
  // code.e("};");
  // code.e("using MainloopParamsDevice = typename Mainloop::template "
  //        "Params<false>;");
  // code.e("MainloopParamsDevice mainloop_params = "
  //        "Mainloop::to_underlying_arguments<false>(problem_shape, "
  //        "mainloop_args);");
  // code.e("typename Epilogue::Params epilogue_params = "
  //        "Epilogue::to_underlying_arguments(problem_shape, epilogue_args);");

  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int Kstages = 5;
  assert(batch_size <= 16);
  int const SMEM_M_SIZE = batch_size;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = 64;

  code.e("using TMA_B = kernel::tma::tma_2d<cutlass::bfloat16_t, $, $, $, $, "
         "$, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_A = kernel::tma::tma_2d<cutlass::bfloat16_t, $, $, $, $, "
         "$, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");

  code.e("kernel::linear_cutlass_ws_hopper<Mainloop, Epilogue, false, "
         "cutlass::bfloat16_t, $, $, $, TMA_A, TMA_B, "
         "$, $>(",
         batch_size,
         output_size,
         reduction_size,
         output_stride,
         with_residual);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->input_ptrs[2]");
  code.e(");");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_CUTLASS_HOPPER, code.to_string());
  }
}

int TaskRegister::register_silu_mul_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, input_stride, output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[1] == output_size * 2);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[1];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_task_impl_hopper<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_SILU_MUL_HOPPER, code.to_string());
}

int TaskRegister::register_embedding_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 1);
  // params[0]: input source (0: tokens, 1: input_token)
  int batch_size = 0, output_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::embedding_kernel_hopper<bfloat16, $, $, $>(",
         batch_size,
         output_size,
         output_stride);
  if (params[0] == 0) {
    code.e("    runtime_config.tokens + runtime_config.step[0], ");
  } else if (params[0] == 1) {
    code.e("    task_desc->input_ptrs[0],");
  }
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_EMBEDDING_HOPPER, code.to_string());
}

// SM100 Tasks
int TaskRegister::register_linear_sm100_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params,
                                             bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define MMA
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int bM = 128;
  constexpr int bN = MMA_N;
  constexpr int bK = 64;
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int num_tmem_columns = bN * num_acc_stages;
  assert(num_tmem_columns <= 512);
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  int const output_tma_cp_size = 128;
  int const output_atom_size = 128;
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_M,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_N,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_N * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, "
         "$, $, $, $, $, true>;",
         0,
         M,
         S,
         batch_size,    /*GMEM_ROW_*/
         output_size,   /*GMEM_COL_*/
         MMA_N,         /*SMEM_ROW_*/
         MMA_M,         /*SMEM_COL_*/
         output_stride, /*GMEM_STRIDE_ROW_*/
         1,             /*GMEM_STRIDE_COL_*/
         1,             /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size, /*SMEM_REPEAT_COL_*/
         MMA_N * MMA_M           /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");
  // Bias Tensor setup
  code.e("cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         batch_size,
         output_size,
         output_stride);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         with_residual ? "task_desc->input_ptrs[2]" : "nullptr");
  code.e("kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, "
         "decltype(mBias), TMA_OUT, "
         "$, $, $, $, $, $, $, "
         "$, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         reduction_size,
         with_residual ? "false" : "true",
         /*SplitK=*/"false",
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    mBias,");
  code.e("    tma_out); ");

  if (with_residual) {
    return register_task_variant(TASK_LINEAR_WITH_RESIDUAL_SM100,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_LINEAR_SM100, code.to_string());
  }
}

int TaskRegister::register_splitk_linear_sm100_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0,
      reduction_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->output_tensors[0].dim[1];
  reduction_stride = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define MMA
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int bM = 128;
  constexpr int bN = MMA_N;
  constexpr int bK = 64;
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int num_tmem_columns = bN * num_acc_stages;
  assert(num_tmem_columns <= 512);
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  int const output_tma_cp_size = 128;
  int const output_atom_size = 128;
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_M,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_stride,  /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_B = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         MMA_N,             /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_stride,  /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_N * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_OUT = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, "
         "$, $, $, $, $, true>;",
         0,
         M,
         S,
         batch_size,    /*GMEM_ROW_*/
         output_size,   /*GMEM_COL_*/
         MMA_N,         /*SMEM_ROW_*/
         MMA_M,         /*SMEM_COL_*/
         output_stride, /*GMEM_STRIDE_ROW_*/
         1,             /*GMEM_STRIDE_COL_*/
         1,             /*SMEM_REPEAT_ROW_*/
         (output_atom_size + output_tma_cp_size - 1) /
             output_tma_cp_size, /*SMEM_REPEAT_COL_*/
         MMA_N * MMA_M           /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");
  // Bias Tensor setup
  code.e("cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         batch_size,
         output_size,
         output_stride);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         with_residual ? "task_desc->input_ptrs[2]" : "nullptr");
  code.e("kernel::linear_sm100_mpk_task_impl<cute::bfloat16_t, TMA_A, TMA_B, "
         "decltype(mBias), TMA_OUT, "
         "$, $, $, $, $, $, $, "
         "$, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         reduction_size,
         with_residual ? "false" : "true",
         /*SplitK=*/"true",
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    mBias,");
  code.e("    tma_out); ");

  return register_task_variant(TASK_SPLITK_LINEAR_SM100, code.to_string());
}

int TaskRegister::register_paged_attention_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  assert(params.size() == 6);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = output_size / num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_sm100_task_impl<bfloat16, $, $, "
         "$, $, "
         "$, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         kv_stride,
         qkv_stride,
         output_size,
         head_dim,
         max_seq_len,
         page_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_ATTN_SM100, code.to_string());
}

int TaskRegister::register_argmax_partial_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_partial_tasks
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_elements = input_ops[0]->output_tensors[0].dim[1];
  int num_partial_tasks = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_partial_sm100_kernel<bfloat16, $, $, $>(",
         batch_size,
         num_elements,
         num_partial_tasks);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->output_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_ARGMAX_PARTIAL_SM100, code.to_string());
}

int TaskRegister::register_argmax_reduce_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: output size
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int num_parts = input_ops[0]->output_tensors[0].dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::argmax_reduce_sm100_kernel<bfloat16, $, $, $>(",
         batch_size,
         params[0],
         num_parts);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);");
  return register_task_variant(TASK_ARGMAX_REDUCE_SM100, code.to_string());
}

int TaskRegister::register_sampling_sm100_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  // params[0]: seed
  assert(params.size() == 1);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = input_ops[0]->output_tensors[0].dim[0];
  int vocab_size = input_ops[0]->output_tensors[0].dim[1];
  int seed = params[0];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::sampling_from_logits_kernel<256, 4, bfloat16, int>(");
  code.e("    static_cast<bfloat16*>(task_desc->input_ptrs[0]),");
  code.e("    static_cast<int*>(task_desc->output_ptrs[0]),");
  code.e("    $,", vocab_size);
  code.e("    $,", seed);
  code.e("    0,  // philox_offset");
  code.e("    $);", batch_size);
  return register_task_variant(TASK_SAMPLING_SM100, code.to_string());
}

int TaskRegister::register_tensor_init_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, output_size, output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(input_ops[0]->dtensor.num_dims == 2);
  batch_size = input_ops[0]->output_tensors[0].dim[0];
  output_size = input_ops[0]->output_tensors[0].dim[1];
  // get input stride
  output_stride = input_ops[0]->dtensor.dim[1];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::tensor_init_sm100_task_impl<cute::bfloat16_t, $, $, $>(",
         /*BATCH_SIZE=*/batch_size,
         /*OUTPUT_SIZE=*/output_size,
         /*OUTPUT_STRIDE=*/output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    0);");
  return register_task_variant(TASK_TENSOR_INIT, code.to_string());
}

int TaskRegister::register_moe_topk_softmax_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts = 0, num_experts_per_tok = 0, input_stride,
      output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 3;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  assert(output_ops[1]->output_tensors[0].num_dims == 2);
  assert(output_ops[2]->output_tensors[0].num_dims == 1);
  num_experts = output_ops[1]->output_tensors[0].dim[0];
  batch_size = output_ops[1]->output_tensors[0].dim[1];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(output_ops[2]->output_tensors[0].dim[0] == num_experts + 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[1] == num_experts);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[1];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::topk_softmax_task_impl<cute::bfloat16_t, $, $, $, $>(",
         /*VPT=*/8,
         /*EXPERTS=*/num_experts,
         /*WARPS_PER_TB=*/8,
         /*BYTES_PER_LDG=*/16);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    nullptr,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $,", batch_size);
  code.e("    $,", num_experts_per_tok);
  code.e("    task_desc->output_ptrs[1],");
  code.e("    task_desc->output_ptrs[2],");
  code.e("    0,");
  code.e("    $,", num_experts);
  code.e("    true);");
  return register_task_variant(TASK_MOE_TOPK_SOFTMAX_SM100, code.to_string());
}

int TaskRegister::register_moe_linear_sm100_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 0);
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      orig_output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[1]->output_tensors[0].dim[1] == output_size);
  assert(input_ops[1]->output_tensors[0].dim[2] == reduction_size);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);
  orig_output_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // MoE constant:
  int expert_stride = (w13_linear) ? 10 : 8;
  // define MMA
  constexpr int MMA_M = 128;
  constexpr int MMA_N = 16;
  constexpr int bM = 128;
  constexpr int bN = MMA_N;
  constexpr int bK = 64;
  constexpr int num_ab_stages = 8;
  constexpr int num_acc_stages = 2;
  constexpr int num_c_stages = 4;
  constexpr int num_tmem_columns = bN * num_acc_stages;
  assert(num_tmem_columns <= 512);
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  int const output_tma_cp_size = 128;
  int const output_atom_size = 128;
  // TMA_B for expert weights
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         (num_experts - 1) * orig_output_size + output_size, /*GMEM_ROW_*/
         reduction_size,                                     /*GMEM_COL_*/
         MMA_M,                                              /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE,                                  /*SMEM_COL_*/
         reduction_size, /*GMEM_STRIDE_ROW_*/
         1,              /*GMEM_STRIDE_COL_*/
         1,              /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  // Bias Tensor setup
  code.e(
      "cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $, $), "
      "cute::make_stride($, cute::Int<1>{}, $));",
      batch_size,
      output_size,
      num_experts,
      output_stride,
      output_stride * batch_size);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         "nullptr");
  // Topk_indices Tensor setup
  code.e("cute::Layout layout_routing_indices = "
         "cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         num_experts,
         batch_size,
         batch_size);
  code.e("cute::Tensor mRoutingIndices = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[2])), layout_routing_indices);");
  // Topk_mask Tensor setup
  code.e("cute::Layout layout_expert_mask = "
         "cute::make_layout(cute::make_shape($), "
         "cute::make_stride(cute::Int<1>{}));",
         num_experts + 1);
  code.e("cute::Tensor mMask = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[3])), layout_expert_mask);");
  // Output Tensor setup
  code.e("cute::Layout layout_output = cute::make_layout(cute::make_shape($, "
         "$, $), "
         "cute::make_stride($, cute::Int<1>{}, $));",
         batch_size,
         output_size,
         num_experts_per_tok,
         num_experts_per_tok * output_stride,
         output_stride);
  code.e("cute::Tensor mOutput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->output_ptrs[0])), layout_output);");
  // Input Tensor setup
  if (w13_linear) {
    code.e(
        "cute::Layout layout_input = cute::make_layout(cute::make_shape($, $), "
        "cute::make_stride($, cute::Int<1>{}));",
        batch_size,
        reduction_size,
        reduction_size);
  } else {
    code.e("cute::Layout layout_input = cute::make_layout(cute::make_shape($, "
           "$, $), "
           "cute::make_stride($, cute::Int<1>{}, $));",
           batch_size,
           reduction_size,
           num_experts_per_tok,
           num_experts_per_tok * reduction_size,
           reduction_size);
  }
  code.e("cute::Tensor mInput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->input_ptrs[0])), layout_input);");

  code.e("kernel::moe_linear_sm100_task_impl<cute::bfloat16_t, TMA_A, "
         "decltype(mInput), decltype(mBias), decltype(mRoutingIndices), "
         "decltype(mMask), decltype(mOutput), "
         "$, $, $, $, $, $, $, $, $, $, $, "
         "$, $, $>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         orig_output_size,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         w13_linear ? "true" : "false",
         /*no_bias*/ "true",
         num_ab_stages,
         num_acc_stages,
         num_c_stages);
  code.e("    tma_a,");
  code.e("    mInput,");
  code.e("    mBias,");
  code.e("    mRoutingIndices,");
  code.e("    mMask,");
  code.e("    mOutput,");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_SM100, code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_SM100, code.to_string());
  }
}

int TaskRegister::register_moe_silu_mul_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts_per_tok = 0, output_size = 0, input_stride,
      output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  assert(input_ops[0]->output_tensors[0].dim[2] == output_size * 2);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[2];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[1]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::silu_mul_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $);", num_experts_per_tok * batch_size);
  return register_task_variant(TASK_SILU_MUL, code.to_string());
}

int TaskRegister::register_moe_swigluoai_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts_per_tok = 0, output_size = 0, input_stride,
      output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  assert(input_ops[0]->output_tensors[0].dim[2] == output_size * 2);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[2];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[1]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::swigluoai_task_impl<bfloat16, $, $, $, $>(",
         batch_size,
         output_size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $);", num_experts_per_tok * batch_size);
  return register_task_variant(TASK_SWIGLUOAI_MI300, code.to_string());
}

int TaskRegister::register_bias_add_mi300_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, size = 0, input_stride = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;  // input tensor + bias
  int num_outputs = 1; // output tensor
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [batch_size, size]
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  size = output_ops[0]->output_tensors[0].dim[1];
  // Input tensor: [batch_size, size]
  assert(input_ops[0]->output_tensors[0].num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[1] == size);
  // Bias: [1, size] (broadcast across batch)
  assert(input_ops[1]->output_tensors[0].num_dims == 2);
  assert(input_ops[1]->output_tensors[0].dim[1] == size);
  // Get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[1];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
  // Get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::bias_add_task_impl<$, $, $, $>(",
         batch_size,
         size,
         input_stride,
         output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_BIAS_ADD_MI300, code.to_string());
}

int TaskRegister::register_attention_sink_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: head_dim
  assert(params.size() == 2);
  int num_q_heads = params[0];
  int head_dim = params[1];
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::attention_sink_correction_impl<$, $>(", num_q_heads, head_dim);
  code.e("    task_desc->input_ptrs[0],"); // attn_out (in-place)
  code.e("    task_desc->input_ptrs[1],"); // lse_acc
  code.e("    task_desc->input_ptrs[2],"); // sinks
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    runtime_config.qo_indptr_buffer);");
  return register_task_variant(TASK_ATTENTION_SINK_MI300, code.to_string());
}

int TaskRegister::register_moe_mul_sum_add_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts_per_tok = 0, output_size = 0, input_stride,
      output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  assert(input_ops[1]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  num_experts_per_tok = input_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[2] ==
             input_ops[2]->output_tensors[0].dim[1] &&
         input_ops[0]->output_tensors[0].dim[2] == output_size);
  // get input stride
  assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
  input_stride = input_ops[0]->dtensor.dim[2];
  assert(input_stride == static_cast<int>(kn_input_op->input_strides[1]));
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mul_sum_add_sm100_task_impl<cute::bfloat16_t, $, $, $, $>(",
         /*BATCH_SIZE=*/batch_size,
         /*OUTPUT_SIZE=*/output_size,
         /*NUM_TOPK=*/num_experts_per_tok,
         /*OUTPUT_STRIDE=*/output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_MOE_MUL_SUM_ADD_SM100, code.to_string());
}

int TaskRegister::register_moe_linear_sm90_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 0);
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      orig_output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 4;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[1]->output_tensors[0].dim[1] == output_size);
  assert(input_ops[1]->output_tensors[0].dim[2] == reduction_size);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);
  orig_output_size = input_ops[1]->dtensor.dim[1];

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // MoE constant:
  int const expert_stride = w13_linear ? 5 : 4;
  // define MMA
  constexpr int MMA_M = 64;
  constexpr int MMA_N = 16;
  constexpr int num_ab_stages = 8;
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  // int const output_tma_cp_size = 128;
  // int const output_atom_size = 128;
  // TMA_B for expert weights
  code.e("using TMA_A = kernel::tma::tma_2d<cute::bfloat16_t, $, $, $, $, $, "
         "$, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         //  (num_experts-1) * orig_output_size + output_size, /*GMEM_ROW_*/
         (num_experts)*orig_output_size, /*GMEM_ROW_*/
         reduction_size,                 /*GMEM_COL_*/
         MMA_M,                          /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE,              /*SMEM_COL_*/
         reduction_size,                 /*GMEM_STRIDE_ROW_*/
         1,                              /*GMEM_STRIDE_COL_*/
         1,                              /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,    /*SMEM_REPEAT_COL_*/
         MMA_M * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  // Bias Tensor setup
  code.e(
      "cute::Layout layout_Bias = cute::make_layout(cute::make_shape($, $, $), "
      "cute::make_stride($, cute::Int<1>{}, $));",
      batch_size,
      output_size,
      num_experts,
      output_stride,
      output_stride * batch_size);
  code.e("cute::Tensor mBias = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "$)), layout_Bias);",
         "nullptr");
  // Topk_indices Tensor setup
  code.e("cute::Layout layout_routing_indices = "
         "cute::make_layout(cute::make_shape($, $), "
         "cute::make_stride($, cute::Int<1>{}));",
         num_experts,
         batch_size,
         batch_size);
  code.e("cute::Tensor mRoutingIndices = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[2])), layout_routing_indices);");
  // Topk_mask Tensor setup
  code.e("cute::Layout layout_expert_mask = "
         "cute::make_layout(cute::make_shape($), "
         "cute::make_stride(cute::Int<1>{}));",
         num_experts);
  code.e("cute::Tensor mMask = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::int32_t*>("
         "task_desc->input_ptrs[3])), layout_expert_mask);");
  // Output Tensor setup
  code.e("cute::Layout layout_output = cute::make_layout(cute::make_shape($, "
         "$, $), "
         "cute::make_stride($, cute::Int<1>{}, $));",
         batch_size,
         output_size,
         num_experts_per_tok,
         num_experts_per_tok * output_stride,
         output_stride);
  code.e("cute::Tensor mOutput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->output_ptrs[0])), layout_output);");
  // Input Tensor setup
  if (w13_linear) {
    code.e(
        "cute::Layout layout_input = cute::make_layout(cute::make_shape($, $), "
        "cute::make_stride($, cute::Int<1>{}));",
        batch_size,
        reduction_size,
        reduction_size);
  } else {
    code.e("cute::Layout layout_input = cute::make_layout(cute::make_shape($, "
           "$, $), "
           "cute::make_stride($, cute::Int<1>{}, $));",
           batch_size,
           reduction_size,
           num_experts_per_tok,
           num_experts_per_tok * reduction_size,
           reduction_size);
  }
  code.e("cute::Tensor mInput = "
         "cute::make_tensor(cute::make_gmem_ptr(static_cast<cute::bfloat16_t*>("
         "task_desc->input_ptrs[0])), layout_input);");

  code.e("kernel::moe_linear_sm90_task_impl<cute::bfloat16_t, TMA_A, "
         "decltype(mInput), decltype(mBias), decltype(mRoutingIndices), "
         "decltype(mMask), decltype(mOutput), "
         "$, $, $, $, $, $, $, $, $, $, $, "
         "$>(",
         MMA_M,
         MMA_N,
         batch_size,
         output_size,
         orig_output_size,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         w13_linear ? "true" : "false",
         /*no_bias*/ "true",
         num_ab_stages);
  code.e("    tma_a,");
  code.e("    mInput,");
  code.e("    mBias,");
  code.e("    mRoutingIndices,");
  code.e("    mMask,");
  code.e("    mOutput,");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_SM90, code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_SM90, code.to_string());
  }
}

// ── MI300/MI350 MoE task registration ──────────────────────────────────

int TaskRegister::register_moe_topk_softmax_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts = 0, num_experts_per_tok = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 1;
  int num_outputs = 3;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  assert(output_ops[1]->output_tensors[0].num_dims == 2);
  assert(output_ops[2]->output_tensors[0].num_dims == 1);
  num_experts = output_ops[1]->output_tensors[0].dim[0];
  batch_size = output_ops[1]->output_tensors[0].dim[1];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(output_ops[2]->output_tensors[0].dim[0] == num_experts + 1);
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[1] == num_experts);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::topk_softmax_mi300_task_impl<hip_bfloat16, $, $, $, $>(",
         /*VPT=*/8,
         /*EXPERTS=*/num_experts,
         /*WARPS_PER_CTA=*/4,
         /*BYTES_PER_LDG=*/16);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $,", batch_size);
  code.e("    $,", num_experts_per_tok);
  code.e("    task_desc->output_ptrs[1],");
  code.e("    task_desc->output_ptrs[2],");
  code.e("    0,");
  code.e("    $,", num_experts);
  code.e("    true);");
  return register_task_variant(TASK_MOE_TOPK_SOFTMAX_MI300, code.to_string());
}

// Sigmoid + e_score_correction_bias router (`noaux_tc`), used by GLM-5.
// Same shapes as the softmax router, plus a [num_experts] bias input.
int TaskRegister::register_moe_topk_sigmoid_bias_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: routed_scaling_factor in 1/1000 units (GLM-5: 2500 => 2.5f)
  // params[1]: norm_topk_prob (0/1)
  assert(params.size() == 2);
  int batch_size = 0, num_experts = 0, num_experts_per_tok = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 3;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  assert(output_ops[1]->output_tensors[0].num_dims == 2);
  assert(output_ops[2]->output_tensors[0].num_dims == 1);
  // The routing tables are sized for the *total* expert count. A shared
  // expert, if any, is the one extra row past the routed experts and takes
  // the one extra routing slot past the k selected ones.
  int num_total_experts = output_ops[1]->output_tensors[0].dim[0];
  batch_size = output_ops[1]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  num_experts = input_ops[0]->output_tensors[0].dim[1];
  int num_shared_experts = num_total_experts - num_experts;
  assert(num_shared_experts == 0 || num_shared_experts == 1);
  num_experts_per_tok =
      output_ops[0]->output_tensors[0].dim[1] - num_shared_experts;
  assert(num_experts_per_tok > 0);
  assert(output_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(output_ops[2]->output_tensors[0].dim[0] == num_total_experts + 1);
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  // e_score_correction_bias: [num_experts]
  assert(input_ops[1]->dtensor.num_dims == 1);
  assert(input_ops[1]->output_tensors[0].dim[0] == num_experts);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::topk_sigmoid_bias_mi300_task_impl<hip_bfloat16, $, $, $, $>(",
         /*VPT=*/8,
         /*EXPERTS=*/num_experts,
         /*WARPS_PER_CTA=*/4,
         /*BYTES_PER_LDG=*/16);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    $,", batch_size);
  code.e("    $,", num_experts_per_tok);
  code.e("    task_desc->output_ptrs[1],");
  code.e("    task_desc->output_ptrs[2],");
  code.e("    0,");
  code.e("    $,", num_experts);
  code.e("    $,", params[1] != 0 ? "true" : "false");
  code.e("    $ / 1000.0f,", params[0]);
  code.e("    $);", num_shared_experts);
  return register_task_variant(TASK_MOE_TOPK_SIGMOID_BIAS_MI300,
                               code.to_string());
}

int TaskRegister::register_moe_linear_mi300_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 0);
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[1]->output_tensors[0].dim[1] == output_size);
  assert(input_ops[1]->output_tensors[0].dim[2] == reduction_size);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  // input_ops[4] is bias: [num_experts, output_stride]
  assert(input_ops[4]->output_tensors[0].num_dims == 2);
  assert(input_ops[4]->output_tensors[0].dim[0] == num_experts);
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);

  // MI300 expert stride: must match grid_dim.x from the Python API
  int expert_stride = bgraph.grid_dim.x;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_linear_kernel_mi300<bfloat16, $, $, $, $, $, $, $, $>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         w13_linear ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_MI300, code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_MI300, code.to_string());
  }
}

int TaskRegister::register_moe_linear_mxfp4_mi300_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  // params[0] = output_per_wg (typically 16)
  assert(params.size() == 1);
  int output_per_wg = params[0];
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  // Output: [batch_size, num_experts_per_tok, output_size]
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];

  // Input activation: [batch_size, reduction_size] for W13, [batch_size, topk,
  // reduction_size] for W2
  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }

  // Weight: [num_experts, expert_wgs, wg_bytes] as uint8
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];

  // Routing indices: [num_experts, batch_size]
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  // Mask: [num_experts + 1]
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  // Bias: [num_experts, 1 (tiled), output_per_wg] bf16
  assert(input_ops[4]->output_tensors[0].num_dims == 3);
  assert(input_ops[4]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[4]->output_tensors[0].dim[2] == output_per_wg);

  // Get output stride from the KN-level input op
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);

  int expert_stride = bgraph.grid_dim.x;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_linear_mxfp4_kernel_mi300<$, $, $, $, $, $, $, $, $>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         output_per_wg,
         w13_linear ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_MXFP4_MI300,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_MXFP4_MI300,
                                 code.to_string());
  }
}

int TaskRegister::register_moe_linear_mxfp4_ck_mi300_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool w13_linear) {
  assert(params.size() == 1);
  int output_per_wg = params[0];
  int num_experts = 0, num_experts_per_tok = 0, batch_size = 0, output_size = 0,
      reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 5;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  output_size = output_ops[0]->output_tensors[0].dim[2];

  if (w13_linear) {
    assert(input_ops[0]->output_tensors[0].num_dims == 2);
    reduction_size = input_ops[0]->output_tensors[0].dim[1];
  } else {
    assert(input_ops[0]->output_tensors[0].num_dims == 3);
    reduction_size = input_ops[0]->output_tensors[0].dim[2];
    assert(input_ops[0]->output_tensors[0].dim[1] == num_experts_per_tok);
  }

  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  num_experts = input_ops[1]->output_tensors[0].dim[0];

  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[2]->output_tensors[0].dim[1] == batch_size);
  assert(input_ops[3]->output_tensors[0].num_dims == 1);
  assert(input_ops[3]->output_tensors[0].dim[0] == num_experts + 1);
  assert(input_ops[4]->output_tensors[0].num_dims == 3);
  assert(input_ops[4]->output_tensors[0].dim[0] == num_experts);
  assert(input_ops[4]->output_tensors[0].dim[2] == output_per_wg);

  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[1]);

  int expert_stride = bgraph.grid_dim.x;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_linear_mxfp4_ck_kernel_mi300<$, $, $, $, $, $, $, $, $>(",
         batch_size,
         output_size,
         output_stride,
         reduction_size,
         num_experts,
         num_experts_per_tok,
         expert_stride,
         output_per_wg,
         w13_linear ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.expert_offset);");
  if (w13_linear) {
    return register_task_variant(TASK_MOE_W13_LINEAR_MXFP4_CK_MI300,
                                 code.to_string());
  } else {
    return register_task_variant(TASK_MOE_W2_LINEAR_MXFP4_CK_MI300,
                                 code.to_string());
  }
}

int TaskRegister::register_moe_mul_sum_add_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  int batch_size = 0, num_experts_per_tok = 0, output_size = 0, output_stride;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 3;
  int num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->output_tensors[0].num_dims == 3);
  assert(input_ops[1]->output_tensors[0].num_dims == 2);
  assert(input_ops[2]->output_tensors[0].num_dims == 2);
  num_experts_per_tok = input_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->output_tensors[0].dim[0] == batch_size);
  assert(input_ops[0]->output_tensors[0].dim[2] ==
             input_ops[2]->output_tensors[0].dim[1] &&
         input_ops[0]->output_tensors[0].dim[2] == output_size);
  // get output stride
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mul_sum_add_mi300_task_impl<hip_bfloat16, $, $, $, $>(",
         /*BATCH_SIZE=*/batch_size,
         /*OUTPUT_SIZE=*/output_size,
         /*NUM_TOPK=*/num_experts_per_tok,
         /*OUTPUT_STRIDE=*/output_stride);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[0]);");
  return register_task_variant(TASK_MOE_MUL_SUM_ADD_MI300, code.to_string());
}

int TaskRegister::register_splitk_linear_swapAB_hopper_task(
    threadblock::Graph const &bgraph,
    std::vector<int> const &params,
    bool with_residual) {
  assert(params.size() == 0);
  assert(with_residual == false);
  int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = with_residual ? 3 : 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  batch_size = output_ops[0]->output_tensors[0].dim[0];
  output_size = output_ops[0]->output_tensors[0].dim[1];
  assert(input_ops[0]->dtensor.num_dims == 2);
  reduction_size = input_ops[0]->dtensor.dim[1];
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_input_op =
      static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  output_stride = static_cast<int>(kn_input_op->input_strides[0]);

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // define TMAs
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE = 64;
  constexpr int TILE_SIZE = 64;
  constexpr int Kstages = 5;
  assert(batch_size <= 16);
  int const SMEM_M_SIZE = batch_size <= 8 ? 8 : 16;
  // int const SMEM_M_SIZE = 16;
  int const output_tma_cp_size = output_size < 64 ? output_size : 64;
  int const output_atom_size = 64;
  code.e("using TMA_B = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,        /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         batch_size,        /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,          /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  code.e("using TMA_A = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $, $, $, true>;",
         B,
         M,
         S,
         output_size,       /*GMEM_ROW_*/
         reduction_size,    /*GMEM_COL_*/
         output_atom_size,  /*SMEM_ROW_*/
         TMA_CP_ASYNC_SIZE, /*SMEM_COL_*/
         reduction_size,    /*GMEM_STRIDE_ROW_*/
         1,                 /*GMEM_STRIDE_COL_*/
         1,                 /*SMEM_REPEAT_ROW_*/
         (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) /
             TMA_CP_ASYNC_SIZE,               /*SMEM_REPEAT_COL_*/
         output_atom_size * TMA_CP_ASYNC_SIZE /*SMEM_STRIDE_*/
  );

  if (with_residual) {
    code.e(
        "using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, "
        "$, $, $, $, $, $, true>;",
        0,
        0,
        0,
        batch_size,                      /*GMEM_ROW_*/
        output_size,                     /*GMEM_COL_*/
        batch_size,                      /*SMEM_ROW_*/
        output_tma_cp_size,              /*SMEM_COL_*/
        output_stride,                   /*GMEM_STRIDE_ROW_*/
        1,                               /*GMEM_STRIDE_COL_*/
        1,                               /*SMEM_REPEAT_ROW_*/
        1,                               /*SMEM_REPEAT_COL_*/
        SMEM_M_SIZE * output_tma_cp_size /*SMEM_STRIDE_*/
    );
  }

  code.e("using TMA_OUT = kernel::tma::tma_2d<bfloat16, $, $, $, $, $, $, $, "
         "$, $, $, $, $, true>;",
         B,
         M,
         S,
         batch_size,                      /*GMEM_ROW_*/
         output_size,                     /*GMEM_COL_*/
         batch_size,                      /*SMEM_ROW_*/
         output_tma_cp_size,              /*SMEM_COL_*/
         output_stride,                   /*GMEM_STRIDE_ROW_*/
         1,                               /*GMEM_STRIDE_COL_*/
         1,                               /*SMEM_REPEAT_ROW_*/
         1,                               /*SMEM_REPEAT_COL_*/
         SMEM_M_SIZE * output_tma_cp_size /*SMEM_STRIDE_*/
  );
  code.inc_indent();
  code.e("TMA_A "
         "tma_a(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[1][0])"
         ");");
  code.e("TMA_B "
         "tma_b(static_cast<CUtensorMap*>(task_desc->input_tma_desc_ptrs[0][0])"
         ");");
  if (with_residual) {
    code.e("TMA_RESIDUAL "
           "tma_residual(static_cast<CUtensorMap*>(task_desc->input_tma_desc_"
           "ptrs[2][0]));");
  }
  code.e("TMA_OUT "
         "tma_out(static_cast<CUtensorMap*>(task_desc->output_tma_desc_ptrs[0]["
         "0]));");

  code.e(
      "kernel::linear_swapAB_kernel_hopper<bfloat16, $, $, $, $, TMA_A, TMA_B, "
      "TMA_OUT, $, $, $>(",
      batch_size,
      output_size,
      reduction_size,
      Kstages,
      with_residual ? "TMA_RESIDUAL" : "void",
      output_stride,
      "true" /*SplitK*/);
  code.e("    tma_a,");
  code.e("    tma_b,");
  code.e("    tma_out, ");
  if (with_residual) {
    code.e("    &tma_residual");
  } else {
    code.e("    nullptr");
  }
  code.e(");");

  return register_task_variant(TASK_SPLITK_LINEAR_SWAPAB_HOPPER,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  // params[6]: num_kv_chunks
  assert(params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3); // lse
  assert(output_ops[1]->output_tensors[0].num_dims == 3); // output_tmp

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int output_size = head_dim * num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 256;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_split_kv_task_impl<bfloat16, $, "
         "$, $, $, $, $, "
         "$, $, $, $, $, $, $>(",
         num_q_heads / num_kv_heads,
         1,
         num_kv_heads,
         kv_stride,
         qkv_stride,
         output_size * num_kv_chunks, // o_stride should consider num_kv_chunks
         head_dim,
         SEQ_LEN_PER_BLOCK,
         max_seq_len,
         page_size,
         max_tokens,
         "true", // PARTITION_KV
         num_kv_chunks);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    task_desc->output_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.kv_idx);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_SM100,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_merge_sm100_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_qo_heads_per_kv
  // params[1]: head_dim
  // params[2]: max_seq_len
  // params[3]: page_size
  // params[4]: num_kv_heads
  assert(params.size() == 5);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int output_size = output_ops[0]->dtensor.dim[1];
  int num_q_heads_per_kv = params[0];
  int head_dim = params[1];
  int max_seq_len = params[2];
  int page_size = params[3];
  int num_kv_heads = params[4];

  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 256;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();

  code.e("kernel::merge_splitkv<bfloat16, $, $, $, $, $, $, "
         "$, $, $>(",
         num_q_heads_per_kv,
         1,
         num_kv_heads,
         head_dim,
         max_tokens,
         true,
         ((max_seq_len + SEQ_LEN_PER_BLOCK - 1) / SEQ_LEN_PER_BLOCK),
         SEQ_LEN_PER_BLOCK,
         page_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.merge_task_offset);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_embed
  // params[4]: max_seq_len
  // params[5]: page_size
  // params[6]: num_kv_chunks
  assert(params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3); // lse
  assert(output_ops[1]->output_tensors[0].num_dims == 3); // output_tmp

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int output_size = head_dim * num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 128;

  // Cap MAX_TOKENS to fit in attention kernel LDS budget.
  // The attention kernel needs Q_ROWS = MAX_TOKENS * qo_per_kv shared memory
  // rows. The scheduler's prepare_next_batch also caps tokens per request via
  // MPK_MAX_TOKENS_PER_REQUEST to ensure runtime num_tokens <= MAX_TOKENS.
  //
  // LDS budget per generation (must match MAX_DYNAMIC_SHARED_MEMORY_SIZE in
  // runtime_header.h, minus WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE=3KB):
  //   MI300X (gfx942): 60 KB total -> 57 KB usable = 58368 bytes
  //   MI350X (gfx950): 155 KB total -> 152 KB usable = 155648 bytes
  // Detected at runtime via hipDeviceProp_t::sharedMemPerBlock.
  {
    int qo_per_kv = num_q_heads / num_kv_heads;
    constexpr int KV_TILE = 64;
    int lds_limit = 58368; // MI300X conservative default
#ifdef MIRAGE_BACKEND_USE_ROCM
    // Query the device's actual LDS budget once and reserve 5 KB
    // (3 KB worker static + 2 KB safety margin).
    static int cached_lds_limit = []() {
      size_t per_block = mirage::utils::get_max_shared_mem();
      // MPK_WORKER_LDS_KB shrinks the DYNAMIC segment the worker kernel is
      // actually launched with -- MAX_DYNAMIC_SHARED_MEMORY_SIZE in
      // runtime_header.h is MPK_WORKER_LDS_KB*1024 minus the 3 KB static
      // reserve. The device query below reports what the HARDWARE offers
      // (160 KB on gfx950), which is a different number entirely the moment
      // that knob is set.
      //
      // Sizing MAX_TOKENS against the hardware number while launching with
      // the smaller one would let the attention task index smem[] past the
      // end of the segment it was actually given, so the clamp below is a
      // real latent fix. It is only that, though: it does NOT explain the
      // HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION this was first written
      // for. That fault was re-isolated to MPK_WORKER_WAVES_PER_EU=3 alone
      // -- it reproduces at the DEFAULT worker count and the DEFAULT
      // MPK_WORKER_LDS_KB (155), where this clamp is inert, and LDS_KB=78
      // on its own does not fault. The knob was guilty by association.
      //
      // This is host code compiled separately from the device headers, so
      // the knob has to be re-read from the environment rather than seen as
      // a -D.
      const char *lds_kb = std::getenv("MPK_WORKER_LDS_KB");
      if (lds_kb != nullptr) {
        int kb = std::atoi(lds_kb);
        if (kb >= 8 && kb <= 155) {
          size_t budget = static_cast<size_t>(kb) * 1024;
          if (budget < per_block) {
            per_block = budget;
          }
        }
      }
      if (per_block > 5 * 1024) {
        return static_cast<int>(per_block - 5 * 1024);
      }
      return 58368;
    }();
    lds_limit = cached_lds_limit;
#endif
    int per_qrow = head_dim * 2 + KV_TILE * 4 + head_dim * 4 + 8;
    int fixed = KV_TILE * head_dim * 2 + 256;
    int max_qrows = (lds_limit - fixed) / per_qrow;
    int max_tokens_lds = max_qrows / qo_per_kv;
    if (max_tokens_lds < 1) {
      max_tokens_lds = 1;
    }
    if (max_tokens > max_tokens_lds) {
      max_tokens = max_tokens_lds;
    }
  }

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_split_kv_task_impl<bfloat16, $, "
         "$, $, $, $, $, "
         "$, $, $, $, $, $>(",
         num_q_heads / num_kv_heads, /* NUM_QO_HEADS */
         1,                          /* NUM_KV_HEADS */
         num_kv_heads,               /* NUM_QO_GROUPS */
         kv_stride,                  /* KV_CACHE_STRIDE */
         qkv_stride,                 /* QKV_STRIDE */
         output_size *
             num_kv_chunks, /* O_STRIDE (accounts for num_kv_chunks) */
         head_dim,          /* HEAD_DIM */
         SEQ_LEN_PER_BLOCK, /* SEQ_LEN_PER_BLOCK */
         max_seq_len,       /* MAX_SEQ_LEN */
         page_size,         /* PAGE_SIZE */
         max_tokens,        /* MAX_TOKENS */
         num_kv_chunks);    /* NUM_KV_CHUNKS */
  code.e("    task_desc->input_ptrs[0],");  // qkv
  code.e("    task_desc->input_ptrs[1],");  // k_cache
  code.e("    task_desc->input_ptrs[2],");  // v_cache
  code.e("    task_desc->output_ptrs[1],"); // output_tmp
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],"); // q_norm
  code.e("    task_desc->input_ptrs[4],"); // k_norm
  code.e("    task_desc->input_ptrs[5],"); // cos
  code.e("    task_desc->input_ptrs[6],"); // sin
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->output_ptrs[0],"); // lse
  code.e("    task_desc->task_metadata.kv_idx);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_MI300,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_merge_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_qo_heads_per_kv
  // params[1]: head_dim
  // params[2]: max_seq_len
  // params[3]: page_size
  // params[4]: num_kv_heads
  assert(params.size() == 5);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 2;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int num_q_heads_per_kv = params[0];
  int head_dim = params[1];
  int max_seq_len = params[2];
  int page_size = params[3];
  int num_kv_heads = params[4];

  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 128;

  // Cap MAX_TOKENS to match split-kv attention kernel's LDS limit
  {
    constexpr int KV_TILE = 64;
    constexpr int LDS_LIMIT = 58368;
    int per_qrow = head_dim * 2 + KV_TILE * 4 + head_dim * 4 + 8;
    int fixed = KV_TILE * head_dim * 2 + 256;
    int max_qrows = (LDS_LIMIT - fixed) / per_qrow;
    int max_tokens_lds = max_qrows / num_q_heads_per_kv;
    if (max_tokens_lds < 1) {
      max_tokens_lds = 1;
    }
    if (max_tokens > max_tokens_lds) {
      max_tokens = max_tokens_lds;
    }
  }

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Reuse the portable merge_splitkv kernel from ampere/
  code.e("kernel::merge_splitkv<bfloat16, $, $, $, $, $, $, "
         "$, $, $>(",
         num_q_heads_per_kv,
         1,
         num_kv_heads,
         head_dim,
         max_tokens,
         true,
         ((max_seq_len + SEQ_LEN_PER_BLOCK - 1) / SEQ_LEN_PER_BLOCK),
         SEQ_LEN_PER_BLOCK,
         page_size);
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->input_ptrs[1],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->output_ptrs[0],");
  code.e("    task_desc->task_metadata.merge_task_offset);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_MI300,
                               code.to_string());
}

int TaskRegister::register_kv_cache_update_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_embed
  // params[4]: max_seq_len
  // params[5]: page_size
  // params[6]: q_workspace_stride
  assert(params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int q_workspace_stride = params[6];
  int max_tokens = input_ops[0]->dtensor.dim[0];

  // Cap MAX_TOKENS to fit in LDS. kv_cache_update uses:
  //   S_Q = sizeof(bf16) * MAX_TOKENS * (num_q/num_kv) * head_dim
  //   S_K = sizeof(bf16) * MAX_TOKENS * head_dim
  // Total dynamic smem ≈ MAX_TOKENS * (qo_per_kv + 1) * head_dim * 2
  // Runtime also caps tokens per request via MPK_MAX_TOKENS_PER_REQUEST (8 on
  // AMD).
  {
    int qo_per_kv = num_q_heads / num_kv_heads;
    constexpr int LDS_LIMIT = 58368;                      // 60KB - 3KB reserved
    int per_token_bytes = (qo_per_kv + 1) * head_dim * 2; // bf16
    int overhead = 256; // alignment + reduction buffer
    int max_tokens_lds = (LDS_LIMIT - overhead) / per_token_bytes;
    if (max_tokens_lds < 1) {
      max_tokens_lds = 1;
    }
    if (max_tokens > max_tokens_lds) {
      max_tokens = max_tokens_lds;
    }
  }

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e(
      "kernel::kv_cache_update_impl<bfloat16, $, $, $, $, $, $, $, $, $, $>(",
      num_q_heads / num_kv_heads,           /* NUM_QO_HEADS */
      1,                                    /* NUM_KV_HEADS */
      num_kv_heads,                         /* NUM_QO_GROUPS */
      kv_stride,                            /* KV_CACHE_STRIDE */
      qkv_stride,                           /* QKV_STRIDE */
      head_dim,                             /* HEAD_DIM */
      max_seq_len,                          /* MAX_SEQ_LEN */
      page_size,                            /* PAGE_SIZE */
      max_tokens,                           /* MAX_TOKENS */
      q_workspace_stride);                  /* Q_WORKSPACE_STRIDE */
  code.e("    task_desc->input_ptrs[0],");  // qkv
  code.e("    task_desc->input_ptrs[1],");  // k_cache
  code.e("    task_desc->input_ptrs[2],");  // v_cache
  code.e("    task_desc->output_ptrs[0],"); // q_workspace
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],"); // q_norm
  code.e("    task_desc->input_ptrs[4],"); // k_norm
  code.e("    task_desc->input_ptrs[5],"); // cos
  code.e("    task_desc->input_ptrs[6],"); // sin
  code.e("    1e-6f,");
  code.e("    1e-6f);");
  return register_task_variant(TASK_KV_CACHE_UPDATE_MI300, code.to_string());
}

int TaskRegister::register_mla_kv_cache_update_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_qo_heads (absorbed heads written to the q workspace)
  // params[1]: kv_lora_rank
  // params[2]: qk_rope_head_dim
  // params[3]: max_seq_len
  // params[4]: page_size
  // params[5]: q_workspace_stride
  // params[6]: kv_input_offset (column the latent starts at, optional)
  assert(params.size() == 6 || params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 6;
  int num_outputs = 1;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int num_qo_heads = params[0];
  int kv_lora_rank = params[1];
  int qk_rope_head_dim = params[2];
  int max_seq_len = params[3];
  int page_size = params[4];
  int q_workspace_stride = params[5];
  int q_input_stride = input_ops[0]->dtensor.dim[1];
  int kv_input_stride = input_ops[1]->dtensor.dim[1];
  int kv_input_offset = params.size() == 7 ? params[6] : 0;
  assert(kv_input_offset + kv_lora_rank + qk_rope_head_dim <= kv_input_stride);
  // Latent cache is (num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim);
  // MLA keeps a single shared latent head, so the cache row stride is just the
  // last dim -- unlike the GQA path, which multiplies by NUM_KV_HEADS.
  int kv_cache_stride = input_ops[2]->output_tensors[0].dim[3];
  assert(kv_cache_stride >= kv_lora_rank + qk_rope_head_dim);

  // No LDS staging here, so unlike kv_cache_update there is nothing to cap
  // MAX_TOKENS against; the only shared memory is the page-index table.
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mla_kv_cache_update_impl<bfloat16, $, $, $, $, $, $, $, $, "
         "$, $>(",
         num_qo_heads,       /* NUM_QO_HEADS */
         kv_lora_rank,       /* KV_LORA_RANK */
         qk_rope_head_dim,   /* QK_ROPE_HEAD_DIM */
         q_input_stride,     /* Q_INPUT_STRIDE */
         kv_input_stride,    /* KV_INPUT_STRIDE */
         kv_cache_stride,    /* KV_CACHE_STRIDE */
         max_seq_len,        /* MAX_SEQ_LEN */
         page_size,          /* PAGE_SIZE */
         q_workspace_stride, /* Q_WORKSPACE_STRIDE */
         kv_input_offset);   /* KV_INPUT_OFFSET */
  code.e("    task_desc->input_ptrs[0],");  // q_absorbed
  code.e("    task_desc->input_ptrs[1],");  // kv_latent
  code.e("    task_desc->input_ptrs[2],");  // paged latent cache
  code.e("    task_desc->output_ptrs[0],"); // q_workspace
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    task_desc->input_ptrs[3],"); // kv_a_layernorm weight
  code.e("    task_desc->input_ptrs[4],"); // cos
  code.e("    task_desc->input_ptrs[5],"); // sin
  code.e("    1e-6f);");
  return register_task_variant(TASK_MLA_KV_CACHE_UPDATE_MI300,
                               code.to_string());
}

int TaskRegister::register_paged_attention_ck_fmha_split_kv_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: head_dim
  // params[3]: page_size
  // params[4]: max_seq_len
  // params[5]: num_kv_chunks
  // params[6]: q_workspace_stride
  // params[7]: kv_cache_stride
  // params[8]: max_num_requests
  // params[9]: sliding_window (0 = disabled)
  assert(params.size() == 10);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_outputs = 2;
  // Inputs: 3 (q_workspace, k_cache, v_cache) or 4 (+ sinks for GPT-OSS)
  int total_ops = (int)bgraph.operators.size();
  int num_inputs = total_ops - num_outputs;
  assert(num_inputs == 3 || num_inputs == 4);
  bool has_sinks = (num_inputs == 4);

  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }

  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = params[2];
  int page_size = params[3];
  int max_seq_len = params[4];
  int num_kv_chunks = params[5];
  int q_workspace_stride = params[6];
  int kv_cache_stride = params[7];
  int max_num_requests = params[8];
  int sliding_window = params[9];
  int num_qo_per_kv = num_q_heads / num_kv_heads;
  float scale_s = 1.0f / sqrtf((float)head_dim) *
                  1.44269504088896340736f; // CK_TILE_FMHA_FWD_FAST_EXP2=1:
                                           // scale includes log2(e)

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::paged_attention_ck_fmha_split_kv_impl<bfloat16, $, $, $, $, "
         "$, $, $, $>(",
         num_qo_per_kv,                     /* NUM_QO_PER_KV */
         head_dim,                          /* HEAD_DIM */
         page_size,                         /* PAGE_SIZE */
         max_seq_len,                       /* MAX_SEQ_LEN */
         num_kv_chunks,                     /* NUM_KV_CHUNKS */
         q_workspace_stride,                /* Q_WORKSPACE_STRIDE */
         kv_cache_stride,                   /* KV_CACHE_STRIDE_T */
         num_kv_heads);                     /* NUM_KV_HEADS_T */
  code.e("    task_desc->input_ptrs[0],");  // q_workspace
  code.e("    task_desc->input_ptrs[1],");  // k_cache
  code.e("    task_desc->input_ptrs[2],");  // v_cache
  code.e("    task_desc->output_ptrs[0],"); // o_acc
  code.e("    task_desc->output_ptrs[1],"); // lse_acc
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");        // request_id
  code.e("    task_desc->task_metadata.merge_task_offset,"); // kv_head_idx
  code.e("    task_desc->task_metadata.kv_idx,");            // kv_chunk_idx
  code.e("    $f,", scale_s);
  code.e("    $,", sliding_window); // sliding_window (0 = disabled)
  if (has_sinks) {
    code.e("    task_desc->input_ptrs[3]);"); // sinks (GPT-OSS per-head
                                              // attention sinks)
  } else {
    code.e("    nullptr);"); // no sinks
  }
  return register_task_variant(TASK_PAGED_ATTENTION_CK_FMHA_SPLIT_KV_MI300,
                               code.to_string());
}

int TaskRegister::register_paged_attention_ck_fmha_merge_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_qo_heads_per_kv
  // params[1]: head_dim
  // params[2]: max_seq_len
  // params[3]: page_size
  // params[4]: num_kv_heads
  // params[5]: num_kv_chunks
  // params[6]: dim_splits (1 = one task per kv head, the GQA default)
  // params[7]: write_through (st_wt the bf16 output straight past L2)
  assert(params.size() == 8);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  // num_inputs is 2 (lse, o) or 3 (lse, o, sinks). The Python wrapper appends
  // the sinks tensor as a 3rd input when GPT-OSS-style sink correction is
  // needed.
  int num_outputs = 1;
  int num_inputs = static_cast<int>(bgraph.operators.size()) - num_outputs;
  assert(num_inputs == 2 || num_inputs == 3);
  bool has_sinks = (num_inputs == 3);

  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int num_q_heads_per_kv = params[0];
  int head_dim = params[1];
  int max_seq_len = params[2];
  int page_size = params[3];
  int num_kv_heads = params[4];
  int num_kv_chunks = params[5];
  int dim_splits = params[6];
  assert(dim_splits >= 1 && head_dim % dim_splits == 0);
  bool write_through = params[7] != 0;

  int max_tokens = input_ops[0]->dtensor.dim[0];

  constexpr int SEQ_LEN_PER_BLOCK = 128;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  // Use dedicated CK FMHA merge kernel that handles separate lse/o and output
  // strides
  code.e("kernel::merge_splitkv_ck_fmha<bfloat16, $, $, $, $, $, $, $, $>(",
         num_q_heads_per_kv,
         num_kv_heads, // NUM_QO_GROUPS (for output stride)
         head_dim,
         num_kv_chunks,
         SEQ_LEN_PER_BLOCK,
         page_size,
         write_through ? "true" : "false",
         dim_splits);
  code.e(
      "    reinterpret_cast<float const*>(task_desc->input_ptrs[0]),"); // lse_acc
  code.e(
      "    reinterpret_cast<float const*>(task_desc->input_ptrs[1]),"); // o_acc
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    reinterpret_cast<bfloat16*>(task_desc->output_ptrs[0]),");
  if (has_sinks) {
    code.e("    task_desc->task_metadata.merge_task_offset,"); // kv_head_idx
    code.e("    task_desc->input_ptrs[2]);");                  // sinks_ptr
  } else {
    code.e("    task_desc->task_metadata.merge_task_offset);"); // kv_head_idx
  }
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_MI300,
                               code.to_string());
}

int TaskRegister::register_paged_attention_split_kv_hopper_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  // params[0]: num_q_heads
  // params[1]: num_kv_heads
  // params[2]: qk_norm
  // params[3]: rotary_emd
  // params[4]: max_seq_len
  // params[5]: page_size
  // params[6]: num_kv_chunks
  assert(params.size() == 7);
  std::vector<tb::TBInputOp *> input_ops;
  std::vector<tb::TBInputOp *> output_ops;
  int num_inputs = 7;
  int num_outputs = 2;

  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  for (auto const &op : bgraph.operators) {
    assert(op->op_type == mirage::type::TB_INPUT_OP);
    if (input_ops.size() < (size_t)num_inputs) {
      input_ops.push_back(static_cast<tb::TBInputOp *>(op));
    } else {
      output_ops.push_back(static_cast<tb::TBInputOp *>(op));
    }
  }
  assert(output_ops[0]->output_tensors[0].num_dims == 3); // lse
  assert(output_ops[1]->output_tensors[0].num_dims == 3); // output_tmp

  int qkv_stride = input_ops[0]->dtensor.dim[1];
  int num_q_heads = params[0];
  int num_kv_heads = params[1];
  int head_dim = input_ops[1]->output_tensors[0].dim[3];
  int output_size = head_dim * num_q_heads;
  int kv_stride = head_dim * num_kv_heads;
  int max_seq_len = params[4];
  int page_size = params[5];
  int num_kv_chunks = params[6];
  // Assert that k_cache has the same head_dim
  assert(input_ops[1]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[1]->output_tensors[0].dim[3]);
  assert(input_ops[2]->output_tensors[0].num_dims == 4);
  assert(head_dim == input_ops[2]->output_tensors[0].dim[3]);
  int max_tokens = input_ops[0]->dtensor.dim[0];
  constexpr int SEQ_LEN_PER_BLOCK = 256;

  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::multitoken_paged_attention_hopper_impl<bfloat16, $, "
         "$, $, $, $, $, "
         "$, $, $, $, $, $, $>(",
         num_q_heads / num_kv_heads, /* NUM_QO_HEADS */
         1,                          /* NUM_KV_HEADS */
         num_kv_heads,               /* NUM_QO_GROUPS */
         kv_stride,                  /* KV_CACHE_STRIDE */
         qkv_stride,                 /* QKV_STRIDE */
         output_size *
             num_kv_chunks, /* O_STRIDE (should consider num_kv_chunks) */
         head_dim,          /* HEAD_DIM */
         SEQ_LEN_PER_BLOCK, /* SEQ_LEN */
         max_seq_len,       /* MAX_SEQ_LEN */
         page_size,         /* PAGE_SIZE */
         max_tokens,        /* MAX_TOKENS */
         "true",            /* PARTITION_KV */
         num_kv_chunks);    /* NUM_KV_CHUNKS */
  code.e("    task_desc->input_ptrs[1],");
  code.e("    task_desc->input_ptrs[2],");
  code.e("    runtime_config.qo_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indptr_buffer,");
  code.e("    runtime_config.paged_kv_indices_buffer,");
  code.e("    runtime_config.paged_kv_last_page_len_buffer,");
  code.e("    task_desc->task_metadata.request_id,");
  code.e("    $,", params[2] > 0);
  code.e("    $,", params[3] > 0);
  code.e("    task_desc->input_ptrs[3],");
  code.e("    task_desc->input_ptrs[4],");
  code.e("    task_desc->input_ptrs[5],");
  code.e("    task_desc->input_ptrs[6],");
  code.e("    1e-6f,");
  code.e("    1e-6f,");
  code.e("    task_desc->input_ptrs[0],");
  code.e("    task_desc->output_ptrs[1],"); // output_tmp
  code.e("    task_desc->output_ptrs[0],"); // lse
  code.e("    task_desc->task_metadata.kv_idx);");
  return register_task_variant(TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER,
                               code.to_string());
}

} // namespace runtime
} // namespace mirage
