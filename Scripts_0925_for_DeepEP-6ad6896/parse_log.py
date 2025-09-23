import re
import sys

# Check argument count
if len(sys.argv) != 2:
    print("Usage: python parse_log.py <log_filename>")
    print("for example: python parse_log.py ../logs/test_hybrid_ep_nvl72_N_8.log")
    sys.exit(1)
print(f"\n start parse log: {sys.argv[1]}")
log_filename = sys.argv[1]

# Read log file
with open(log_filename, 'r') as f:
    log = f.read()

#################################
# Parse torch API bandwidth
#################################
dispatch_fp8_torch_api_pattern = r"HybridEP dispatch torch API \(FP8\): ([\d.]+) GB/s"
dispatch_bf16_torch_api_pattern = r"HybridEP dispatch torch API \(BF16\): ([\d.]+) GB/s"
combine_torch_api_pattern = r"HybridEP combine torch API: ([\d.]+) GB/s"

dispatch_fp8_torch_api_bandwidths = [float(x) for x in re.findall(dispatch_fp8_torch_api_pattern, log)]
dispatch_bf16_torch_api_bandwidths = [float(x) for x in re.findall(dispatch_bf16_torch_api_pattern, log)]
combine_torch_api_bandwidths = [float(x) for x in re.findall(combine_torch_api_pattern, log)]

# Print statistics and parse results
if dispatch_fp8_torch_api_bandwidths:
    print(f"dispatch FP8 torch API bandwidth Average : {sum(dispatch_fp8_torch_api_bandwidths) / len(dispatch_fp8_torch_api_bandwidths) :.2f} GB/s, Max: {max(dispatch_fp8_torch_api_bandwidths) :.2f} GB/s, Min: {min(dispatch_fp8_torch_api_bandwidths) :.2f} GB/s")
else:
    print("dispatch FP8 torch API bandwidth is empty")
if dispatch_bf16_torch_api_bandwidths:
    print(f"dispatch BF16 torch API bandwidth Average : {sum(dispatch_bf16_torch_api_bandwidths) / len(dispatch_bf16_torch_api_bandwidths) :.2f} GB/s, Max: {max(dispatch_bf16_torch_api_bandwidths) :.2f} GB/s, Min: {min(dispatch_bf16_torch_api_bandwidths) :.2f} GB/s")
else:
    print("dispatch BF16 torch API bandwidth is empty")
if combine_torch_api_bandwidths:
    print(f"combine torch API bandwidth Average : {sum(combine_torch_api_bandwidths) / len(combine_torch_api_bandwidths) :.2f} GB/s, Max: {max(combine_torch_api_bandwidths) :.2f} GB/s, Min: {min(combine_torch_api_bandwidths) :.2f} GB/s")
else:
    print("combine torch API bandwidth is empty")

#################################
# Parse kernel bandwidth
#################################
dispatch_fp8_kernel_pattern = r"HybridEP dispatch kernel \(FP8\): ([\d.]+) GB/s"
dispatch_bf16_kernel_pattern = r"HybridEP dispatch kernel \(BF16\): ([\d.]+) GB/s"
combine_kernel_pattern = r"HybridEP combine kernel: ([\d.]+) GB/s"

dispatch_fp8_kernel_bandwidths = [float(x) for x in re.findall(dispatch_fp8_kernel_pattern, log)]
dispatch_bf16_kernel_bandwidths = [float(x) for x in re.findall(dispatch_bf16_kernel_pattern, log)]
combine_kernel_bandwidths = [float(x) for x in re.findall(combine_kernel_pattern, log)]

# Print statistics and parse results
if dispatch_fp8_kernel_bandwidths:
    print(f"dispatch FP8 kernel bandwidth Average : {sum(dispatch_fp8_kernel_bandwidths) / len(dispatch_fp8_kernel_bandwidths) :.2f} GB/s, Max: {max(dispatch_fp8_kernel_bandwidths) :.2f} GB/s, Min: {min(dispatch_fp8_kernel_bandwidths) :.2f} GB/s")
else:
    print("dispatch FP8 kernel bandwidth is empty")
if dispatch_bf16_kernel_bandwidths:
    print(f"dispatch BF16 kernel bandwidth Average : {sum(dispatch_bf16_kernel_bandwidths) / len(dispatch_bf16_kernel_bandwidths) :.2f} GB/s, Max: {max(dispatch_bf16_kernel_bandwidths) :.2f} GB/s, Min: {min(dispatch_bf16_kernel_bandwidths) :.2f} GB/s")
else:
    print("dispatch BF16 kernel bandwidth is empty")
if combine_kernel_bandwidths:
    print(f"combine kernel bandwidth Average : {sum(combine_kernel_bandwidths) / len(combine_kernel_bandwidths) :.2f} GB/s, Max: {max(combine_kernel_bandwidths) :.2f} GB/s, Min: {min(combine_kernel_bandwidths) :.2f} GB/s")
else:
    print("combine kernel bandwidth is empty")
