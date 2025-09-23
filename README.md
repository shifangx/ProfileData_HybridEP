
In order to reproduce the performance benchmark in [Hybrid-EP_Implementation.md](https://github.com/deepseek-ai/DeepEP/blob/hybrid-ep/Hybrid-EP_Implementation.md), please use DeepEP hybrid-ep branch with commit id [6ad6896](https://github.com/deepseek-ai/DeepEP/tree/6ad6896ddcc4f02531a008851006af1e191d9cd4) and run the scripts in  ([Scripts_0925_for_DeepEP-6ad6896](https://gitlab-master.nvidia.com/shifangx/ProfileData_HybridEP/-/tree/main/Scripts_0925_for_DeepEP-6ad6896)). Logs are stored in ([GB200_SM16](https://gitlab-master.nvidia.com/shifangx/ProfileData_HybridEP/-/tree/main/GB200_SM16), [GB200_SM32](https://gitlab-master.nvidia.com/shifangx/ProfileData_HybridEP/-/tree/main/GB200_SM32), [B200_SM16](https://gitlab-master.nvidia.com/shifangx/ProfileData_HybridEP/-/tree/main/B200_SM16)).

As shown in the following image, we cacluate duration and thoughput for torch API and kernel only respectively.

<img src="images/nsys-profile.png" width="400" style="height:auto;"/>

If you want to run tests with the version that includes JIT functionality, please use the scripts in [Scripts_1118_for_DeepEP-1dddd19](https://github.com/shifangx/ProfileData_HybridEP/tree/main/Scripts_1118_for_DeepEP-1dddd19).
