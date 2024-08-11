# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack.package import *


class PyTorchNvidiaApex(PythonPackage, CudaPackage):
    """A PyTorch Extension: Tools for easy mixed precision and
    distributed training in Pytorch"""

    homepage = "https://github.com/nvidia/apex/"
    git = "https://github.com/nvidia/apex/"

    license("BSD-3-Clause")

    version("master", branch="master")
    version("2020-10-19", commit="8a1ed9e8d35dfad26fb973996319965e4224dcdd")

    depends_on("python@3:", type=("build", "run"))
    depends_on("py-pip", type="build")
    depends_on("py-setuptools", type="build")
    depends_on("py-packaging", type="build")
    depends_on("py-torch@0.4:", type=("build", "run"))
    depends_on("cuda@9:", when="+cuda")
    depends_on("py-pybind11", type=("build", "link", "run"))
    depends_on("ninja", type="build")

    variant("cuda", default=True, description="Build with CUDA")
    variant("distributed_adam", when="+cuda", default=True, description="CUDA kernels for multi-tensor Adam optimizer")
    variant("distributed_lamb", when="+cuda", default=True, description="CUDA kernels for multi-tensor Lamb optimizer")
    variant("permutation_search", when="+cuda", default=True, description="CUDA kernels for permutation search")
    variant("bnp", when="+cuda", default=True, description="CUDA kernels for group batch normalization")
    variant("xentropy", when="+cuda", default=True, description="CUDA kernels for cross entropy")
    variant("focal_loss", when="+cuda", default=True, description="CUDA kernels for focal loss")
    variant("group_norm", when="+cuda", default=True, description="CUDA kernels for group normalization")
    variant("index_mul_2d", when="+cuda", default=True, description="CUDA kernels for index mul calculations")
    variant("deprecated_fused_adam", when="+cuda", default=False, description="CUDA kernels for fused Adam optimizer")
    variant("deprecated_fused_lamb", when="+cuda", default=False, description="CUDA kernels for fused Lamb optimizer")
    variant("fast_layer_norm", when="+cuda", default=True, description="CUDA kernels for Layer Norm")
    variant("fmha", when="+cuda", default=True, description="CUDA kernels for fused multihead attention")
    variant("fast_multihead_attn", when="+cuda", default=True, description="CUDA kernels for multihead attention")
    variant("transducer", when="+cuda", default=True, description="CUDA kernels for transducer joint/loss")
    variant("cudnn_gbn", when="+cuda", default=True, description="CUDA kernels for group normalization with cudnn")
    variant("peer_memory", when="+cuda", default=True, description="CUDA kernels for peer memory")
    variant("nccl_p2p", when="+cuda", default=True, description="NCCL point-to-point communication support")
    variant("fast_bottleneck", when="+cuda +peer_memory +nccl_p2p", default=True, description="CUDA kernels for bottleneck")
    variant("fused_conv_bias_relu", when="+cuda", default=True, description="CUDA kernels for fused conv bias relu with cudnn")
    variant("nccl_allocator", when="+cuda", default=True, description="NCCL allocator support")
    variant("gpu_direct_storage", when="+cuda", default=True, description="GPU direct storage support")

    depends_on("cuda@11:", when="+fmha")
    depends_on("cudnn@8.5:", when="+cudnn_gbn", type=("build", "link", "run"))
    depends_on("nccl@2.10.3:", when="+nccl_p2p", type=("build", "link", "run"))
    depends_on("cudnn@8.4:", when="+fast_bottleneck", type=("build", "link", "run"))
    depends_on("cudnn@8.4:", when="+fused_conv_bias_relu", type=("build", "link", "run"))
    depends_on("nccl@2.19:", when="+nccl_allocator", type=("build", "link", "run"))

    # https://github.com/NVIDIA/apex/issues/1498
    # https://github.com/NVIDIA/apex/pull/1499
    patch("1499.patch", when="@2020-10-19")

    def setup_build_environment(self, env):
        if "+cuda" in self.spec:
            env.set("CUDA_HOME", self.spec["cuda"].prefix)
        else:
            env.unset("CUDA_HOME")

    @when("^py-pip@:22.99")
    def global_options(self, spec, prefix):

        def append_global_args(spec, args, option):
            if f"+{option}" in spec:
                args.append("--{option}")
            return args

        args = []
        if spec.satisfies("^py-torch@1.0:"):
            args.append("--cpp_ext")
            if "+cuda" in spec:
                args.append("--cuda_ext")
                args = append_global_args(spec, args, "distributed_adam")
                args = append_global_args(spec, args, "distributed_lamb")
                args = append_global_args(spec, args, "permutation_search")
                args = append_global_args(spec, args, "bnp")
                args = append_global_args(spec, args, "xentropy")
                args = append_global_args(spec, args, "focal_loss")
                args = append_global_args(spec, args, "group_norm")
                args = append_global_args(spec, args, "index_mul_2d")
                args = append_global_args(spec, args, "deprecated_fused_adam")
                args = append_global_args(spec, args, "deprecated_fused_lamb")
                args = append_global_args(spec, args, "fast_layer_norm")
                args = append_global_args(spec, args, "fmha")
                args = append_global_args(spec, args, "fast_multihead_attn")
                args = append_global_args(spec, args, "transducer")
                args = append_global_args(spec, args, "cudnn_gbn")
                args = append_global_args(spec, args, "peer_memory")
                args = append_global_args(spec, args, "nccl_p2p")
                args = append_global_args(spec, args, "fast_bottleneck")
                args = append_global_args(spec, args, "fused_conv_bias_relu")
                args = append_global_args(spec, args, "nccl_allocator")
                args = append_global_args(spec, args, "gpu_direct_storage")
        return args

    @when("^py-pip@23:")
    def config_settings(self, spec, prefix):

        def append_config_args(spec, args, option):
            if f"+{option}" in spec:
                args = f"{args} --{option}"
            return args

        args = ""
        if spec.satisfies("^py-torch@1.0:"):
            args="--cpp_ext"
            if "+cuda" in spec:
                args = f"{args} --cuda_ext"
                args = append_config_args(spec, args, "distributed_adam")
                args = append_config_args(spec, args, "distributed_lamb")
                args = append_config_args(spec, args, "permutation_search")
                args = append_config_args(spec, args, "bnp")
                args = append_config_args(spec, args, "xentropy")
                args = append_config_args(spec, args, "focal_loss")
                args = append_config_args(spec, args, "group_norm")
                args = append_config_args(spec, args, "index_mul_2d")
                args = append_config_args(spec, args, "deprecated_fused_adam")
                args = append_config_args(spec, args, "deprecated_fused_lamb")
                args = append_config_args(spec, args, "fast_layer_norm")
                args = append_config_args(spec, args, "fmha")
                args = append_config_args(spec, args, "fast_multihead_attn")
                args = append_config_args(spec, args, "transducer")
                args = append_config_args(spec, args, "cudnn_gbn")
                args = append_config_args(spec, args, "peer_memory")
                args = append_config_args(spec, args, "nccl_p2p")
                args = append_config_args(spec, args, "fast_bottleneck")
                args = append_config_args(spec, args, "fused_conv_bias_relu")
                args = append_config_args(spec, args, "nccl_allocator")
                args = append_config_args(spec, args, "gpu_direct_storage")
        return {
            "builddir": "build",
            "compile-args": f"-j{make_jobs}",
            "--build-option": args,
        }
