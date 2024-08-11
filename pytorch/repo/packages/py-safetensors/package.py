# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class PySafetensors(PythonPackage):
    """Fast and Safe Tensor serialization."""

    homepage = "https://github.com/huggingface/safetensors"
    pypi = "safetensors/safetensors-0.3.1.tar.gz"

    version("0.4.1", sha256="2304658e6ada81a5223225b4efe84748e760c46079bffedf7e321763cafb36c9")
    version("0.3.1", sha256="571da56ff8d0bec8ae54923b621cda98d36dcef10feb36fd492c4d0c2cd0e869")

    depends_on("py-setuptools", when="@:0.4", type="build")
    depends_on("py-setuptools-rust", when="@:0.4", type="build")
    depends_on("py-maturin@1:", when="@0.4.1:", type="build")
