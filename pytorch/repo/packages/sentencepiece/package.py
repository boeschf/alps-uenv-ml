# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack.package import *


class Sentencepiece(CMakePackage):
    """Unsupervised text tokenizer for Neural Network-based text generation.

    This is the C++ package."""

    homepage = "https://github.com/google/sentencepiece"
    url = "https://github.com/google/sentencepiece/archive/v0.1.85.tar.gz"

    maintainers("adamjstewart")

    license("Apache-2.0")

    version("0.2.0", sha256="9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86")
    version("0.1.99", sha256="63617eaf56c7a3857597dcd8780461f57dd21381b56a27716ef7d7e02e14ced4")
    version("0.1.98", sha256="e8e09beffacd9667ed40c4652306f7e7990100164dfa26d8bd8a66b097471cb2")
    version("0.1.97", sha256="41c3a07f315e3ac87605460c8bb8d739955bc8e7f478caec4017ef9b7d78669b")
    version("0.1.96", sha256="5198f31c3bb25e685e9e68355a3bf67a1db23c9e8bdccc33dc015f496a44df7a")
    version("0.1.95", sha256="1c0bd83e03f71a10fc934b7ce996e327488b838587f03159fd392c77c7701389")
    version("0.1.94", sha256="a23133caa67c38c3bf7f978fcea07947072783b32554a034cbbe99a8cf776192")
    version("0.1.93", sha256="778c5ab27f65bd427aef9b9404daed153e09b77ae21ce6e94b1d7fd91e7dec35")
    version("0.1.91", sha256="acbc7ea12713cd2a8d64892f8d2033c7fd2bb4faecab39452496120ace9a4b1b")
    version("0.1.85", sha256="dd4956287a1b6af3cbdbbd499b7227a859a4e3f41c9882de5e6bdd929e219ae6")

    variant("with-TCMalloc", default=True, description="Enable TCMalloc if available")
    variant("with-TCMalloc-static", default=False, description="Link static library of TCMALLOC")
    variant("no-tl", default=False, description="Disable thread_local operator")

    depends_on("cmake@3.1:", type="build")
    depends_on("gperftools", when="+with-TCMalloc")  # optional, 10-40% performance improvement

    def cmake_args(self):
        args = [
            self.define_from_variant("SPM_ENABLE_TCMALLOC", "with-TCMalloc"),
            self.define_from_variant("SPM_TCMALLOC_STATIC", "with-TCMalloc-static"),
            self.define_from_variant("SPM_NO_THREADLOCAL", "no-tl"),
        ]

        return args
