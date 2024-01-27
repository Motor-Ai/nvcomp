/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 #include "BatchData.h"
 #include "zlib.h"
 #include "nvcomp/deflate.h"
 #include "nvcomp/zstd.h"
 #include "nvcomp/gdeflate.h"
 #include "nvcomp/lz4.h"
 #include "nvcomp/cascaded.h"
 #include "nvcomp/bitcomp.h"
 #include "compression_pipeline_gpu.cuh"
 #include <opencv2/opencv.hpp>  

 extern "C" {

    void destroy_mem(CompressedVector data){
        delete data.data;
        delete data.sizes;
    }

    CompressedVector run_compression_pipeline(CompressedVector image_vector){
        
        // deflate
        static nvcompBatchedDeflateOpts_t nvcompBatchedDeflateOpts = {0};
        // gdeflate
        static nvcompBatchedGdeflateOpts_t nvcompBatchedGdeflateOpts = {0};
        // zstd
        static nvcompBatchedZstdOpts_t nvcompBatchedZstdTestOpts{};
        // lz4
        static nvcompBatchedLZ4Opts_t nvcompBatchedLZ4TestOpts{NVCOMP_TYPE_CHAR};
        // cascaded
        static nvcompBatchedCascadedOpts_t nvcompBatchedCascadedTestOpts = {4096, NVCOMP_TYPE_UINT, 2, 1, 1};
        // bitcomp
        static nvcompBatchedBitcompFormatOpts nvcompBatchedBitcompOpts = {0, NVCOMP_TYPE_UCHAR};

        std::string compression_algo(image_vector.compression);

        if (compression_algo == "deflate"){
            return run_compressor(nvcompBatchedDeflateOpts, nvcompBatchedDeflateCompressGetTempSize, 
                                  nvcompBatchedDeflateCompressGetMaxOutputChunkSize, 
                                  nvcompBatchedDeflateCompressAsync, image_vector);
        }

        else if (compression_algo == "gdeflate"){
            return run_compressor(nvcompBatchedGdeflateOpts, nvcompBatchedGdeflateCompressGetTempSize,
                                  nvcompBatchedGdeflateCompressGetMaxOutputChunkSize,
                                  nvcompBatchedGdeflateCompressAsync, image_vector);
        }

        else if (compression_algo == "zstd"){
            return run_compressor(nvcompBatchedZstdTestOpts, nvcompBatchedZstdCompressGetTempSize,
                                  nvcompBatchedZstdCompressGetMaxOutputChunkSize,
                                  nvcompBatchedZstdCompressAsync, image_vector);
        }

        else if (compression_algo == "lz4"){
            return run_compressor(nvcompBatchedLZ4TestOpts, nvcompBatchedLZ4CompressGetTempSize,
                                  nvcompBatchedLZ4CompressGetMaxOutputChunkSize,
                                  nvcompBatchedLZ4CompressAsync, image_vector);
        }

        else if (compression_algo == "cascaded"){
            return run_compressor(nvcompBatchedCascadedTestOpts, nvcompBatchedCascadedCompressGetTempSize,
                                  nvcompBatchedCascadedCompressGetMaxOutputChunkSize,
                                  nvcompBatchedCascadedCompressAsync, image_vector);
        }

        else if (compression_algo == "bitcomp"){
            return run_compressor(nvcompBatchedBitcompOpts, nvcompBatchedBitcompCompressGetTempSize,
                                  nvcompBatchedBitcompCompressGetMaxOutputChunkSize,
                                  nvcompBatchedBitcompCompressAsync, image_vector);
        }

        else{
            std::cout << "compression algorithm:: " << image_vector.compression << std::endl;
            throw std::invalid_argument( "the compression algorithm is not familiar/available" );
        }
        
    }
 }