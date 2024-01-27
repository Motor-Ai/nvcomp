 #include "BatchData.h"
 #include "zlib.h"
 #include "nvcomp/deflate.h"
 #include "nvcomp/zstd.h"
 #include "nvcomp/gdeflate.h"
 #include "nvcomp/lz4.h"
 #include "nvcomp/cascaded.h"
 #include "nvcomp/bitcomp.h"
 #include "decompression_pipeline_gpu.cuh"
 #include <opencv2/opencv.hpp> 

 extern "C" {
    void destroy_mem(CompressedVector data){
        delete data.data;
        delete data.sizes;
    }

    CompressedVector run_decompression_pipeline(CompressedVector compressed_vector){
        
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

        std::string compression_algo(compressed_vector.compression);

        if (compression_algo == "deflate"){
            return run_decompressor(nvcompBatchedDeflateOpts, nvcompBatchedDeflateDecompressGetTempSize,
                                    nvcompBatchedDeflateDecompressAsync, compressed_vector);
        }

        else if (compression_algo == "gdeflate"){
            return run_decompressor(nvcompBatchedGdeflateOpts, nvcompBatchedGdeflateDecompressGetTempSize,
                                    nvcompBatchedGdeflateDecompressAsync, compressed_vector);
        }

        else if (compression_algo == "zstd"){
            return run_decompressor(nvcompBatchedZstdTestOpts, nvcompBatchedZstdDecompressGetTempSize,
                                    nvcompBatchedZstdDecompressAsync, compressed_vector);
        }

        else if (compression_algo == "lz4"){
            return run_decompressor(nvcompBatchedLZ4TestOpts, nvcompBatchedLZ4DecompressGetTempSize,
                                    nvcompBatchedLZ4DecompressAsync, compressed_vector);
        }

        else if (compression_algo == "cascaded"){
            return run_decompressor(nvcompBatchedCascadedTestOpts, nvcompBatchedCascadedDecompressGetTempSize,
                                    nvcompBatchedCascadedDecompressAsync, compressed_vector);
        }

        else if (compression_algo == "bitcomp"){
            return run_decompressor(nvcompBatchedBitcompOpts, nvcompBatchedBitcompDecompressGetTempSize,
                                    nvcompBatchedBitcompDecompressAsync, compressed_vector);
        }

        else{
            std::cout << "compression algorithm:: " << compressed_vector.compression << std::endl;
            throw std::invalid_argument( "the compression algorithm is not familiar/available" );
        }
        
    }
 }
 