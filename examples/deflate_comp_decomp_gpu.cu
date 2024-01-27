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
 #include "nvcomp/bitcomp.h"
 #include "nvcomp/zstd.h"
 #include "nvcomp/gdeflate.h"
 #include <opencv2/opencv.hpp>  

 extern "C" {

    typedef struct {
        char* data;
        size_t* sizes;
        size_t size;
        char* compression;
        size_t total_bytes;
        size_t input_batch_size;
        size_t input_chunk_size;
    } CompressedVector;

    void concatanate_compressed_chunks(const std::vector<char*> data, std::vector<size_t> compressed_sizes_host,
                                       char* &data_array) {
        // Calculate the total length needed for the concatenated string
        size_t total_bytes = 0; // 1 for null terminator
        size_t index = 0;

        for(int i=0; i < compressed_sizes_host.size(); i++){
            total_bytes += compressed_sizes_host[i];
        }

        // std::cout << "pre cout before:: " << totalLength << std::endl;
        // Allocate memory for the concatenated string
        // char* data_array = new char[totalLength];
        
        for(int i = 0; i < data.size(); i++){
            for(int j = 0; j < compressed_sizes_host[i]; j++){
                data_array[index] = data[i][j];
                index++;
            }
        }
    }

    BatchDataCPU GetBatchDataCPU(const BatchData& batch_data, bool copy_data)
    {
        BatchDataCPU compress_data_cpu(
            batch_data.ptrs(),
            batch_data.sizes(),
            batch_data.data(),
            batch_data.size(),
            copy_data);
        return compress_data_cpu;
    }

    void convert_to_vector(char* data, std::vector<size_t> compressed_sizes_host, size_t batch_size, 
                           std::vector<char*> &comp_vector){
        
        size_t byte_size;
        size_t offset = 0;

        for (int i = 0; i < batch_size; i++){
            byte_size = compressed_sizes_host[i];
            char* tmp_buf = new char[byte_size];
            
            memcpy(tmp_buf, data + offset /* offset/ starting idx */, byte_size /* length */);
            comp_vector.push_back(tmp_buf);

            offset += byte_size;
        }
    }

    void destroy_mem(CompressedVector data){
        delete data.data;
        delete data.sizes;
    }

    // Benchmark performance from the binary data file fname
    CompressedVector run_example(CompressedVector image_vector)
    {
        std::vector<char> host_data(image_vector.data, image_vector.data + image_vector.size);
        std::vector<std::vector<char>> data; 
        std::vector<char*> comp_vector;
        data.push_back(host_data);

        size_t total_bytes = 0;
        for (const std::vector<char>& part : data) {
            total_bytes += part.size();
        }
        
        std::cout << "compression algorithm:: " << image_vector.compression << std::endl;
        std::cout << "----------" << std::endl;
        std::cout << "files: " << data.size() << std::endl;
        std::cout << "uncompressed (B): " << total_bytes << std::endl;
        
        const size_t chunk_size = 1 << 16;
        
        // build up metadata
        BatchData input_data(data, chunk_size);
        static nvcompBatchedDeflateOpts_t nvcompBatchedDeflateOpts = {0};
        // Compress on the GPU using batched API
        size_t comp_temp_bytes;
        nvcompStatus_t status = nvcompBatchedDeflateCompressGetTempSize(
            input_data.size(),
            chunk_size,
            nvcompBatchedDeflateOpts,
            &comp_temp_bytes);
        if( status != nvcompSuccess){
            throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetTempSize() not successful");
        }
        
        void* d_comp_temp;
        CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
        
        size_t max_out_bytes;
        status = nvcompBatchedDeflateCompressGetMaxOutputChunkSize(
            chunk_size, nvcompBatchedDeflateOpts, &max_out_bytes);
        if( status != nvcompSuccess){
            throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetMaxOutputChunkSize() not successful");
        }

        BatchData compress_data(max_out_bytes, input_data.size());
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, stream);
        
        // status = nvcompBatchedDeflateCompressAsync(
        //     input_data.ptrs(),
        //     input_data.sizes(),
        //     chunk_size,
        //     input_data.size(),
        //     d_comp_temp,
        //     comp_temp_bytes,
        //     compress_data.ptrs(),
        //     compress_data.sizes(),
        //     nvcompBatchedDeflateOpts,
        //     stream);
        // if (status != nvcompSuccess) {
        //     throw std::runtime_error("nvcompBatchedDeflateCompressAsync() failed.");
        // }

        status = nvcompBatchedDeflateCompressAsync(
            input_data.ptrs(),
            input_data.sizes(),
            chunk_size,
            input_data.size(),
            d_comp_temp,
            comp_temp_bytes,
            compress_data.ptrs(),
            compress_data.sizes(),
            nvcompBatchedDeflateOpts,
            stream);
        if (status != nvcompSuccess) {
            throw std::runtime_error("nvcompBatchedZstdCompressAsync() failed.");
        }
        
        cudaEventRecord(end, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // free compression memory
        cudaFree(d_comp_temp);
        
        float ms;
        cudaEventElapsedTime(&ms, start, end);
        
        // compute compression ratio
        std::vector<size_t> compressed_sizes_host(compress_data.size());
        cudaMemcpy(
            compressed_sizes_host.data(),
            compress_data.sizes(),
            compress_data.size() * sizeof(*compress_data.sizes()),
            cudaMemcpyDeviceToHost);
        
        size_t comp_bytes = 0;
        for (const size_t s : compressed_sizes_host) {
            comp_bytes += s;
        }
        
        std::cout << "comp_size: " << comp_bytes
                    << ", compressed ratio: " << std::fixed << std::setprecision(2)
                    << (double)total_bytes / comp_bytes << std::endl;
        std::cout << "compression throughput (GB/s): "
                    << (double)total_bytes / (1.0e6 * ms) << std::endl;
        
        // Allocate and prepare output/compressed batch
        BatchDataCPU compress_data_cpu = GetBatchDataCPU(compress_data, true);
        BatchDataCPU input_data_cpu = GetBatchDataCPU(input_data, true);


        CompressedVector example_data;
        example_data.data = new char[comp_bytes];
        example_data.sizes = new size_t[compress_data_cpu.size()];
        example_data.size = compress_data_cpu.size();
        example_data.total_bytes = comp_bytes;
        example_data.input_batch_size = input_data.size();
        example_data.input_chunk_size = chunk_size;

        int offset = 0;

        for(int i=0; i < compress_data_cpu.size(); i++){            
            
            offset += compressed_sizes_host[i];
            char* tmp_data = static_cast<char*>(compress_data_cpu.ptrs()[i]);
            comp_vector.push_back(tmp_data);
            
            example_data.sizes[i] = compress_data_cpu.sizes()[i];
        }

        cudaEventDestroy(start);
        cudaEventDestroy(end);
        cudaStreamDestroy(stream);
        
        concatanate_compressed_chunks(comp_vector, compressed_sizes_host, example_data.data);
        
        return example_data;
    }

    void run_decompression(CompressedVector compressed_vector){

        std::vector<size_t> comp_sizes(compressed_vector.sizes, compressed_vector.sizes + compressed_vector.size);

        std::vector<char*> reconverted_comp;
        convert_to_vector(compressed_vector.data, comp_sizes, compressed_vector.size, reconverted_comp);

        BatchData compressed_data(reconverted_comp, comp_sizes, compressed_vector.size);
        BatchData decomp_data(compressed_vector.input_chunk_size, compressed_vector.input_batch_size);

        // Create CUDA stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // CUDA events to measure decompression time
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        // deflate GPU decompression
        size_t decomp_temp_bytes;

        nvcompStatus_t status = nvcompBatchedDeflateDecompressGetTempSize(compressed_data.size(), compressed_vector.input_chunk_size, 
                                                                          &decomp_temp_bytes);
        if (status != nvcompSuccess) {
            throw std::runtime_error("nvcompBatchedDeflateDecompressGetTempSize() failed.");
        }
        
        void* d_decomp_temp;
        CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));
        
        size_t* d_decomp_sizes;
        CUDA_CHECK(cudaMalloc(&d_decomp_sizes, decomp_data.size() * sizeof(size_t)));
        
        nvcompStatus_t* d_status_ptrs;
        CUDA_CHECK(cudaMalloc(&d_status_ptrs, decomp_data.size() * sizeof(nvcompStatus_t)));
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Run decompression
        status = nvcompBatchedDeflateDecompressAsync(
            compressed_data.ptrs(),
            compressed_data.sizes(),
            decomp_data.sizes(),
            d_decomp_sizes,
            compressed_data.size(),
            d_decomp_temp,
            decomp_temp_bytes,
            decomp_data.ptrs(),
            d_status_ptrs,
            stream);
        if( status != nvcompSuccess){
            throw std::runtime_error("ERROR: nvcompBatchedDeflateDecompressAsync() not successful");
        }
        
        
        cudaFree(d_decomp_temp);
        
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        cudaStreamDestroy(stream);

        BatchDataCPU decomp_data_cpu(decomp_data.ptrs(), decomp_data.sizes(), decomp_data.data(), decomp_data.size(), true);
        

        // cv::Mat img(2048, 2592, CV_8UC3, decomp_data_cpu.data());
        // // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        // cv::imwrite("/home/benchmarker/myimage.png", img);
    }
 }