#include <iostream>
#include "BatchData.h"
#include "zlib.h"
#include "nvcomp/deflate.h"
#include "nvcomp/zstd.h"
#include "nvcomp/gdeflate.h"


extern "C++"{
    typedef struct {
        char* data;
        size_t* sizes;
        size_t size;
        char* compression;
        size_t total_bytes;
        size_t input_batch_size;
        size_t input_chunk_size;
        size_t input_total_bytes;
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
            std::cout << "compressed data:: " << data[i] << std::endl;
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

    // // Benchmark performance from the binary data file fname
    // template<typename FormatOptsT, typename CompGetTempT, typename CompGetSizeT, typename CompAsyncT>
    // CompressedVector run_compressor(FormatOptsT format_opts,
    //                                 CompGetTempT BatchedCompressGetTempSize,
    //                                 CompGetSizeT BatchedCompressGetMaxOutputChunkSize,
    //                                 CompAsyncT BatchedCompressAsync, 
    //                                 CompressedVector image_vector)
    // {
    //     std::vector<char> host_data(image_vector.data, image_vector.data + image_vector.size);
    //     std::vector<std::vector<char>> data; 
    //     std::vector<char*> comp_vector;
    //     data.push_back(host_data);

    //     size_t total_bytes = 0;
    //     for (const std::vector<char>& part : data) {
    //         total_bytes += part.size();
    //     }
        
    //     std::cout << "compression algorithm:: " << image_vector.compression << std::endl;
    //     std::cout << "----------" << std::endl;
    //     std::cout << "total frames: " << data.size() << std::endl;
    //     std::cout << "uncompressed (B): " << total_bytes << std::endl;
        
    //     const size_t chunk_size = 1 << 16;
        
    //     // build up metadata
    //     BatchData input_data(data, chunk_size);
    //     // BatchData input_data(data);
    //     // static FormatOptsT nvcompBatchedDeflateOpts = {0};
    //     // Compress on the GPU using batched API
    //     size_t comp_temp_bytes;
    //     nvcompStatus_t status = BatchedCompressGetTempSize(
    //         input_data.size(),
    //         chunk_size,
    //         format_opts,
    //         &comp_temp_bytes);
    //     if( status != nvcompSuccess){
    //         throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetTempSize() not successful");
    //     }
        
    //     void* d_comp_temp;
    //     CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));
        
    //     size_t max_out_bytes;
    //     status = BatchedCompressGetMaxOutputChunkSize(chunk_size, 
    //                                                   format_opts, &max_out_bytes);

    //     if( status != nvcompSuccess){
    //         throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetMaxOutputChunkSize() not successful");
    //     }

    //     BatchData compress_data(max_out_bytes, input_data.size());
        
    //     cudaStream_t stream;
    //     cudaStreamCreate(&stream);
        
    //     cudaEvent_t start, end;
    //     cudaEventCreate(&start);
    //     cudaEventCreate(&end);
    //     cudaEventRecord(start, stream);

    //     status = BatchedCompressAsync(
    //         input_data.ptrs(),
    //         input_data.sizes(),
    //         chunk_size,
    //         input_data.size(),
    //         d_comp_temp,
    //         comp_temp_bytes,
    //         compress_data.ptrs(),
    //         compress_data.sizes(),
    //         format_opts,
    //         stream);
    //     if (status != nvcompSuccess) {
    //         throw std::runtime_error("nvcompBatchedZstdCompressAsync() failed.");
    //     }
        
    //     cudaEventRecord(end, stream);
    //     CUDA_CHECK(cudaStreamSynchronize(stream));
        
    //     // free compression memory
    //     cudaFree(d_comp_temp);
        
    //     float ms;
    //     cudaEventElapsedTime(&ms, start, end);
        
    //     // compute compression ratio
    //     std::vector<size_t> compressed_sizes_host(compress_data.size());
    //     cudaMemcpy(
    //         compressed_sizes_host.data(),
    //         compress_data.sizes(),
    //         compress_data.size() * sizeof(*compress_data.sizes()),
    //         cudaMemcpyDeviceToHost);
        
    //     size_t comp_bytes = 0;
    //     for (const size_t s : compressed_sizes_host) {
    //         comp_bytes += s;
    //     }
        
    //     std::cout << "comp_size: " << comp_bytes
    //                 << ", compressed ratio: " << std::fixed << std::setprecision(2)
    //                 << comp_bytes / (double)total_bytes << std::endl;
    //     std::cout << "compression throughput (GB/s): "
    //                 << (double)total_bytes / (1.0e6 * ms) << std::endl;
        
    //     // Allocate and prepare output/compressed batch
    //     BatchDataCPU compress_data_cpu = GetBatchDataCPU(compress_data, true);
    //     BatchDataCPU input_data_cpu = GetBatchDataCPU(input_data, true);


    //     CompressedVector example_data;
    //     example_data.data = new char[comp_bytes];
    //     example_data.sizes = new size_t[compress_data_cpu.size()];
    //     example_data.size = compress_data_cpu.size();
    //     example_data.total_bytes = comp_bytes;
    //     example_data.input_batch_size = input_data.size();
    //     example_data.input_chunk_size = chunk_size;

    //     int offset = 0;

    //     for(int i=0; i < compress_data_cpu.size(); i++){            
            
    //         offset += compressed_sizes_host[i];
    //         char* tmp_data = static_cast<char*>(compress_data_cpu.ptrs()[i]);
    //         comp_vector.push_back(tmp_data);
            
    //         example_data.sizes[i] = compress_data_cpu.sizes()[i];
    //     }

    //     cudaEventDestroy(start);
    //     cudaEventDestroy(end);
    //     cudaStreamDestroy(stream);
        
    //     concatanate_compressed_chunks(comp_vector, compressed_sizes_host, example_data.data);
        
    //     return example_data;
    // }

    // Benchmark performance from the binary data file fname
    template<typename FormatOptsT, typename CompGetTempT, typename CompGetSizeT, typename CompAsyncT>
    CompressedVector run_compressor(FormatOptsT format_opts,
                                    CompGetTempT BatchedCompressGetTempSize,
                                    CompGetSizeT BatchedCompressGetMaxOutputChunkSize,
                                    CompAsyncT BatchedCompressAsync, 
                                    CompressedVector image_vector)
    {
        std::vector<char> host_data(image_vector.data, image_vector.data + image_vector.size);
        std::vector<std::vector<char>> data;
        std::vector<char*> comp_vector;
        data.push_back(host_data);

        size_t total_bytes = 0;
        size_t chunk_size = 0;
        for (const std::vector<char>& part : data) {
            total_bytes += part.size();
            if (part.size() > chunk_size) {
                chunk_size = part.size();
            }
        }
        
        std::cout << "compression algorithm:: " << image_vector.compression << std::endl;
        std::cout << "----------" << std::endl;
        std::cout << "files: " << data.size() << std::endl;
        std::cout << "uncompressed (B): " << total_bytes << std::endl;

        // build up metadata
        BatchData input_data(data);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        const size_t batch_size = input_data.size();

        std::vector<size_t> h_input_sizes(batch_size);
        CUDA_CHECK(cudaMemcpy(h_input_sizes.data(), input_data.sizes(),
                   sizeof(size_t)*batch_size, cudaMemcpyDeviceToHost));
        
        // compression
        nvcompStatus_t status;

        // Compress on the GPU using batched API
        size_t comp_temp_bytes;
        status = BatchedCompressGetTempSize(batch_size, chunk_size, format_opts, &comp_temp_bytes);
        
        if( status != nvcompSuccess){
            throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetTempSize() not successful");
        }
        

        void* d_comp_temp;
        CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

        size_t max_out_bytes;
        status = BatchedCompressGetMaxOutputChunkSize(chunk_size, format_opts, &max_out_bytes);
        
        if( status != nvcompSuccess){
            throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetMaxOutputChunkSize() not successful");
        }
        

        BatchData compress_data(max_out_bytes, batch_size);

        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        CUDA_CHECK(cudaEventRecord(start, stream));

        status = BatchedCompressAsync(
            input_data.ptrs(),
            input_data.sizes(),
            chunk_size,
            batch_size,
            d_comp_temp,
            comp_temp_bytes,
            compress_data.ptrs(),
            compress_data.sizes(),
            format_opts,
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
                    << comp_bytes / (double)total_bytes << std::endl;
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
        example_data.input_total_bytes = total_bytes;
        
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
} 