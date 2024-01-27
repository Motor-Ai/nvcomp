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
            for(int j = 0; j < compressed_sizes_host[i]; j++){
                data_array[index] = data[i][j];
                index++;
            }
        }
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

    template<typename FormatOptsT, typename DecompGetTempT, typename DecompAsyncT>
    CompressedVector run_decompressor(FormatOptsT format_opts,
                                      DecompGetTempT BatchedDecompressGetTempSize,
                                      DecompAsyncT BatchedDecompressAsync, 
                                      CompressedVector compressed_vector){
        
        std::vector<size_t> comp_sizes(compressed_vector.sizes, compressed_vector.sizes + compressed_vector.size);

        std::vector<char*> reconverted_comp;
        std::vector<char*> decomp_vector;
        std::vector<size_t> decompressed_sizes_host;
        size_t chunk_size = compressed_vector.input_chunk_size;
        size_t batch_size = compressed_vector.input_batch_size;

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

        // compression
        nvcompStatus_t status;
        size_t decomp_temp_bytes;
        status = BatchedDecompressGetTempSize(compressed_data.size(), chunk_size, &decomp_temp_bytes);
        
        if( status != nvcompSuccess){
            throw std::runtime_error("ERROR: nvcompBatchedDeflateCompressGetTempSize() not successful");
        }

        void* d_decomp_temp;
        CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

        size_t* d_decomp_sizes;
        CUDA_CHECK(cudaMalloc(&d_decomp_sizes, batch_size*sizeof(*d_decomp_sizes)));

        nvcompStatus_t* d_decomp_statuses;
        CUDA_CHECK(cudaMalloc(&d_decomp_statuses, batch_size*sizeof(*d_decomp_statuses)));

        // std::vector<void*> h_output_ptrs(batch_size);
        // for (size_t i = 0; i < batch_size; ++i) {
        //     CUDA_CHECK(cudaMalloc(&h_output_ptrs[i], comp_sizes[i]));
        // }
        // void ** d_output_ptrs;
        // CUDA_CHECK(cudaMalloc(&d_output_ptrs, sizeof(*d_output_ptrs)*batch_size));
        // CUDA_CHECK(cudaMemcpy(d_output_ptrs, h_output_ptrs.data(), sizeof(*d_output_ptrs)*batch_size, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start, stream));
        status = BatchedDecompressAsync(
            compressed_data.ptrs(),
            compressed_data.sizes(),
            decomp_data.sizes(),
            d_decomp_sizes,
            batch_size,
            d_decomp_temp,
            decomp_temp_bytes,
            decomp_data.ptrs(),
            d_decomp_statuses,
            stream);
        
        if( status != nvcompSuccess){
            throw std::runtime_error("ERROR: nvcompBatchedDeflateDecompressAsync() not successful");
        }

        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaFree(d_decomp_temp);

        BatchDataCPU decomp_data_cpu(decomp_data.ptrs(), decomp_data.sizes(), decomp_data.data(), 
                                     decomp_data.size(), true);

        CompressedVector example_data;
        example_data.data = new char[compressed_vector.input_total_bytes];
        example_data.sizes = new size_t[decomp_data_cpu.size()];
        example_data.size = decomp_data_cpu.size();
        example_data.total_bytes = compressed_vector.input_total_bytes;
        example_data.input_batch_size = compressed_data.size();
        example_data.input_chunk_size = compressed_vector.input_chunk_size;

        int offset = 0;

        for(int i=0; i < decomp_data_cpu.size(); i++){            
            
            offset += decomp_data_cpu.sizes()[i];
            char* tmp_data = static_cast<char*>(decomp_data_cpu.ptrs()[i]);
            decomp_vector.push_back(tmp_data);
            
            example_data.sizes[i] = decomp_data_cpu.sizes()[i];
            decompressed_sizes_host.push_back(decomp_data_cpu.sizes()[i]);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(end);
        cudaStreamDestroy(stream);
        
        concatanate_compressed_chunks(decomp_vector, decompressed_sizes_host, example_data.data);

        return example_data;
    }
 
 }