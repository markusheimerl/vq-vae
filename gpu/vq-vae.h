#ifndef VQVAE_H
#define VQVAE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include "../transformer/gpu/transformer.h"

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLASLt Error checking macro
#ifndef CHECK_CUBLASLT
#define CHECK_CUBLASLT(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLASLt error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLASLt matrix multiplication macro
#ifndef LT_MATMUL
#define LT_MATMUL(vqvae, opA, opB, alpha, A, layA, B, layB, beta, C, layC) do { \
    cublasOperation_t _opA = opA, _opB = opB; \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(vqvae->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSA, &_opA, sizeof(_opA))); \
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(vqvae->matmul_desc, \
                   CUBLASLT_MATMUL_DESC_TRANSB, &_opB, sizeof(_opB))); \
    CHECK_CUBLASLT(cublasLtMatmul(vqvae->cublaslt_handle, vqvae->matmul_desc, \
                                  alpha, A, layA, B, layB, \
                                  beta, C, layC, \
                                  C, layC, NULL, NULL, 0, 0)); \
} while(0)
#endif

typedef struct {
    // Encoder and Decoder Transformers
    Transformer* encoder;
    Transformer* decoder;
    
    // Input/Output projections
    float* d_input_proj_W;      // [input_dim x latent_dim]
    float* d_input_proj_grad;
    float* d_input_proj_m;
    float* d_input_proj_v;
    
    float* d_output_proj_W;     // [latent_dim x input_dim]
    float* d_output_proj_grad;
    float* d_output_proj_m;
    float* d_output_proj_v;
    
    // Codebook parameters
    float* d_codebook;           // [num_codebook_vectors x code_dim]
    
    // EMA for codebook updates
    float* d_codebook_ema_sum;   // [num_codebook_vectors x code_dim]
    float* d_codebook_ema_count; // [num_codebook_vectors]
    float ema_decay;
    float epsilon;
    
    // Forward pass buffers
    float* d_projected_input;    // [batch_size x latent_dim]
    float* d_quantized;          // [batch_size x latent_dim]
    float* d_decoder_input;      // [batch_size x seq_len x d_model]
    float* d_reconstructed;      // [batch_size x input_dim]
    int* d_encoding_indices;     // [batch_size x num_codes]
    
    // Backward pass buffers
    float* d_grad_decoder_input;  // [batch_size x seq_len x d_model]
    float* d_grad_encoder_input;  // [batch_size x seq_len x d_model]
    float* d_grad_quantized;      // [batch_size x latent_dim]
    float* d_grad_input;          // [batch_size x input_dim]
    
    // Loss computation buffers
    float* d_commitment_loss;    // [1]
    
    // Adam parameters for projections
    float beta1, beta2, adam_epsilon;
    int t;
    float weight_decay;
    
    // Dimensions
    int input_dim;               // 3072 for CIFAR-10
    int latent_dim;              // num_codes * code_dim
    int seq_len;                 // Number of sequence positions (= num_codes)
    int d_model;                 // Dimension per position (= code_dim)
    int hidden_dim;              // Hidden dimension for transformers
    int num_layers;              // Number of transformer layers
    int num_codes;               // Number of code positions
    int code_dim;                // Dimension of each code
    int num_codebook_vectors;    // Size of codebook
    int batch_size;
    
    // Loss weights
    float beta;                  // Commitment loss weight
    
    // cuBLASLt handle and descriptors
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t input_layout;
    cublasLtMatrixLayout_t latent_layout;
    cublasLtMatrixLayout_t input_proj_weight_layout;
    cublasLtMatrixLayout_t output_proj_weight_layout;
} VQVAE;

// Function prototypes
VQVAE* init_vqvae(int input_dim, int latent_dim, int hidden_dim, int num_layers, int num_codes, int num_codebook_vectors, int batch_size, float beta, cublasLtHandle_t cublaslt_handle);
void free_vqvae(VQVAE* vqvae);
void forward_pass_vqvae(VQVAE* vqvae, float* d_input);
void calculate_losses_vqvae(VQVAE* vqvae, float* d_input, float* losses);
void zero_gradients_vqvae(VQVAE* vqvae);
void backward_pass_vqvae(VQVAE* vqvae, float* d_input);
void update_weights_vqvae(VQVAE* vqvae, float learning_rate);
void encode_vqvae(VQVAE* vqvae, float* d_input, int* d_indices);
void decode_vqvae(VQVAE* vqvae, int* d_indices, float* d_output);
void save_vqvae(VQVAE* vqvae, const char* filename);
VQVAE* load_vqvae(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);

#endif