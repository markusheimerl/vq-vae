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
    
    // Patch embedding projection
    float* d_patch_proj_W;      // [patch_dim x d_model]
    float* d_patch_proj_grad;
    float* d_patch_proj_m;
    float* d_patch_proj_v;
    
    // Output projection (from patches back to pixels)
    float* d_output_proj_W;     // [d_model x patch_dim]
    float* d_output_proj_grad;
    float* d_output_proj_m;
    float* d_output_proj_v;
    
    // Codebook parameters
    float* d_codebook;           // [num_codebook_vectors x d_model]
    
    // EMA for codebook updates
    float* d_codebook_ema_sum;   // [num_codebook_vectors x d_model]
    float* d_codebook_ema_count; // [num_codebook_vectors]
    float ema_decay;
    float epsilon;
    
    // Forward pass buffers
    float* d_patches;            // [batch_size x num_patches x patch_dim]
    float* d_patch_embeddings;   // [batch_size x num_patches x d_model]
    float* d_quantized;          // [batch_size x num_patches x d_model]
    float* d_decoder_input;      // [batch_size x num_patches x d_model]
    float* d_reconstructed_patches; // [batch_size x num_patches x patch_dim]
    float* d_reconstructed;      // [batch_size x img_height x img_width x 3]
    int* d_encoding_indices;     // [batch_size x num_patches]
    
    // Backward pass buffers
    float* d_grad_decoder_input;  // [batch_size x num_patches x d_model]
    float* d_grad_encoder_input;  // [batch_size x num_patches x d_model]
    float* d_grad_quantized;      // [batch_size x num_patches x d_model]
    float* d_grad_patches;        // [batch_size x num_patches x patch_dim]
    float* d_grad_input;          // [batch_size x img_height x img_width x 3]
    
    // Loss computation buffers
    float* d_commitment_loss;    // [1]
    
    // Adam parameters for projections
    float beta1, beta2, adam_epsilon;
    int t;
    float weight_decay;
    
    // Dimensions
    int img_height;              // 32 for CIFAR-10
    int img_width;               // 32 for CIFAR-10
    int img_channels;            // 3 for CIFAR-10
    int patch_height;            // e.g., 4
    int patch_width;             // e.g., 4
    int num_patches;             // (img_height/patch_height) * (img_width/patch_width)
    int patch_dim;               // patch_height * patch_width * img_channels
    int d_model;                 // Transformer dimension
    int hidden_dim;              // Hidden dimension for transformers
    int num_layers;              // Number of transformer layers
    int num_codebook_vectors;    // Size of codebook
    int batch_size;
    
    // Loss weights
    float beta;                  // Commitment loss weight
    
    // cuBLASLt handle and descriptors
    cublasLtHandle_t cublaslt_handle;
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t patch_proj_weight_layout;
    cublasLtMatrixLayout_t output_proj_weight_layout;
    cublasLtMatrixLayout_t patches_layout;
    cublasLtMatrixLayout_t embeddings_layout;
} VQVAE;

// Function prototypes
VQVAE* init_vqvae(int img_height, int img_width, int img_channels, int patch_height, int patch_width,
                  int d_model, int hidden_dim, int num_layers, int num_codebook_vectors, 
                  int batch_size, float beta, cublasLtHandle_t cublaslt_handle);
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