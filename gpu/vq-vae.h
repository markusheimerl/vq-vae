#ifndef VQVAE_H
#define VQVAE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include "../mlp/gpu/mlp.h"

typedef struct {
    // Encoder and Decoder MLPs
    MLP* encoder;
    MLP* decoder;
    
    // Codebook parameters
    float* d_codebook;           // [num_codebook_vectors x code_dim]
    
    // EMA for codebook updates
    float* d_codebook_ema_sum;   // [num_codebook_vectors x code_dim]
    float* d_codebook_ema_count; // [num_codebook_vectors]
    float ema_decay;
    float epsilon;
    
    // Forward pass buffers
    float* d_quantized;          // [batch_size x latent_dim]
    int* d_encoding_indices;     // [batch_size x num_codes]
    
    // Backward pass buffers
    float* d_grad_encoded;       // [batch_size x latent_dim]
    
    // Loss computation buffers
    float* d_loss_buffer;        // [batch_size] - reusable
    float* d_commitment_loss;    // [1]
    
    // Dimensions
    int input_dim;               // 3072 for CIFAR-10
    int latent_dim;              // num_codes * code_dim
    int hidden_dim;              // Hidden dimension for MLPs
    int num_codes;               // Number of code positions
    int code_dim;                // Dimension of each code
    int num_codebook_vectors;    // Size of codebook
    int batch_size;
    
    // Loss weights
    float beta;                  // Commitment loss weight
    
    // cuBLASLt handle
    cublasLtHandle_t cublaslt_handle;
} VQVAE;

// Function prototypes
VQVAE* init_vqvae(int input_dim, int latent_dim, int hidden_dim, int num_codes, int num_codebook_vectors, int batch_size, float beta, cublasLtHandle_t cublaslt_handle);
void free_vqvae(VQVAE* vqvae);
void forward_pass_vqvae(VQVAE* vqvae, float* d_input);
void calculate_losses_vqvae(VQVAE* vqvae, float* d_input, float* losses);
void zero_gradients_vqvae(VQVAE* vqvae);
void backward_pass_vqvae(VQVAE* vqvae, float* d_input);
void update_weights_vqvae(VQVAE* vqvae, float learning_rate);
void encode_vqvae(VQVAE* vqvae, float* d_input, int* d_indices);
void decode_vqvae(VQVAE* vqvae, int* d_indices);
void save_vqvae(VQVAE* vqvae, const char* filename);
VQVAE* load_vqvae(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);

#endif