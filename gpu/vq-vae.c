#include "vq-vae.h"

// CUDA kernel for adding positional encoding
__global__ static void add_positional_encoding_kernel(float* data,
                                                       int batch_size, int seq_len, int d_model) {
    int batch_idx = blockIdx.x;
    int pos = blockIdx.y;
    
    if (batch_idx >= batch_size || pos >= seq_len) return;
    
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        int idx = batch_idx * seq_len * d_model + pos * d_model + d;
        
        // Compute positional encoding on the fly
        float angle = pos / powf(10000.0f, 2.0f * floorf(d / 2.0f) / (float)d_model);
        float pos_enc = (d % 2 == 0) ? sinf(angle) : cosf(angle);
        
        data[idx] += pos_enc;
    }
}

// CUDA kernel for vector quantization
__global__ static void quantize_kernel(float* quantized, int* indices, 
                                      float* encoded, float* codebook,
                                      int batch_size, int num_codes,
                                      int code_dim, int num_codebook_vectors) {
    int batch_idx = blockIdx.x;
    int code_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || code_idx >= num_codes) return;
    
    extern __shared__ float shared_mem[];
    float* shared_enc = shared_mem;
    float* shared_dists = &shared_mem[code_dim];
    int* shared_indices = (int*)&shared_dists[blockDim.x];
    
    // Load encoded vector to shared memory
    int enc_offset = batch_idx * num_codes * code_dim + code_idx * code_dim;
    for (int d = threadIdx.x; d < code_dim; d += blockDim.x) {
        shared_enc[d] = encoded[enc_offset + d];
    }
    __syncthreads();
    
    // Each thread handles multiple codebook entries
    float min_dist = 1e30f;
    int min_idx = 0;
    
    for (int k = threadIdx.x; k < num_codebook_vectors; k += blockDim.x) {
        float dist = 0.0f;
        
        #pragma unroll 8
        for (int d = 0; d < code_dim; d++) {
            float diff = shared_enc[d] - codebook[k * code_dim + d];
            dist += diff * diff;
        }
        
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = k;
        }
    }
    
    // Store results in shared memory
    shared_dists[threadIdx.x] = min_dist;
    shared_indices[threadIdx.x] = min_idx;
    __syncthreads();
    
    // Parallel reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_dists[threadIdx.x + stride] < shared_dists[threadIdx.x]) {
                shared_dists[threadIdx.x] = shared_dists[threadIdx.x + stride];
                shared_indices[threadIdx.x] = shared_indices[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (threadIdx.x == 0) {
        indices[batch_idx * num_codes + code_idx] = shared_indices[0];
    }
    
    // All threads cooperate to write quantized vector
    int best_idx = shared_indices[0];
    for (int d = threadIdx.x; d < code_dim; d += blockDim.x) {
        quantized[enc_offset + d] = codebook[best_idx * code_dim + d];
    }
}

// CUDA kernel for commitment loss computation
__global__ static void compute_commitment_loss_kernel(float* encoded, float* quantized,
                                                       float* commitment_loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float diff = encoded[idx] - quantized[idx];
    atomicAdd(commitment_loss, diff * diff);
}

// CUDA kernel for straight-through gradient
__global__ static void straight_through_gradient_kernel(float* grad_encoded, 
                                                        float* grad_decoder_input,
                                                        float* encoded, float* quantized,
                                                        float beta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Straight-through: copy gradient + commitment loss gradient
    grad_encoded[idx] = grad_decoder_input[idx] + beta * (encoded[idx] - quantized[idx]);
}

// CUDA kernel for EMA codebook update
__global__ static void ema_update_kernel(float* codebook, float* ema_sum, float* ema_count,
                                         int* indices, float* encoded,
                                         int batch_size, int num_codes, int code_dim,
                                         float decay, float epsilon) {
    int k = blockIdx.x;
    int d = threadIdx.x;
    
    if (d >= code_dim) return;
    
    // Step 1: Decay previous EMA
    if (d == 0) ema_count[k] *= decay;
    ema_sum[k * code_dim + d] *= decay;
    
    __syncthreads();
    
    // Step 2: Accumulate new values
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < num_codes; c++) {
            if (indices[b * num_codes + c] == k) {
                if (d == 0) atomicAdd(&ema_count[k], 1.0f);
                atomicAdd(&ema_sum[k * code_dim + d], encoded[b * num_codes * code_dim + c * code_dim + d]);
            }
        }
    }
    
    __syncthreads();
    
    // Step 3: Update codebook
    float count = ema_count[k] + epsilon;
    codebook[k * code_dim + d] = ema_sum[k * code_dim + d] / count;
}

// CUDA kernel for codebook lookup
__global__ static void codebook_lookup_kernel(float* output, int* indices, float* codebook,
                                              int batch_size, int num_codes, int code_dim) {
    int batch_idx = blockIdx.x;
    int code_idx = blockIdx.y;
    int d = threadIdx.x;
    
    if (batch_idx >= batch_size || code_idx >= num_codes || d >= code_dim) return;
    
    int codebook_idx = indices[batch_idx * num_codes + code_idx];
    output[batch_idx * num_codes * code_dim + code_idx * code_dim + d] = codebook[codebook_idx * code_dim + d];
}

// CUDA kernel for reconstruction loss and gradient
__global__ static void compute_reconstruction_loss_and_gradient_kernel(float* grad_output, float* predictions, 
                                                                        float* targets, float* loss_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        grad_output[idx] = diff;
        atomicAdd(loss_result, diff * diff);
    }
}

// CUDA kernel for AdamW update
__global__ static void adamw_update_kernel(float* weight, float* grad, float* m, float* v,
                                           float beta1, float beta2, float epsilon, float learning_rate,
                                           float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Initialize VQ-VAE
VQVAE* init_vqvae(int input_dim, int latent_dim, int hidden_dim, int num_layers, int num_codes, 
                  int num_codebook_vectors, int batch_size, float beta, cublasLtHandle_t cublaslt_handle) {
    VQVAE* vqvae = (VQVAE*)malloc(sizeof(VQVAE));
    
    vqvae->input_dim = input_dim;
    vqvae->latent_dim = latent_dim;
    vqvae->hidden_dim = hidden_dim;
    vqvae->num_layers = num_layers;
    vqvae->num_codes = num_codes;
    vqvae->code_dim = latent_dim / num_codes;
    vqvae->seq_len = num_codes;
    vqvae->d_model = vqvae->code_dim;
    vqvae->num_codebook_vectors = num_codebook_vectors;
    vqvae->batch_size = batch_size;
    vqvae->beta = beta;
    vqvae->ema_decay = 0.99f;
    vqvae->epsilon = 1e-5f;
    vqvae->cublaslt_handle = cublaslt_handle;
    
    // Adam parameters
    vqvae->beta1 = 0.9f;
    vqvae->beta2 = 0.999f;
    vqvae->adam_epsilon = 1e-8f;
    vqvae->t = 0;
    vqvae->weight_decay = 0.01f;
    
    // Initialize encoder and decoder transformers
    vqvae->encoder = init_transformer(vqvae->seq_len, vqvae->d_model, hidden_dim, num_layers, batch_size, false, cublaslt_handle);
    vqvae->decoder = init_transformer(vqvae->seq_len, vqvae->d_model, hidden_dim, num_layers, batch_size, false, cublaslt_handle);
    
    // Allocate projection weights
    int input_proj_size = input_dim * latent_dim;
    int output_proj_size = latent_dim * input_dim;
    
    float* h_input_proj_W = (float*)malloc(input_proj_size * sizeof(float));
    float* h_output_proj_W = (float*)malloc(output_proj_size * sizeof(float));
    
    float scale_in = 1.0f / sqrtf(input_dim);
    float scale_out = 1.0f / sqrtf(latent_dim);
    
    for (int i = 0; i < input_proj_size; i++) {
        h_input_proj_W[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
    }
    for (int i = 0; i < output_proj_size; i++) {
        h_output_proj_W[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_out;
    }
    
    CHECK_CUDA(cudaMalloc(&vqvae->d_input_proj_W, input_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_input_proj_grad, input_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_input_proj_m, input_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_input_proj_v, input_proj_size * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&vqvae->d_output_proj_W, output_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_output_proj_grad, output_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_output_proj_m, output_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_output_proj_v, output_proj_size * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(vqvae->d_input_proj_W, h_input_proj_W, input_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(vqvae->d_output_proj_W, h_output_proj_W, output_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemset(vqvae->d_input_proj_m, 0, input_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(vqvae->d_input_proj_v, 0, input_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(vqvae->d_output_proj_m, 0, output_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(vqvae->d_output_proj_v, 0, output_proj_size * sizeof(float)));
    
    free(h_input_proj_W);
    free(h_output_proj_W);
    
    // Allocate codebook
    int codebook_size = num_codebook_vectors * vqvae->code_dim;
    CHECK_CUDA(cudaMalloc(&vqvae->d_codebook, codebook_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_codebook_ema_sum, codebook_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_codebook_ema_count, num_codebook_vectors * sizeof(float)));
    
    // Initialize codebook
    float* h_codebook = (float*)malloc(codebook_size * sizeof(float));
    float scale = 1.0f / sqrtf(vqvae->code_dim);
    for (int i = 0; i < codebook_size; i++) {
        h_codebook[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    CHECK_CUDA(cudaMemcpy(vqvae->d_codebook, h_codebook, codebook_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_codebook);
    
    CHECK_CUDA(cudaMemset(vqvae->d_codebook_ema_sum, 0, codebook_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(vqvae->d_codebook_ema_count, 0, num_codebook_vectors * sizeof(float)));
    
    // Allocate forward pass buffers
    CHECK_CUDA(cudaMalloc(&vqvae->d_projected_input, batch_size * latent_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_quantized, batch_size * latent_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_decoder_input, batch_size * latent_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_reconstructed, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_encoding_indices, batch_size * num_codes * sizeof(int)));
    
    // Allocate backward pass buffers
    CHECK_CUDA(cudaMalloc(&vqvae->d_grad_decoder_input, batch_size * latent_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_grad_quantized, batch_size * latent_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_grad_encoder_input, batch_size * latent_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_grad_input, batch_size * input_dim * sizeof(float)));
    
    // Allocate loss buffers
    CHECK_CUDA(cudaMalloc(&vqvae->d_commitment_loss, sizeof(float)));
    
    // Create cuBLASLt descriptors
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&vqvae->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&vqvae->input_layout, CUDA_R_32F, batch_size, input_dim, input_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(vqvae->input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&vqvae->latent_layout, CUDA_R_32F, batch_size, latent_dim, latent_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(vqvae->latent_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&vqvae->input_proj_weight_layout, CUDA_R_32F, input_dim, latent_dim, latent_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(vqvae->input_proj_weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&vqvae->output_proj_weight_layout, CUDA_R_32F, latent_dim, input_dim, input_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(vqvae->output_proj_weight_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    return vqvae;
}

// Free VQ-VAE memory
void free_vqvae(VQVAE* vqvae) {
    free_transformer(vqvae->encoder);
    free_transformer(vqvae->decoder);
    
    cudaFree(vqvae->d_input_proj_W);
    cudaFree(vqvae->d_input_proj_grad);
    cudaFree(vqvae->d_input_proj_m);
    cudaFree(vqvae->d_input_proj_v);
    cudaFree(vqvae->d_output_proj_W);
    cudaFree(vqvae->d_output_proj_grad);
    cudaFree(vqvae->d_output_proj_m);
    cudaFree(vqvae->d_output_proj_v);
    cudaFree(vqvae->d_codebook);
    cudaFree(vqvae->d_codebook_ema_sum);
    cudaFree(vqvae->d_codebook_ema_count);
    cudaFree(vqvae->d_projected_input);
    cudaFree(vqvae->d_quantized);
    cudaFree(vqvae->d_decoder_input);
    cudaFree(vqvae->d_reconstructed);
    cudaFree(vqvae->d_encoding_indices);
    cudaFree(vqvae->d_grad_decoder_input);
    cudaFree(vqvae->d_grad_quantized);
    cudaFree(vqvae->d_grad_encoder_input);
    cudaFree(vqvae->d_grad_input);
    cudaFree(vqvae->d_commitment_loss);
    
    cublasLtMatmulDescDestroy(vqvae->matmul_desc);
    cublasLtMatrixLayoutDestroy(vqvae->input_layout);
    cublasLtMatrixLayoutDestroy(vqvae->latent_layout);
    cublasLtMatrixLayoutDestroy(vqvae->input_proj_weight_layout);
    cublasLtMatrixLayoutDestroy(vqvae->output_proj_weight_layout);
    
    free(vqvae);
}

// Encode to indices
void encode_vqvae(VQVAE* vqvae, float* d_input, int* d_indices) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Project input: [B x 3072] -> [B x 4096]
    LT_MATMUL(vqvae, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_input, vqvae->input_layout,
              vqvae->d_input_proj_W, vqvae->input_proj_weight_layout,
              &beta, vqvae->d_projected_input, vqvae->latent_layout);
    
    // Add positional encoding
    dim3 grid(vqvae->batch_size, vqvae->seq_len);
    add_positional_encoding_kernel<<<grid, 256>>>(
        vqvae->d_projected_input,
        vqvae->batch_size, vqvae->seq_len, vqvae->d_model
    );
    
    // Transformer encoder
    forward_pass_transformer(vqvae->encoder, vqvae->d_projected_input);
    
    // Quantize
    dim3 quant_grid(vqvae->batch_size, vqvae->num_codes);
    int smem_size = vqvae->code_dim * sizeof(float) + 256 * (sizeof(float) + sizeof(int));
    quantize_kernel<<<quant_grid, 256, smem_size>>>(
        vqvae->d_quantized, d_indices,
        vqvae->encoder->mlp_layers[vqvae->num_layers - 1]->d_output, vqvae->d_codebook,
        vqvae->batch_size, vqvae->num_codes,
        vqvae->code_dim, vqvae->num_codebook_vectors
    );
}

// Decode from indices
void decode_vqvae(VQVAE* vqvae, int* d_indices, float* d_output) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Lookup codebook vectors
    dim3 grid(vqvae->batch_size, vqvae->num_codes);
    codebook_lookup_kernel<<<grid, vqvae->code_dim>>>(
        vqvae->d_decoder_input, d_indices, vqvae->d_codebook,
        vqvae->batch_size, vqvae->num_codes, vqvae->code_dim
    );
    
    // Transformer decoder
    forward_pass_transformer(vqvae->decoder, vqvae->d_decoder_input);
    
    // Project output directly from decoder output: [B x 4096] -> [B x 3072]
    LT_MATMUL(vqvae, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              vqvae->decoder->mlp_layers[vqvae->num_layers - 1]->d_output, vqvae->latent_layout,
              vqvae->d_output_proj_W, vqvae->output_proj_weight_layout,
              &beta, d_output, vqvae->input_layout);
}

// Forward pass
void forward_pass_vqvae(VQVAE* vqvae, float* d_input) {
    encode_vqvae(vqvae, d_input, vqvae->d_encoding_indices);
    decode_vqvae(vqvae, vqvae->d_encoding_indices, vqvae->d_reconstructed);
}

// Calculate losses
void calculate_losses_vqvae(VQVAE* vqvae, float* d_input, float* losses) {
    // Reconstruction loss
    CHECK_CUDA(cudaMemset(vqvae->d_commitment_loss, 0, sizeof(float)));
    
    int total_pixels = vqvae->batch_size * vqvae->input_dim;
    int block_size = 256;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    
    compute_reconstruction_loss_and_gradient_kernel<<<num_blocks, block_size>>>(
        vqvae->d_grad_input, vqvae->d_reconstructed, d_input,
        vqvae->d_commitment_loss, total_pixels
    );
    
    float recon_loss;
    CHECK_CUDA(cudaMemcpy(&recon_loss, vqvae->d_commitment_loss, sizeof(float), cudaMemcpyDeviceToHost));
    recon_loss /= total_pixels;
    
    // Commitment loss
    CHECK_CUDA(cudaMemset(vqvae->d_commitment_loss, 0, sizeof(float)));
    
    int total_latent = vqvae->batch_size * vqvae->latent_dim;
    num_blocks = (total_latent + block_size - 1) / block_size;
    
    compute_commitment_loss_kernel<<<num_blocks, block_size>>>(
        vqvae->encoder->mlp_layers[vqvae->num_layers - 1]->d_output, vqvae->d_quantized,
        vqvae->d_commitment_loss, total_latent
    );
    
    float commitment_loss;
    CHECK_CUDA(cudaMemcpy(&commitment_loss, vqvae->d_commitment_loss, sizeof(float), cudaMemcpyDeviceToHost));
    commitment_loss /= total_latent;
    
    losses[0] = recon_loss;
    losses[1] = commitment_loss;
    losses[2] = recon_loss + vqvae->beta * commitment_loss;
}

// Zero gradients
void zero_gradients_vqvae(VQVAE* vqvae) {
    zero_gradients_transformer(vqvae->encoder);
    zero_gradients_transformer(vqvae->decoder);
    
    int input_proj_size = vqvae->input_dim * vqvae->latent_dim;
    int output_proj_size = vqvae->latent_dim * vqvae->input_dim;
    
    CHECK_CUDA(cudaMemset(vqvae->d_input_proj_grad, 0, input_proj_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(vqvae->d_output_proj_grad, 0, output_proj_size * sizeof(float)));
}

// Backward pass
void backward_pass_vqvae(VQVAE* vqvae, float* d_input) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Gradient of output projection
    LT_MATMUL(vqvae, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              vqvae->decoder->mlp_layers[vqvae->num_layers - 1]->d_output, vqvae->latent_layout,
              vqvae->d_grad_input, vqvae->input_layout,
              &beta, vqvae->d_output_proj_grad, vqvae->output_proj_weight_layout);
    
    // Gradient w.r.t. decoder output
    LT_MATMUL(vqvae, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              vqvae->d_grad_input, vqvae->input_layout,
              vqvae->d_output_proj_W, vqvae->output_proj_weight_layout,
              &beta, vqvae->decoder->mlp_layers[vqvae->num_layers - 1]->d_grad_output, vqvae->latent_layout);
    
    // Backward through decoder transformer
    backward_pass_transformer(vqvae->decoder, vqvae->d_decoder_input, vqvae->d_grad_decoder_input);
    
    // Straight-through estimator
    int total_latent = vqvae->batch_size * vqvae->latent_dim;
    int block_size = 256;
    int num_blocks = (total_latent + block_size - 1) / block_size;
    
    straight_through_gradient_kernel<<<num_blocks, block_size>>>(
        vqvae->encoder->mlp_layers[vqvae->num_layers - 1]->d_grad_output, vqvae->d_grad_decoder_input,
        vqvae->encoder->mlp_layers[vqvae->num_layers - 1]->d_output, vqvae->d_quantized,
        vqvae->beta, total_latent
    );
    
    // Backward through encoder transformer
    backward_pass_transformer(vqvae->encoder, vqvae->d_projected_input, vqvae->d_grad_encoder_input);
    
    // Gradient of input projection
    LT_MATMUL(vqvae, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_input, vqvae->input_layout,
              vqvae->d_grad_encoder_input, vqvae->latent_layout,
              &beta, vqvae->d_input_proj_grad, vqvae->input_proj_weight_layout);
}

// Update weights
void update_weights_vqvae(VQVAE* vqvae, float learning_rate) {
    vqvae->t++;
    
    // Update encoder and decoder transformers
    update_weights_transformer(vqvae->encoder, learning_rate);
    update_weights_transformer(vqvae->decoder, learning_rate);
    
    // Update projection weights
    float beta1_t = powf(vqvae->beta1, vqvae->t);
    float beta2_t = powf(vqvae->beta2, vqvae->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    int input_proj_size = vqvae->input_dim * vqvae->latent_dim;
    int input_proj_blocks = (input_proj_size + block_size - 1) / block_size;
    adamw_update_kernel<<<input_proj_blocks, block_size>>>(
        vqvae->d_input_proj_W, vqvae->d_input_proj_grad, 
        vqvae->d_input_proj_m, vqvae->d_input_proj_v,
        vqvae->beta1, vqvae->beta2, vqvae->adam_epsilon, learning_rate, 
        vqvae->weight_decay, alpha_t, input_proj_size, vqvae->batch_size
    );
    
    int output_proj_size = vqvae->latent_dim * vqvae->input_dim;
    int output_proj_blocks = (output_proj_size + block_size - 1) / block_size;
    adamw_update_kernel<<<output_proj_blocks, block_size>>>(
        vqvae->d_output_proj_W, vqvae->d_output_proj_grad,
        vqvae->d_output_proj_m, vqvae->d_output_proj_v,
        vqvae->beta1, vqvae->beta2, vqvae->adam_epsilon, learning_rate,
        vqvae->weight_decay, alpha_t, output_proj_size, vqvae->batch_size
    );
    
    // EMA update for codebook
    ema_update_kernel<<<vqvae->num_codebook_vectors, vqvae->code_dim>>>(
        vqvae->d_codebook, vqvae->d_codebook_ema_sum, vqvae->d_codebook_ema_count,
        vqvae->d_encoding_indices, vqvae->encoder->mlp_layers[vqvae->num_layers - 1]->d_output,
        vqvae->batch_size, vqvae->num_codes, vqvae->code_dim,
        vqvae->ema_decay, vqvae->epsilon
    );
}

// Save VQ-VAE
void save_vqvae(VQVAE* vqvae, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&vqvae->input_dim, sizeof(int), 1, file);
    fwrite(&vqvae->latent_dim, sizeof(int), 1, file);
    fwrite(&vqvae->hidden_dim, sizeof(int), 1, file);
    fwrite(&vqvae->num_layers, sizeof(int), 1, file);
    fwrite(&vqvae->num_codes, sizeof(int), 1, file);
    fwrite(&vqvae->num_codebook_vectors, sizeof(int), 1, file);
    fwrite(&vqvae->batch_size, sizeof(int), 1, file);
    fwrite(&vqvae->beta, sizeof(float), 1, file);
    fwrite(&vqvae->t, sizeof(int), 1, file);
    
    // Save projection weights
    int input_proj_size = vqvae->input_dim * vqvae->latent_dim;
    int output_proj_size = vqvae->latent_dim * vqvae->input_dim;
    
    float* h_input_proj_W = (float*)malloc(input_proj_size * sizeof(float));
    float* h_output_proj_W = (float*)malloc(output_proj_size * sizeof(float));
    float* h_input_proj_m = (float*)malloc(input_proj_size * sizeof(float));
    float* h_input_proj_v = (float*)malloc(input_proj_size * sizeof(float));
    float* h_output_proj_m = (float*)malloc(output_proj_size * sizeof(float));
    float* h_output_proj_v = (float*)malloc(output_proj_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_input_proj_W, vqvae->d_input_proj_W, input_proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_proj_W, vqvae->d_output_proj_W, output_proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_input_proj_m, vqvae->d_input_proj_m, input_proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_input_proj_v, vqvae->d_input_proj_v, input_proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_proj_m, vqvae->d_output_proj_m, output_proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_proj_v, vqvae->d_output_proj_v, output_proj_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_input_proj_W, sizeof(float), input_proj_size, file);
    fwrite(h_output_proj_W, sizeof(float), output_proj_size, file);
    fwrite(h_input_proj_m, sizeof(float), input_proj_size, file);
    fwrite(h_input_proj_v, sizeof(float), input_proj_size, file);
    fwrite(h_output_proj_m, sizeof(float), output_proj_size, file);
    fwrite(h_output_proj_v, sizeof(float), output_proj_size, file);
    
    free(h_input_proj_W);
    free(h_output_proj_W);
    free(h_input_proj_m);
    free(h_input_proj_v);
    free(h_output_proj_m);
    free(h_output_proj_v);
    
    // Save codebook
    int codebook_size = vqvae->num_codebook_vectors * vqvae->code_dim;
    float* h_codebook = (float*)malloc(codebook_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_codebook, vqvae->d_codebook, codebook_size * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_codebook, sizeof(float), codebook_size, file);
    free(h_codebook);
    
    // Save EMA state
    float* h_ema_sum = (float*)malloc(codebook_size * sizeof(float));
    float* h_ema_count = (float*)malloc(vqvae->num_codebook_vectors * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_ema_sum, vqvae->d_codebook_ema_sum, codebook_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ema_count, vqvae->d_codebook_ema_count, vqvae->num_codebook_vectors * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_ema_sum, sizeof(float), codebook_size, file);
    fwrite(h_ema_count, sizeof(float), vqvae->num_codebook_vectors, file);
    free(h_ema_sum);
    free(h_ema_count);
    
    fclose(file);
    
    // Save encoder and decoder
    char encoder_filename[256], decoder_filename[256];
    char base_filename[256];
    
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) *dot_pos = '\0';
    
    snprintf(encoder_filename, sizeof(encoder_filename), "%s_encoder.bin", base_filename);
    snprintf(decoder_filename, sizeof(decoder_filename), "%s_decoder.bin", base_filename);
    
    save_transformer(vqvae->encoder, encoder_filename);
    save_transformer(vqvae->decoder, decoder_filename);
    
    printf("Model saved to %s\n", filename);
}

// Load VQ-VAE
VQVAE* load_vqvae(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int input_dim, latent_dim, hidden_dim, num_layers, num_codes, num_codebook_vectors, stored_batch_size, stored_t;
    float beta;
    
    fread(&input_dim, sizeof(int), 1, file);
    fread(&latent_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&num_codes, sizeof(int), 1, file);
    fread(&num_codebook_vectors, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&beta, sizeof(float), 1, file);
    fread(&stored_t, sizeof(int), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    VQVAE* vqvae = init_vqvae(input_dim, latent_dim, hidden_dim, num_layers, num_codes, num_codebook_vectors, batch_size, beta, cublaslt_handle);
    vqvae->t = stored_t;
    
    // Load projection weights
    int input_proj_size = input_dim * latent_dim;
    int output_proj_size = latent_dim * input_dim;
    
    float* h_input_proj_W = (float*)malloc(input_proj_size * sizeof(float));
    float* h_output_proj_W = (float*)malloc(output_proj_size * sizeof(float));
    float* h_input_proj_m = (float*)malloc(input_proj_size * sizeof(float));
    float* h_input_proj_v = (float*)malloc(input_proj_size * sizeof(float));
    float* h_output_proj_m = (float*)malloc(output_proj_size * sizeof(float));
    float* h_output_proj_v = (float*)malloc(output_proj_size * sizeof(float));
    
    fread(h_input_proj_W, sizeof(float), input_proj_size, file);
    fread(h_output_proj_W, sizeof(float), output_proj_size, file);
    fread(h_input_proj_m, sizeof(float), input_proj_size, file);
    fread(h_input_proj_v, sizeof(float), input_proj_size, file);
    fread(h_output_proj_m, sizeof(float), output_proj_size, file);
    fread(h_output_proj_v, sizeof(float), output_proj_size, file);
    
    CHECK_CUDA(cudaMemcpy(vqvae->d_input_proj_W, h_input_proj_W, input_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(vqvae->d_output_proj_W, h_output_proj_W, output_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(vqvae->d_input_proj_m, h_input_proj_m, input_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(vqvae->d_input_proj_v, h_input_proj_v, input_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(vqvae->d_output_proj_m, h_output_proj_m, output_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(vqvae->d_output_proj_v, h_output_proj_v, output_proj_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_input_proj_W);
    free(h_output_proj_W);
    free(h_input_proj_m);
    free(h_input_proj_v);
    free(h_output_proj_m);
    free(h_output_proj_v);
    
    // Load codebook
    int codebook_size = num_codebook_vectors * vqvae->code_dim;
    float* h_codebook = (float*)malloc(codebook_size * sizeof(float));
    fread(h_codebook, sizeof(float), codebook_size, file);
    CHECK_CUDA(cudaMemcpy(vqvae->d_codebook, h_codebook, codebook_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_codebook);
    
    // Load EMA state
    float* h_ema_sum = (float*)malloc(codebook_size * sizeof(float));
    float* h_ema_count = (float*)malloc(num_codebook_vectors * sizeof(float));
    fread(h_ema_sum, sizeof(float), codebook_size, file);
    fread(h_ema_count, sizeof(float), num_codebook_vectors, file);
    CHECK_CUDA(cudaMemcpy(vqvae->d_codebook_ema_sum, h_ema_sum, codebook_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(vqvae->d_codebook_ema_count, h_ema_count, num_codebook_vectors * sizeof(float), cudaMemcpyHostToDevice));
    free(h_ema_sum);
    free(h_ema_count);
    
    fclose(file);
    
    // Load encoder and decoder
    char encoder_filename[256], decoder_filename[256];
    char base_filename[256];
    
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) *dot_pos = '\0';
    
    snprintf(encoder_filename, sizeof(encoder_filename), "%s_encoder.bin", base_filename);
    snprintf(decoder_filename, sizeof(decoder_filename), "%s_decoder.bin", base_filename);
    
    free_transformer(vqvae->encoder);
    free_transformer(vqvae->decoder);
    
    vqvae->encoder = load_transformer(encoder_filename, batch_size, cublaslt_handle);
    vqvae->decoder = load_transformer(decoder_filename, batch_size, cublaslt_handle);
    
    printf("Model loaded from %s\n", filename);
    return vqvae;
}