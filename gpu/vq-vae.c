#include "vq-vae.h"

// Initialize VQ-VAE
VQVAE* init_vqvae(int input_dim, int latent_dim, int hidden_dim, int num_codes, int num_codebook_vectors, int batch_size, float beta, cublasLtHandle_t cublaslt_handle) {
    VQVAE* vqvae = (VQVAE*)malloc(sizeof(VQVAE));
    
    vqvae->input_dim = input_dim;
    vqvae->latent_dim = latent_dim;
    vqvae->hidden_dim = hidden_dim;
    vqvae->num_codes = num_codes;
    vqvae->code_dim = latent_dim / num_codes;
    vqvae->num_codebook_vectors = num_codebook_vectors;
    vqvae->batch_size = batch_size;
    vqvae->beta = beta;
    vqvae->ema_decay = 0.99f;
    vqvae->epsilon = 1e-5f;
    vqvae->cublaslt_handle = cublaslt_handle;
    
    // Initialize encoder and decoder MLPs
    vqvae->encoder = init_mlp(input_dim, hidden_dim, latent_dim, batch_size, cublaslt_handle);
    vqvae->decoder = init_mlp(latent_dim, hidden_dim, input_dim, batch_size, cublaslt_handle);
    
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
    
    // Initialize EMA buffers
    CHECK_CUDA(cudaMemset(vqvae->d_codebook_ema_sum, 0, codebook_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(vqvae->d_codebook_ema_count, 0, num_codebook_vectors * sizeof(float)));
    
    // Allocate forward pass buffers
    CHECK_CUDA(cudaMalloc(&vqvae->d_quantized, batch_size * latent_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_encoding_indices, batch_size * num_codes * sizeof(int)));
    
    // Allocate backward pass buffers
    CHECK_CUDA(cudaMalloc(&vqvae->d_grad_encoded, batch_size * latent_dim * sizeof(float)));
    
    // Allocate loss buffers
    CHECK_CUDA(cudaMalloc(&vqvae->d_loss_buffer, batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_codebook_loss, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&vqvae->d_commitment_loss, sizeof(float)));
    
    return vqvae;
}

// Free VQ-VAE memory
void free_vqvae(VQVAE* vqvae) {
    free_mlp(vqvae->encoder);
    free_mlp(vqvae->decoder);
    
    cudaFree(vqvae->d_codebook);
    cudaFree(vqvae->d_codebook_ema_sum);
    cudaFree(vqvae->d_codebook_ema_count);
    cudaFree(vqvae->d_quantized);
    cudaFree(vqvae->d_encoding_indices);
    cudaFree(vqvae->d_grad_encoded);
    cudaFree(vqvae->d_loss_buffer);
    cudaFree(vqvae->d_codebook_loss);
    cudaFree(vqvae->d_commitment_loss);
    
    free(vqvae);
}

// CUDA kernel for vector quantization
__global__ void quantize_kernel(float* quantized, int* indices, 
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
    
    // Load encoded vector to shared memory (coalesced)
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
        
        // Unrolled distance computation
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
    
    // Parallel reduction (tree-based)
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


// CUDA kernel for VQ losses computation
__global__ static void compute_vq_losses_kernel(float* encoded, float* quantized,
                                                float* codebook_loss, float* commitment_loss,
                                                int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float enc = encoded[idx];
    float quant = quantized[idx];
    float diff = enc - quant;
    float sq_diff = diff * diff;
    
    // Codebook loss: ||sg[z_e] - e||² (gradient flows to codebook)
    atomicAdd(codebook_loss, sq_diff);
    
    // Commitment loss: ||z_e - sg[e]||² (gradient flows to encoder)
    atomicAdd(commitment_loss, sq_diff);
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
__global__ static void ema_update_accumulate_kernel(float* ema_sum, float* ema_count,
                                                    int* indices, float* encoded,
                                                    int batch_size, int num_codes,
                                                    int code_dim, float decay) {
    int k = blockIdx.x;
    int d = threadIdx.x;
    
    if (d >= code_dim) return;
    
    // Decay previous EMA
    if (d == 0) {
        ema_count[k] *= decay;
    }
    ema_sum[k * code_dim + d] *= decay;
    
    __syncthreads();
    
    // Accumulate new values
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < num_codes; c++) {
            if (indices[b * num_codes + c] == k) {
                if (d == 0) atomicAdd(&ema_count[k], 1.0f);
                atomicAdd(&ema_sum[k * code_dim + d], encoded[b * num_codes * code_dim + c * code_dim + d]);
            }
        }
    }
}

// CUDA kernel for EMA codebook finalization
__global__ static void ema_update_finalize_kernel(float* codebook, float* ema_sum,
                                                  float* ema_count, int num_codebook_vectors,
                                                  int code_dim, float epsilon) {
    int k = blockIdx.x;
    int d = threadIdx.x;
    
    if (k >= num_codebook_vectors || d >= code_dim) return;
    
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

// Encode to indices
void encode_vqvae(VQVAE* vqvae, float* d_input, int* d_indices) {
    forward_pass_mlp(vqvae->encoder, d_input);
    
    dim3 grid(vqvae->batch_size, vqvae->num_codes);
    int smem_size = vqvae->code_dim * sizeof(float) + 256 * (sizeof(float) + sizeof(int));
    quantize_kernel<<<grid, 256, smem_size>>>(
        vqvae->d_quantized, d_indices,
        vqvae->encoder->d_output, vqvae->d_codebook,
        vqvae->batch_size, vqvae->num_codes,
        vqvae->code_dim, vqvae->num_codebook_vectors
    );
}

// Decode from indices
void decode_vqvae(VQVAE* vqvae, int* d_indices) {
    // Lookup codebook vectors
    dim3 grid(vqvae->batch_size, vqvae->num_codes);
    codebook_lookup_kernel<<<grid, vqvae->code_dim>>>(
        vqvae->d_quantized, d_indices, vqvae->d_codebook,
        vqvae->batch_size, vqvae->num_codes, vqvae->code_dim
    );
    
    // Decode
    forward_pass_mlp(vqvae->decoder, vqvae->d_quantized);
}

// Forward pass
void forward_pass_vqvae(VQVAE* vqvae, float* d_input) {
    encode_vqvae(vqvae, d_input, vqvae->d_encoding_indices);
    decode_vqvae(vqvae, vqvae->d_encoding_indices);
}

// Calculate losses
void calculate_losses_vqvae(VQVAE* vqvae, float* d_input, float* losses) {
    // Reconstruction loss
    float recon_loss = calculate_loss_mlp(vqvae->decoder, d_input);
    
    // VQ losses
    CHECK_CUDA(cudaMemset(vqvae->d_codebook_loss, 0, sizeof(float)));
    CHECK_CUDA(cudaMemset(vqvae->d_commitment_loss, 0, sizeof(float)));
    
    int total_elements = vqvae->batch_size * vqvae->latent_dim;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    compute_vq_losses_kernel<<<num_blocks, block_size>>>(
        vqvae->encoder->d_output, vqvae->d_quantized,
        vqvae->d_codebook_loss, vqvae->d_commitment_loss, total_elements
    );
    
    float codebook_loss, commitment_loss;
    CHECK_CUDA(cudaMemcpy(&codebook_loss, vqvae->d_codebook_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&commitment_loss, vqvae->d_commitment_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    codebook_loss /= total_elements;
    commitment_loss /= total_elements;
    
    losses[0] = recon_loss;
    losses[1] = codebook_loss;
    losses[2] = commitment_loss;
    losses[3] = recon_loss + codebook_loss + vqvae->beta * commitment_loss;
}

// Zero gradients
void zero_gradients_vqvae(VQVAE* vqvae) {
    zero_gradients_mlp(vqvae->encoder);
    zero_gradients_mlp(vqvae->decoder);
}

// Backward pass
void backward_pass_vqvae(VQVAE* vqvae, float* d_input) {
    // Backward through decoder
    backward_pass_mlp(vqvae->decoder, vqvae->d_quantized, vqvae->d_grad_encoded);
    
    // Straight-through estimator
    int total_elements = vqvae->batch_size * vqvae->latent_dim;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    straight_through_gradient_kernel<<<num_blocks, block_size>>>(
        vqvae->encoder->d_grad_output, vqvae->d_grad_encoded,
        vqvae->encoder->d_output, vqvae->d_quantized,
        vqvae->beta, total_elements
    );
    
    // Backward through encoder
    backward_pass_mlp(vqvae->encoder, d_input, NULL);
}

// Update weights
void update_weights_vqvae(VQVAE* vqvae, float learning_rate) {
    // Update encoder and decoder
    update_weights_mlp(vqvae->encoder, learning_rate);
    update_weights_mlp(vqvae->decoder, learning_rate);
    
    // EMA update for codebook
    dim3 grid(vqvae->num_codebook_vectors);
    
    // Accumulate with decay
    ema_update_accumulate_kernel<<<grid, vqvae->code_dim>>>(
        vqvae->d_codebook_ema_sum, vqvae->d_codebook_ema_count,
        vqvae->d_encoding_indices, vqvae->encoder->d_output,
        vqvae->batch_size, vqvae->num_codes, vqvae->code_dim,
        vqvae->ema_decay
    );
    
    // Update codebook
    ema_update_finalize_kernel<<<grid, vqvae->code_dim>>>(
        vqvae->d_codebook, vqvae->d_codebook_ema_sum,
        vqvae->d_codebook_ema_count, vqvae->num_codebook_vectors,
        vqvae->code_dim, vqvae->epsilon
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
    fwrite(&vqvae->num_codes, sizeof(int), 1, file);
    fwrite(&vqvae->num_codebook_vectors, sizeof(int), 1, file);
    fwrite(&vqvae->batch_size, sizeof(int), 1, file);
    fwrite(&vqvae->beta, sizeof(float), 1, file);
    
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
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    snprintf(encoder_filename, sizeof(encoder_filename), "%s_encoder.bin", base_filename);
    snprintf(decoder_filename, sizeof(decoder_filename), "%s_decoder.bin", base_filename);
    
    save_mlp(vqvae->encoder, encoder_filename);
    save_mlp(vqvae->decoder, decoder_filename);
    
    printf("Model saved to %s\n", filename);
}

// Load VQ-VAE
VQVAE* load_vqvae(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int input_dim, latent_dim, hidden_dim, num_codes, num_codebook_vectors, stored_batch_size;
    float beta;
    
    fread(&input_dim, sizeof(int), 1, file);
    fread(&latent_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_codes, sizeof(int), 1, file);
    fread(&num_codebook_vectors, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&beta, sizeof(float), 1, file);
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    VQVAE* vqvae = init_vqvae(input_dim, latent_dim, hidden_dim, num_codes,
                              num_codebook_vectors, batch_size, beta, cublaslt_handle);
    
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
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    snprintf(encoder_filename, sizeof(encoder_filename), "%s_encoder.bin", base_filename);
    snprintf(decoder_filename, sizeof(decoder_filename), "%s_decoder.bin", base_filename);
    
    free_mlp(vqvae->encoder);
    free_mlp(vqvae->decoder);
    
    vqvae->encoder = load_mlp(encoder_filename, batch_size, cublaslt_handle);
    vqvae->decoder = load_mlp(decoder_filename, batch_size, cublaslt_handle);
    
    printf("Model loaded from %s\n", filename);
    return vqvae;
}