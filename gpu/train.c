#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include "../data.h"
#include "vq-vae.h"

VQVAE* vqvae = NULL;

void handle_sigint(int signum) {
    if (vqvae) {
        char model_filename[64];
        time_t now = time(NULL);
        strftime(model_filename, sizeof(model_filename), "%Y%m%d_%H%M%S_vqvae.bin", localtime(&now));
        save_vqvae(vqvae, model_filename);
    }
    exit(128 + signum);
}

int main(int argc, char* argv[]) {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);
    
    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));
    
    // Parameters
    const int img_height = 32;
    const int img_width = 32;
    const int img_channels = 3;
    const int patch_height = 4;
    const int patch_width = 4;
    const int d_model = 512;
    const int hidden_dim = 2048;
    const int num_layers = 12;
    const int num_codebook_vectors = 8192;
    const int batch_size = 256;
    const float beta = 0.25f;
    
    int num_patches = (img_height / patch_height) * (img_width / patch_width);
    int patch_dim = patch_height * patch_width * img_channels;
    
    // Load CIFAR-10 data
    unsigned char* cifar_images = NULL;
    unsigned char* cifar_labels = NULL;
    int num_images = 0;
    
    const char* batch_files[] = {
        "../cifar-10-batches-bin/data_batch_1.bin",
        "../cifar-10-batches-bin/data_batch_2.bin",
        "../cifar-10-batches-bin/data_batch_3.bin",
        "../cifar-10-batches-bin/data_batch_4.bin",
        "../cifar-10-batches-bin/data_batch_5.bin"
    };
    
    load_cifar10_data(&cifar_images, &cifar_labels, &num_images, batch_files, 5);
    
    // Normalize images to [-1, 1]
    int input_dim = img_height * img_width * img_channels;
    float* normalized_images = (float*)malloc(num_images * input_dim * sizeof(float));
    for (int i = 0; i < num_images * input_dim; i++) {
        normalized_images[i] = (cifar_images[i] / 127.5f) - 1.0f;
    }
    
    // Initialize or load VQ-VAE
    if (argc > 1) {
        printf("Loading checkpoint: %s\n", argv[1]);
        vqvae = load_vqvae(argv[1], batch_size, cublaslt_handle);
    } else {
        printf("Initializing new VQ-VAE with patch-based Transformer encoder/decoder...\n");
        vqvae = init_vqvae(img_height, img_width, img_channels, patch_height, patch_width,
                          d_model, hidden_dim, num_layers, num_codebook_vectors, 
                          batch_size, beta, cublaslt_handle);
    }
    
    printf("Architecture:\n");
    printf("  Input: [%d x %d x %d x %d] (CIFAR-10 images)\n", batch_size, img_height, img_width, img_channels);
    printf("  Patch size: %d x %d\n", patch_height, patch_width);
    printf("  Number of patches: %d (%d x %d grid)\n", num_patches, img_height/patch_height, img_width/patch_width);
    printf("  Patch dimension: %d (%d x %d x %d)\n", patch_dim, patch_height, patch_width, img_channels);
    printf("  Patch projection: [%d x %d] -> [%d x %d]\n", batch_size * num_patches, patch_dim, batch_size * num_patches, d_model);
    printf("  Positional encoding: sinusoidal, added to patch embeddings\n");
    printf("  Encoder: Transformer with %d layers, d_model=%d, hidden=%d\n", num_layers, d_model, hidden_dim);
    printf("  Quantization: %d patches, %d codebook vectors, d_model=%d\n", num_patches, num_codebook_vectors, d_model);
    printf("  Decoder: Transformer with %d layers, d_model=%d, hidden=%d\n", num_layers, d_model, hidden_dim);
    printf("  Output projection: [%d x %d] -> [%d x %d]\n", batch_size * num_patches, d_model, batch_size * num_patches, patch_dim);
    
    // Count parameters
    int patch_proj_params = patch_dim * d_model + d_model * patch_dim;
    int encoder_attention_params = num_layers * 4 * d_model * d_model;
    int encoder_mlp_params = num_layers * (d_model * hidden_dim + hidden_dim * d_model);
    int decoder_attention_params = num_layers * 4 * d_model * d_model;
    int decoder_mlp_params = num_layers * (d_model * hidden_dim + hidden_dim * d_model);
    int codebook_params = num_codebook_vectors * d_model;
    int total_params = patch_proj_params + encoder_attention_params + encoder_mlp_params + 
                       decoder_attention_params + decoder_mlp_params + codebook_params;
    
    printf("Parameters:\n");
    printf("  Patch projections: %.2fM\n", patch_proj_params / 1e6f);
    printf("  Encoder: %.2fM (attention: %.2fM, mlp: %.2fM)\n", 
           (encoder_attention_params + encoder_mlp_params) / 1e6f,
           encoder_attention_params / 1e6f, encoder_mlp_params / 1e6f);
    printf("  Decoder: %.2fM (attention: %.2fM, mlp: %.2fM)\n",
           (decoder_attention_params + decoder_mlp_params) / 1e6f,
           decoder_attention_params / 1e6f, decoder_mlp_params / 1e6f);
    printf("  Codebook: %.2fM\n", codebook_params / 1e6f);
    printf("  Total: %.2fM\n", total_params / 1e6f);
    
    // Training parameters
    const int num_epochs = 600;
    const float learning_rate = 0.0001f;
    const int num_batches = num_images / batch_size;
    
    // Allocate device memory for batch data
    float *d_batch_images;
    CHECK_CUDA(cudaMalloc(&d_batch_images, batch_size * input_dim * sizeof(float)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_recon_loss = 0.0f;
        float epoch_commitment_loss = 0.0f;
        float epoch_total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_offset = batch * batch_size * input_dim;
            
            // Copy batch to device
            CHECK_CUDA(cudaMemcpy(d_batch_images, &normalized_images[batch_offset], batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_vqvae(vqvae, d_batch_images);
            
            // Calculate losses
            float losses[3];
            calculate_losses_vqvae(vqvae, d_batch_images, losses);
            
            epoch_recon_loss += losses[0];
            epoch_commitment_loss += losses[1];
            epoch_total_loss += losses[2];
            
            // Backward pass and update
            zero_gradients_vqvae(vqvae);
            backward_pass_vqvae(vqvae, d_batch_images);
            update_weights_vqvae(vqvae, learning_rate);
            
            // Print progress
            if (batch % 2 == 0) {
                printf("Epoch [%d/%d], Batch [%d/%d], Recon: %.4f, Commit: %.4f, Total: %.4f\n",
                       epoch, num_epochs, batch, num_batches,
                       losses[0], losses[1], losses[2]);
            }
            
            // Generate reconstructions periodically
            if (batch > 0 && batch % 100 == 0) {
                printf("--- Generating reconstructions (epoch %d, batch %d) ---\n", epoch, batch);
                
                // Get reconstructions
                float* h_reconstructions = (float*)malloc(batch_size * input_dim * sizeof(float));
                CHECK_CUDA(cudaMemcpy(h_reconstructions, vqvae->d_reconstructed, batch_size * input_dim * sizeof(float), cudaMemcpyDeviceToHost));
                
                // Save a few examples
                for (int i = 0; i < 3; i++) {
                    unsigned char* recon_img = (unsigned char*)malloc(input_dim);
                    unsigned char* orig_img = (unsigned char*)malloc(input_dim);
                    
                    for (int j = 0; j < input_dim; j++) {
                        // Denormalize reconstruction
                        float val = h_reconstructions[i * input_dim + j];
                        val = (val + 1.0f) * 127.5f;
                        recon_img[j] = (unsigned char)fmaxf(0.0f, fminf(255.0f, val));
                        
                        // Get original
                        orig_img[j] = cifar_images[batch_offset + i * input_dim + j];
                    }
                    
                    char orig_filename[256], recon_filename[256];
                    snprintf(orig_filename, sizeof(orig_filename), "original_epoch_%d_batch_%d_sample_%d.png", epoch, batch, i);
                    snprintf(recon_filename, sizeof(recon_filename), "reconstructed_epoch_%d_batch_%d_sample_%d.png", epoch, batch, i);
                    
                    save_cifar10_image_png(orig_img, orig_filename);
                    save_cifar10_image_png(recon_img, recon_filename);
                    
                    free(recon_img);
                    free(orig_img);
                }
                
                free(h_reconstructions);
            }
        }
        
        epoch_recon_loss /= num_batches;
        epoch_commitment_loss /= num_batches;
        epoch_total_loss /= num_batches;
        
        printf("========================================\n");
        printf("Epoch [%d/%d] Summary:\n", epoch, num_epochs);
        printf("  Reconstruction Loss:  %.4f\n", epoch_recon_loss);
        printf("  Commitment Loss:      %.4f\n", epoch_commitment_loss);
        printf("  Total Loss:           %.4f\n", epoch_total_loss);
        printf("========================================\n\n");
        
        // Save checkpoint
        if ((epoch + 1) % 10 == 0) {
            char checkpoint_fname[64];
            snprintf(checkpoint_fname, sizeof(checkpoint_fname), "checkpoint_vqvae.bin");
            save_vqvae(vqvae, checkpoint_fname);
        }
    }
    
    // Save final model
    char model_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_vqvae.bin", localtime(&now));
    save_vqvae(vqvae, model_fname);
    
    // Cleanup
    free(cifar_images);
    free(cifar_labels);
    free(normalized_images);
    CHECK_CUDA(cudaFree(d_batch_images));
    free_vqvae(vqvae);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}