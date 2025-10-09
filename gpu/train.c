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
    const int input_dim = 32 * 32 * 3;
    const int latent_dim = 1024;         // 64 codes × 8 dims
    const int hidden_dim = 4096;
    const int num_codes = 128;           // 8×8 spatial grid
    const int num_codebook_vectors = 1024;
    const int batch_size = 128;
    const float beta = 0.5f;
    
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
    float* normalized_images = (float*)malloc(num_images * input_dim * sizeof(float));
    for (int i = 0; i < num_images * input_dim; i++) {
        normalized_images[i] = (cifar_images[i] / 127.5f) - 1.0f;
    }
    
    // Initialize or load VQ-VAE
    if (argc > 1) {
        printf("Loading checkpoint: %s\n", argv[1]);
        vqvae = load_vqvae(argv[1], batch_size, cublaslt_handle);
    } else {
        printf("Initializing new VQ-VAE...\n");
        vqvae = init_vqvae(input_dim, latent_dim, hidden_dim, num_codes,
                          num_codebook_vectors, batch_size, beta, cublaslt_handle);
    }
    
    int encoder_params = input_dim * hidden_dim + hidden_dim * latent_dim;
    int decoder_params = latent_dim * hidden_dim + hidden_dim * input_dim;
    int codebook_params = num_codebook_vectors * (latent_dim / num_codes);
    printf("Total parameters: ~%.1fM\n", (encoder_params + decoder_params + codebook_params) / 1e6f);
    
    // Training parameters
    const int num_epochs = 300;
    const float learning_rate = 0.0003f;
    const int num_batches = num_images / batch_size;
    
    // Allocate device memory for batch data
    float *d_batch_images;
    CHECK_CUDA(cudaMalloc(&d_batch_images, batch_size * input_dim * sizeof(float)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_recon_loss = 0.0f;
        float epoch_codebook_loss = 0.0f;
        float epoch_commitment_loss = 0.0f;
        float epoch_total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_offset = batch * batch_size * input_dim;
            
            // Copy batch to device
            CHECK_CUDA(cudaMemcpy(d_batch_images, &normalized_images[batch_offset],
                                  batch_size * input_dim * sizeof(float),
                                  cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_vqvae(vqvae, d_batch_images);
            
            // Calculate losses
            float losses[4];
            calculate_losses_vqvae(vqvae, d_batch_images, losses);
            
            epoch_recon_loss += losses[0];
            epoch_codebook_loss += losses[1];
            epoch_commitment_loss += losses[2];
            epoch_total_loss += losses[3];
            
            // Backward pass and update
            backward_pass_vqvae(vqvae, d_batch_images);
            update_weights_vqvae(vqvae, learning_rate);
            
            // Print progress
            if (batch % 50 == 0) {
                printf("Epoch [%d/%d], Batch [%d/%d], Recon: %.4f, Codebook: %.4f, Commit: %.4f, Total: %.4f\n",
                       epoch, num_epochs, batch, num_batches,
                       losses[0], losses[1], losses[2], losses[3]);
            }
            
            // Generate reconstructions periodically
            if (batch > 0 && batch % 200 == 0) {
                printf("\n--- Generating reconstructions (epoch %d, batch %d) ---\n", epoch, batch);
                
                // Get reconstructions from decoder output
                float* h_reconstructions = (float*)malloc(batch_size * input_dim * sizeof(float));
                CHECK_CUDA(cudaMemcpy(h_reconstructions, vqvae->decoder->d_output,
                                      batch_size * input_dim * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                
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
                    snprintf(orig_filename, sizeof(orig_filename),
                            "original_epoch_%d_batch_%d_sample_%d.png", epoch, batch, i);
                    snprintf(recon_filename, sizeof(recon_filename),
                            "reconstructed_epoch_%d_batch_%d_sample_%d.png", epoch, batch, i);
                    
                    save_cifar10_image_png(orig_img, orig_filename);
                    save_cifar10_image_png(recon_img, recon_filename);
                    
                    free(recon_img);
                    free(orig_img);
                }
                
                free(h_reconstructions);
                printf("--- End reconstruction ---\n\n");
            }
        }
        
        epoch_recon_loss /= num_batches;
        epoch_codebook_loss /= num_batches;
        epoch_commitment_loss /= num_batches;
        epoch_total_loss /= num_batches;
        
        printf("========================================\n");
        printf("Epoch [%d/%d] Summary:\n", epoch, num_epochs);
        printf("  Reconstruction Loss:  %.4f\n", epoch_recon_loss);
        printf("  Codebook Loss:        %.4f\n", epoch_codebook_loss);
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