#include "data.h"

void load_cifar10_data(unsigned char** X, unsigned char** labels, int* num_samples, 
                       const char** batch_files, int num_batches) {
    const int samples_per_batch = 10000;
    const int image_size = 32 * 32 * 3;
    
    *num_samples = samples_per_batch * num_batches;
    *X = (unsigned char*)malloc(*num_samples * image_size * sizeof(unsigned char));
    *labels = (unsigned char*)malloc(*num_samples * sizeof(unsigned char));
    
    int sample_offset = 0;
    
    for (int b = 0; b < num_batches; b++) {
        FILE* file = fopen(batch_files[b], "rb");
        if (!file) {
            printf("Error: Could not open CIFAR-10 file: %s\n", batch_files[b]);
            free(*X);
            free(*labels);
            *X = NULL;
            *labels = NULL;
            *num_samples = 0;
            return;
        }
        
        for (int i = 0; i < samples_per_batch; i++) {
            unsigned char label;
            unsigned char image[3072];
            
            fread(&label, sizeof(unsigned char), 1, file);
            fread(image, sizeof(unsigned char), 3072, file);
            
            (*labels)[sample_offset + i] = label;
            
            // Interleave R,G,B channels
            for (int pixel = 0; pixel < 1024; pixel++) {
                int out_idx = (sample_offset + i) * image_size + pixel * 3;
                (*X)[out_idx + 0] = image[pixel];
                (*X)[out_idx + 1] = image[pixel + 1024];
                (*X)[out_idx + 2] = image[pixel + 2048];
            }
        }
        
        fclose(file);
        sample_offset += samples_per_batch;
        printf("Loaded CIFAR-10 batch: %s (%d samples)\n", batch_files[b], samples_per_batch);
    }
    
    printf("Total CIFAR-10 data loaded: %d samples (32x32x3)\n", *num_samples);
}

void save_cifar10_image_png(unsigned char* image_data, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, 32, 32, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    png_bytep* row_pointers = (png_bytep*)malloc(32 * sizeof(png_bytep));
    for (int y = 0; y < 32; y++) {
        row_pointers[y] = &image_data[y * 32 * 3];
    }

    png_write_rows(png, row_pointers, 32);
    png_write_end(png, NULL);

    free(row_pointers);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}