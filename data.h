#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <png.h>

void load_cifar10_data(unsigned char** X, unsigned char** labels, int* num_samples, 
                       const char** batch_files, int num_batches);
void save_cifar10_image_png(unsigned char* image_data, const char* filename);

#endif