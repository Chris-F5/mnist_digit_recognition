#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "data.h"
#include "rands.h"

#define ASSERT(cond) if (!(cond)) { fprintf(stderr, "Assert Failed!\n"); exit(1); }

struct network {
    int layer_count;
    int *layer_size;
    float **weight; /* layer list of matrix */
    float **bias; /* layer list of vector */
};

struct network_scratch {
    float **activation;
    float **weighted_input;
    float **error;
    float **batch_del_weight;
    float **batch_del_bias;
};

static unsigned char training_labels[60000];
static unsigned char training_images[60000][784];

static int rand_index = 0;
static float next_rand() {
    if (rand_index >= sizeof(rands) / sizeof(rands[0]))
        rand_index = 0;
    return rands[rand_index++];
}

static void
init_network(struct network *net, struct network_scratch *scratch)
{
    int l, k, j, r;

    /* allocate network */
    net->layer_count = 3;
    net->layer_size = malloc(net->layer_count * sizeof(net->layer_size));
    net->layer_size[0] = 784;
    net->layer_size[1] = 30;
    net->layer_size[2] = 10;
    net->weight = malloc(net->layer_count * sizeof(net->weight[0]));
    net->bias = malloc(net->layer_count * sizeof(net->bias[0]));
    net->weight[0] = net->bias[0] = NULL;
    for (l = 1; l < net->layer_count; l++) {
        net->weight[l] = malloc(net->layer_size[l] * net->layer_size[l-1] * sizeof(net->weight[0][0]));
        net->bias[l] = malloc(net->layer_size[l] * sizeof(net->bias[0][0]));
    }

    /* allocate network scratch */
    scratch->activation = malloc(net->layer_count * sizeof(scratch->activation[0]));
    scratch->weighted_input = malloc(net->layer_count * sizeof(scratch->weighted_input[0]));
    scratch->error = malloc(net->layer_count * sizeof(scratch->error[0]));
    scratch->batch_del_weight = malloc(net->layer_count * sizeof(scratch->batch_del_weight[0]));
    scratch->batch_del_bias = malloc(net->layer_count * sizeof(scratch->batch_del_bias[0]));
    scratch->weighted_input[0] = scratch->error[0] = NULL;
    scratch->batch_del_weight[0] = scratch->batch_del_bias[0] =  NULL;
    for (l = 0; l < net->layer_count; l++) {
        scratch->activation[l] = malloc(net->layer_size[l] * sizeof(scratch->activation[0][0]));
    }
    for (l = 1; l < net->layer_count; l++) {
        scratch->weighted_input[l] = malloc(net->layer_size[l] * sizeof(scratch->weighted_input[0][0]));
        scratch->error[l] = malloc(net->layer_size[l] * sizeof(scratch->error[0][0]));
        scratch->batch_del_weight[l] = malloc(net->layer_size[l] * net->layer_size[l-1] * sizeof(scratch->batch_del_weight[0][0]));
        scratch->batch_del_bias[l] = malloc(net->layer_size[l] * sizeof(scratch->batch_del_bias[0][0]));
    }

    /* init gaussian weights and biases */
    for (l = 1; l < net->layer_count; l++) {
        for (j = 0; j < net->layer_size[l]; j++) {
            net->bias[l][j] = next_rand();
            for (k = 0; k < net->layer_size[l-1]; k++)
                net->weight[l][j*net->layer_size[l-1] + k] = next_rand();
        }
    }
}

static void
free_network(struct network *net, struct network_scratch *scratch)
{
    int l;
    for (l = 0; l < net->layer_count; l++) {
        free(net->weight[l]);
        free(net->bias[l]);
        free(scratch->activation[l]);
        free(scratch->weighted_input[l]);
        free(scratch->error[l]);
        free(scratch->batch_del_weight[l]);
        free(scratch->batch_del_bias[l]);
    }
    free(net->layer_size);
    free(net->weight);
    free(net->bias);
    free(scratch->activation);
    free(scratch->weighted_input);
    free(scratch->error);
    free(scratch->batch_del_weight);
    free(scratch->batch_del_bias);
}

static float
sigmoid(float x) {
    return 1.0/(1.0+exp(-x));
}

static float
sigmoid_derivative(float x) {
    return sigmoid(x) * sigmoid(-x);
}

static void
backpropagate(const struct network *net, struct network_scratch *scratch, int label, const unsigned char image[784])
{
    int l, j, k;
    float t;

    /* input activation */
    ASSERT(net->layer_size[0] == 784)
    ASSERT(net->layer_size[net->layer_count-1] == 10)
    for (j = 0; j < 784; j++)
        scratch->activation[0][j] = (float)image[j] / 255.0f;

    /* feedforward (weignted_input, activation) */
    for (l = 1; l < net->layer_count; l++) {
        for (j = 0; j < net->layer_size[l]; j++) {
            scratch->weighted_input[l][j] = 0.0;
            for (k = 0; k < net->layer_size[l-1]; k++)
                scratch->weighted_input[l][j] += net->weight[l][j*net->layer_size[l-1] + k] * scratch->activation[l-1][k];
            scratch->weighted_input[l][j] += net->bias[l][j];
            scratch->activation[l][j] = sigmoid(scratch->weighted_input[l][j]);
        }
    }

    /* output layer error */
    l = net->layer_count - 1;
    for (j = 0; j < net->layer_size[l]; j++) {
        /* this depends on quadratic cost derivative */
        scratch->error[l][j] = (scratch->activation[l][j] - (j == label ? 1.0 : 0.0))
            * sigmoid_derivative(scratch->weighted_input[l][j]);
    }

    /* backward pass (error) */
    for (l = net->layer_count - 2; l > 0; l--) {
        for (k = 0; k < net->layer_size[l]; k++) {
            t = 0;
            for (j = 0; j < net->layer_size[l+1]; j++)
                t += net->weight[l+1][j*net->layer_size[l] + k] * scratch->error[l+1][j];
            scratch->error[l][k] = t * sigmoid_derivative(scratch->weighted_input[l][k]);
        }
    }

    /* batch_del_{weight,bias} from error */
    for (l = 1; l < net->layer_count; l++) {
        for (j = 0; j < net->layer_size[l]; j++) {
            scratch->batch_del_bias[l][j] += scratch->error[l][j];
            for (k = 0; k < net->layer_size[l-1]; k++)
                scratch->batch_del_weight[l][j*net->layer_size[l-1] + k]
                    += scratch->activation[l-1][k] * scratch->error[l][j];
        }
    }
}

int
predict_digit(struct network *net, struct network_scratch *scratch, const unsigned char image[784])
{
    int i, best;
    float max;
    /* TODO: skip backwards pass when predicting */
    backpropagate(net, scratch, 0, image);
    ASSERT(net->layer_size[net->layer_count-1] == 10)
    best = -1;
    max = -99999.0;
    for (i = 0; i < 10; i++) {
        if (scratch->activation[net->layer_count-1][i] > max) {
            max = scratch->activation[net->layer_count-1][i];
            best = i;
        }
    }
    return best;
}

float compute_accuracy(struct network *net, struct network_scratch *scratch, int total, unsigned char *labels, unsigned char (*images)[784])
{
    int correct, i, predict;
    correct = 0;
    for (i = 0; i < total; i++) {
        predict = predict_digit(net, scratch, images[i]);
        if (predict == labels[i])
            correct++;
    }
    return (float)correct/(float)total;
}

void
batch(struct network *net, struct network_scratch *scratch, int b, int batch_size, float eta)
{
    int i, l, j, k;
    for (l = 1; l < net->layer_count; l++) {
        memset(scratch->batch_del_weight[l], 0, net->layer_size[l] * net->layer_size[l-1] * sizeof(scratch->batch_del_weight[0][0]));
        memset(scratch->batch_del_bias[l], 0, net->layer_size[l] * sizeof(scratch->batch_del_bias[0][0]));
    }
    for (i = b; i < b + batch_size; i++) {
        backpropagate(net, scratch, training_labels[i], training_images[i]);
    }
    for (l = 1; l < net->layer_count; l++) {
        for (j = 0; j < net->layer_size[l]; j++) {
            net->bias[l][j] -= eta * scratch->batch_del_bias[l][j] / batch_size;
            for (k = 0; k < net->layer_size[l-1]; k++) {
                net->weight[l][j*net->layer_size[l-1] + k]
                    -= eta * scratch->batch_del_weight[l][j*net->layer_size[l-1] + k] / batch_size;
            }
        }
    }
}

void
stochastic_gradient_descent_epoch(struct network *net, struct network_scratch *scratch, int training_count)
{
    const int batch_size = 10;
    const float eta = 3.0;
    int b;
    /* TODO: randomize batches. */
    for (b = 0; b+batch_size <= training_count; b+=batch_size) {
        batch(net, scratch, b, batch_size, eta);
    }
}

int
main(int argc, char *argv[])
{
    struct network net;
    struct network_scratch scratch;
    int epoch, i;

    init_network(&net, &scratch);
    load_images("mnist_train.csv", 60000, training_labels, training_images);

    printf("initial validation accuracy is %f\n", compute_accuracy(&net, &scratch, 10000, training_labels+50000, training_images+50000));
    for (epoch = 0; epoch < 10; epoch++) {
        printf("epoch %d...\n", epoch);
        stochastic_gradient_descent_epoch(&net, &scratch, 50000);
    }
    printf("validation accuracy is %f\n", compute_accuracy(&net, &scratch, 10000, training_labels+50000, training_images+50000));

    for (i = 0; i < 5; i++) {
        printf("label\t%d\n", training_labels[i]);
        printf("predict\t%d\n", predict_digit(&net, &scratch, training_images[i]));
        print_image(training_images[i]);
    }

    free_network(&net, &scratch);
    return 0;
}
