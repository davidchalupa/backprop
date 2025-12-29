#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*
 * A basic backpropagation implementation with 1 hidden layer
 *
 * - Uses biases on neurons (hidden and output)
 * - Supports variable input size and hidden layer size (single hidden layer)
 * - Defaults to XOR dataset (2 inputs, 2 hidden units)
 * - Keeps the original runtime behavior: infinite training loop, periodic printing
 *   of weights & outputs and a getchar() pause every PRINT_EVERY iterations.
 * - Designed so training dataset can be replace with another boolean
 *   or pseudo-boolean function easily.
 */

#define MAX_INPUT 10
#define MAX_HIDDEN 10
#define MAX_DATA 1024

int input_size = 2;
int hidden_size = 2;
// number of training samples
int data_count = 4;

const long long PRINT_EVERY = 10000;

double lrate = 0.1;

// network parameters
double w_input[MAX_HIDDEN][MAX_INPUT];
double b_hidden[MAX_HIDDEN];                    // bias for each hidden unit
double w_hidden[MAX_HIDDEN];                    // output weights from hidden -> output
double b_output;                                // output bias

double current_hidden[MAX_HIDDEN];

// training dataset (default: XOR)
// the data is stored row-major: trainX[k][i]
double trainX[MAX_DATA][MAX_INPUT];
double trainY[MAX_DATA];

static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

static double rand_small()
{
    return (((double) rand() / (double) RAND_MAX) - 0.5) * 0.2;
}

// initializing network weights
void init_network()
{
    for (int h = 0; h < hidden_size; ++h)
    {
        for (int i = 0; i < input_size; ++i)
        {
            w_input[h][i] = rand_small();
        }
        b_hidden[h] = rand_small();
        w_hidden[h] = rand_small();
    }
    b_output = rand_small();
}

// computing forward pass for an input vector x (length input_size); returns output
double compute_x_out(const double *x)
{
    for (int h = 0; h < hidden_size; ++h)
    {
        double net = b_hidden[h];
        for (int i = 0; i < input_size; ++i)
        {
            net += w_input[h][i] * x[i];
        }
        current_hidden[h] = sigmoid(net);
    }

    double net_out = b_output;
    for (int h = 0; h < hidden_size; ++h)
    {
        net_out += w_hidden[h] * current_hidden[h];
    }
    return sigmoid(net_out);
}

// populate default XOR dataset
void load_default_xor()
{
    if (input_size != 2)
    {
        fprintf(stderr, "default XOR data requires input_size == 2\n");
        exit(1);
    }
    data_count = 4;
    trainX[0][0] = 0.0; trainX[0][1] = 0.0; trainY[0] = 0.0;
    trainX[1][0] = 0.0; trainX[1][1] = 1.0; trainY[1] = 1.0;
    trainX[2][0] = 1.0; trainX[2][1] = 0.0; trainY[2] = 1.0;
    trainX[3][0] = 1.0; trainX[3][1] = 1.0; trainY[3] = 0.0;
}

// Example helper to load a generic parity-like dataset for arbitrary input_size.
// This fills trainX/trainY with all 2^input_size binary inputs and their parity.
// Use only for small input_size (<= MAX_INPUT and 2^input_size <= MAX_DATA).
void load_parity_dataset()
{
    int maxrows = 1 << input_size;
    if (maxrows > MAX_DATA)
    {
        fprintf(stderr, "parity dataset too large for MAX_DATA=%d\n", MAX_DATA);
        exit(1);
    }
    data_count = maxrows;
    for (int r = 0; r < maxrows; ++r)
    {
        int parity = 0;
        for (int i = 0; i < input_size; ++i)
        {
            int bit = (r >> i) & 1;
            trainX[r][i] = (double)bit;
            parity ^= bit;
        }
        trainY[r] = (double)parity;
    }
}

/* print current weights & biases (compact) */
void print_weights()
{
    printf("Weights and biases:\n");
    for (int h = 0; h < hidden_size; ++h)
    {
        printf("hidden %d: w_input:", h);
        for (int i = 0; i < input_size; ++i)
            printf(" %0.3f", w_input[h][i]);
        printf(" | b_hidden: %0.3f | w_out: %0.3f\n", b_hidden[h], w_hidden[h]);
    }
    printf("b_output: %0.3f\n", b_output);
}

/* print dataset outputs; for input_size==2 keep original layout for familiarity */
void print_dataset_outputs()
{
    if (input_size == 2)
    {
        for (int k = 0; k < data_count; ++k)
        {
            double out = compute_x_out(trainX[k]);
            printf("%0.3f %0.3f | %0.3f %0.3f\n",
                   trainX[k][0], trainX[k][1], out, trainY[k]);
        }
    }
    else
    {
        for (int k = 0; k < data_count; ++k)
        {
            double out = compute_x_out(trainX[k]);
            printf("in:");
            for (int i = 0; i < input_size; ++i) printf(" %0.0f", trainX[k][i]);
            printf(" | out: %0.3f | target: %0.3f\n", out, trainY[k]);
        }
    }
}

int main()
{
    /* default settings already set at top; change here if you want different sizes */
    input_size = 2;
    hidden_size = 2;

    load_default_xor();

    srand((unsigned) time(NULL));

    init_network();

    long long t = 0;

    while (1)
    {
        // pick a random training sample
        int j = rand() % data_count;

        if (t % PRINT_EVERY == 0)
        {
            print_weights();
        }

        double out = compute_x_out(trainX[j]);
        double error_out = trainY[j] - out;

        // update output weights and bias
        for (int h = 0; h < hidden_size; ++h)
        {
            w_hidden[h] += lrate * error_out * current_hidden[h];
        }
        b_output += lrate * error_out;

        // update input -> hidden weights and hidden biases
        for (int h = 0; h < hidden_size; ++h)
        {
            double hval = current_hidden[h];
            double dh = hval * (1.0 - hval);
            double back = error_out * w_hidden[h] * dh;
            for (int i = 0; i < input_size; ++i)
            {
                w_input[h][i] += lrate * back * trainX[j][i];
            }
            b_hidden[h] += lrate * back;
        }

        if (t % PRINT_EVERY == 0)
        {
            print_dataset_outputs();
            getchar();
        }

        t++;
    }

    return 0;
}
