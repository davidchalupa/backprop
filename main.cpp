#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "random_generator.h"

int data_count = 4;
double trainx1[10] = {0.0,0.0,1.0,1.0};
double trainx2[10] = {0.0,1.0,0.0,1.0};
double trainy[10] = {0.0,1.0,1.0,0.0};

int layer_size = 2;
double w_input[10][10] = {{0.01,0.01},{0.01,0.01}};
double w_hidden[10] = {0.01,0.01};

double delta_hidden[10] = {0.01,0.01};

double x_out;

double current_hidden[10];

double error_out;
double error_hidden[10];
double error_input[10];

double n = 1;

double lrate = 0.1;

double compute_x_out(double trainx1val, double trainx2val)
{
    int i;

    // input and hidden layer
    for (i=0;i<layer_size;i++)
    {
        current_hidden[i] = 0.0;
        current_hidden[i] += w_input[i][0]*trainx1val;
        current_hidden[i] += w_input[i][1]*trainx2val;
        current_hidden[i] = 1.0 / (1.0 + exp(-current_hidden[i]));
    }

    // output layer
    double x_out = 0.0;
    for (i=0;i<layer_size;i++)
    {
        x_out += w_hidden[i]*current_hidden[i];
    }
    x_out = 1.0 / (1.0 + exp(-x_out));

    return x_out;
}

int main()
{
    int i,j,k; long long t = 0;
    random_generator generator;

    srand((unsigned long long) time(NULL));

    for (i=0;i<layer_size;i++)
    {
        w_input[i][0] = generator.random_double();
        w_input[i][1] = generator.random_double();
        w_hidden[i] = generator.random_double();
    }

    while (1)
    {
        j = generator.random(0,data_count-1);
        if (t % 10000 == 0)
        {
            printf("Weights:\n");
            for (i=0;i<layer_size;i++)
            {
                printf("w_input: %0.3lf, %0.3lf\n", w_input[i][0], w_input[i][1]);
                printf("w_hidden: %0.3lf\n", w_hidden[i]);
            }
        }

        x_out = compute_x_out(trainx1[j], trainx2[j]);

        error_out = trainy[j] - x_out;

        // backprop - output layer
        // delta = η*error*z
        for (i=0;i<layer_size;i++)
        {
            delta_hidden[i] = lrate * error_out * current_hidden[i];
        }

        // backprop - input and hidden layer
        // delta = η*error*weightout*z*(1−z)*input
        for (i=0;i<layer_size;i++)
        {
            w_input[i][0] += lrate * error_out * w_hidden[i] * (current_hidden[i]) * (1.0-current_hidden[i]) * trainx1[j];
            w_input[i][1] += lrate * error_out * w_hidden[i] * (current_hidden[i]) * (1.0-current_hidden[i]) * trainx2[j];
        }

        for (i=0;i<layer_size;i++)
        {
            w_hidden[i] += delta_hidden[i];
        }

        if (t % 10000 == 0)
        {
            for (k=0;k<data_count;k++)
            {
                printf("%0.3lf %0.3lf | %0.3lf %0.3lf\n", trainx1[k], trainx2[k], compute_x_out(trainx1[k], trainx2[k]), trainy[k]);
            }
            getchar();
        }

        t++;
    }

    return 0;
}
