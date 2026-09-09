// Decides three requirements of spec v2.0 by measurement instead of argument:
//   1. the Z_MAX cap on |zero|            (v2 5.3)
//   2. the (max-min)/2 absolute bound     (v2 7.1)
//   3. clamping scale up to the smallest f16 normal (v2 5.3)
// Models the encoder/decoder round trip in double with real f16 rounding, so it needs no ggml.
//   cc -O2 -o q4_hqq_zmax_probe q4_hqq_zmax_probe.c -lm && ./q4_hqq_zmax_probe
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define F16_MAX 65504.0f
#define F16_MIN_NORMAL 6.1035156e-5

static float rf16(float v) { _Float16 h = (_Float16) v; return (float) h; }

// one block round trip. zmax caps |zero|, smin clamps scale from below; 0 disables either
static double rmse_block(double zmax, double smin, const double * x, int n) {
    double vmin = x[0], vmax = x[0];
    for (int i = 1; i < n; i++) { if (x[i] < vmin) vmin = x[i]; if (x[i] > vmax) vmax = x[i]; }

    double scale = (vmax > vmin) ? 15.0/(vmax - vmin) : 1.0;
    const double a = fabs(vmin);
    if (a > 0.0) { const double cap = zmax/a; if (scale > cap) scale = cap; }
    if (scale > F16_MAX) scale = F16_MAX;
    if (smin > 0.0 && scale < smin) scale = smin;

    const float s = rf16((float) scale);
    const float z = rf16((float) (-vmin*scale));

    double se = 0.0;
    for (int i = 0; i < n; i++) {
        double t = x[i]*s + z;
        int q = (int) lround(t); if (q < 0) q = 0; if (q > 15) q = 15;
        const double y = s != 0.0f ? ((double) q - z)/s : 0.0;   // decoder guard
        se += (y - x[i])*(y - x[i]);
    }
    return sqrt(se/n);
}

int main(void) {
    puts("1. Z_MAX cap: shipped (65504) vs v2 (256), 400 random blocks per ratio");
    puts("   ratio      shipped        v2         shipped better");
    srand(7);
    const double ratios[] = { 20, 100, 1e3, 1e4, 1e5, 1e6, 1e7 };
    for (unsigned k = 0; k < sizeof(ratios)/sizeof(*ratios); k++) {
        const double r = ratios[k];
        double sum_a = 0, sum_b = 0, wins = 0;
        const int N = 400;
        for (int t = 0; t < N; t++) {
            const double off  = 1.0 + (double) rand()/RAND_MAX*2000.0;
            const double band = off/r;
            double x[32];
            for (int i = 0; i < 32; i++) x[i] = off + band*((double) rand()/RAND_MAX);
            const double a = rmse_block(65504.0, 0.0, x, 32);
            const double b = rmse_block(256.0, F16_MIN_NORMAL, x, 32);
            sum_a += a; sum_b += b; if (a < b) wins++;
        }
        printf("   %-8g %.4e  %.4e  %3.0f%%\n", r, sum_a/N, sum_b/N, 100.0*wins/N);
    }

    puts("\n2. (max-min)/2 bound on a narrow band far from zero");
    {
        double x[32];
        for (int i = 0; i < 32; i++) x[i] = 1000.0 + 1e-6*(i % 32);
        printf("   half-band            = %.5e\n", (x[31] - x[0])/2);
        printf("   rmse shipped         = %.5e\n", rmse_block(65504.0, 0.0, x, 32));
        printf("   rmse v2 encoder      = %.5e\n", rmse_block(256.0, F16_MIN_NORMAL, x, 32));
        printf("   |min|*2^-11 floor    = %.5e\n", 1000.0/2048.0);
    }

    puts("\n3. clamping scale up to the smallest f16 normal, range 1e-6..1e6");
    {
        double x[32];
        for (int i = 0; i < 32; i++) x[i] = (i % 2 == 0) ? 1e-6 : 1e6;
        printf("   rmse no lower clamp  = %.5e\n", rmse_block(65504.0, 0.0, x, 32));
        printf("   rmse clamp to normal = %.5e\n", rmse_block(65504.0, F16_MIN_NORMAL, x, 32));
    }
    return 0;
}
