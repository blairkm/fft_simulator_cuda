#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NX 1024
#define LX 100.0
#define DT 0.0005
#define NSTEP 10000
#define OUTPUT_EVERY 200

// CUDA error checking macro
#define CUDA_CALL(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Global parameters (set from argv)
static double X0, S0, E0, BH, BW, EH;
static int option;
static double A = 0.0, B = 0.0, C = 0.0, k_ = 0.0, omega = 0.0;

static double psi[NX][2], v[NX], u[NX][2];
static double al[2][2], bux[2][NX][2], blx[2][NX][2];
static double dx;

// Device arrays
static double* psi_d; // psi[i*2], psi[i*2+1]
static double* wrk_d;
static double* v_d;
static double* u_d;   // u[i*2], u[i*2+1]
static double* al_d;  // al[stp][s], stored as linear array of size 4
static double* bux_d; // bux[2][NX][2]
static double* blx_d; // blx[2][NX][2]

// KERNELS

// potential half-step: psi(x)*=u(x)
__global__ void pot_prop_kernel(double* psi, double* u) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX) {
        double re = psi[2 * i];
        double im = psi[2 * i + 1];
        double ur = u[2 * i];
        double ui = u[2 * i + 1];
        double wr = re * ur - im * ui;
        double wi = re * ui + im * ur;
        psi[2 * i] = wr;
        psi[2 * i + 1] = wi;
    }
}

// kinetic step kernel
__global__ void kin_prop_kernel(int t, double* psi, double* wrk,
    double* al, double* bux, double* blx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX) {
        int ip = (i + 1) % NX;
        int im = (i + NX - 1) % NX;

        double alr = al[t * 2 + 0];
        double ali = al[t * 2 + 1];

        double buxr = bux[(t * NX + i) * 2 + 0];
        double buxi = bux[(t * NX + i) * 2 + 1];

        double blxr = blx[(t * NX + i) * 2 + 0];
        double blxi = blx[(t * NX + i) * 2 + 1];

        double psi_re = psi[2 * i];
        double psi_im = psi[2 * i + 1];

        double psi_re_im = psi[2 * im];
        double psi_im_im = psi[2 * im + 1];

        double psi_re_ip = psi[2 * ip];
        double psi_im_ip = psi[2 * ip + 1];

        double wr_out = alr * psi_re - ali * psi_im
            + blxr * psi_re_im - blxi * psi_im_im
            + buxr * psi_re_ip - buxi * psi_im_ip;

        double wi_out = alr * psi_im + ali * psi_re
            + blxr * psi_im_im + blxi * psi_re_im
            + buxr * psi_im_ip + buxi * psi_re_ip;

        wrk[2 * i] = wr_out;
        wrk[2 * i + 1] = wi_out;
    }
}

// copy wrk back to psi
__global__ void copy_kernel(double* psi, double* wrk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX * 2) {
        psi[i] = wrk[i];
    }
}

// update potential operator u(x) = exp(-i*v(x)*DT/2)
__global__ void update_u_kernel(double* v, double* u, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX) {
        double phase = -0.5 * v[i] * dt;
        u[2 * i] = cos(phase);
        u[2 * i + 1] = sin(phase);
    }
}

// update the potential for time-dependent/custom potentials
__global__ void update_potential_kernel(double* v, double A, double B, double C,
    double k_, double omega, double t, int option) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX) {
        double dx = LX / NX;
        double Xval = i * dx;
        double val = 0.0;
        switch (option) {
        case 0:
            // time-independent barrier; no update needed
            // Just keep the original barrier
            // (This kernel can be skipped if option=0)
            break;
        case 1:
            // custom sine/cos no time dep: A*cos(k*x)+B*sin(k*x)+C
            val = A * cos(k_ * Xval) + B * sin(k_ * Xval) + C;
            break;
        case 2:
            // time-dependent linear: A*x + B*t
            val = A * Xval + B * t;
            break;
        case 3:
            // time-dependent harmonic: A*cos(omega*t)*x^2 + C
            val = A * cos(omega * t) * Xval * Xval + C;
            break;
        case 4:
            // combination: A*cos(k*x) + B*sin(omega*t) + C
            val = A * cos(k_ * Xval) + B * sin(omega * t) + C;
            break;
        default:
            val = 0.0;
        }
        if (option != 0) v[i] = val;
    }
}

// HOST FUNCTIONS

void init_param() {
    dx = LX / NX;
}

void init_wavefn() {
    double sum = 0.0;
    for (int sx = 0; sx < NX; sx++) {
        double x = dx * (sx)-X0;
        double gauss = exp(-0.25 * x * x / (S0 * S0));
        double re = gauss * cos(sqrt(2.0 * E0) * x);
        double im = gauss * sin(sqrt(2.0 * E0) * x);
        psi[sx][0] = re;
        psi[sx][1] = im;
        sum += (re * re + im * im);
    }
    sum *= dx;
    double norm_fac = 1.0 / sqrt(sum);
    for (int sx = 0; sx < NX; sx++) {
        psi[sx][0] *= norm_fac;
        psi[sx][1] *= norm_fac;
    }
}

void init_potential() {
    double dx = LX / NX;
    for (int i = 0; i < NX; i++) {
        double Xval = i * dx;
        double val = 0.0;
        switch (option) {
        case 0:
            // Default barrier
            if (i == 0 || i == NX - 1) val = EH;
            else if (0.5 * (LX - BW) < Xval && Xval < 0.5 * (LX + BW))
                val = BH;
            else val = 0.0;
            break;
        case 1:
            // A*cos(k*x)+B*sin(k*x)+C
            val = A * cos(k_ * Xval) + B * sin(k_ * Xval) + C;
            break;
        case 2:
            // At t=0: A*x + B*0
            val = A * Xval;
            break;
        case 3:
            // At t=0: A * x^2 + C
            val = A * Xval * Xval + C;
            break;
        case 4:
            // At t=0: A*cos(k*x) + C
            val = A * cos(k_ * Xval) + C;
            break;
        default:
            val = 0.0;
        }
        v[i] = val;
    }
}

void init_prop() {
    double a = 0.5 / (dx * dx);
    for (int stp = 0; stp < 2; stp++) {
        double exp_p_re = cos(-(stp + 1) * DT * a);
        double exp_p_im = sin(-(stp + 1) * DT * a);

        double ep_re = 0.5 * (1.0 + exp_p_re);
        double ep_im = 0.5 * exp_p_im;
        double em_re = 0.5 * (1.0 - exp_p_re);
        double em_im = -0.5 * exp_p_im;

        al[stp][0] = ep_re;
        al[stp][1] = ep_im;

        for (int i = 0; i < NX; i++) {
            int up, lw;
            if (stp == 0) {
                up = i % 2;
                lw = (i + 1) % 2;
            }
            else {
                up = (i + 1) % 2;
                lw = i % 2;
            }
            bux[stp][i][0] = up * em_re;
            bux[stp][i][1] = up * em_im;
            blx[stp][i][0] = lw * em_re;
            blx[stp][i][1] = lw * em_im;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 9) {
        printf("Usage: %s X0 S0 E0 BH BW EH option outfile [A B C k omega]\n", argv[0]);
        return 1;
    }

    X0 = atof(argv[1]);
    S0 = atof(argv[2]);
    E0 = atof(argv[3]);
    BH = atof(argv[4]);
    BW = atof(argv[5]);
    EH = atof(argv[6]);
    option = atoi(argv[7]);

    FILE* fp = fopen(argv[8], "w+");
    if (!fp) {
        printf("Error opening file\n");
        return 1;
    }

    if (argc > 9) A = atof(argv[9]);
    if (argc > 10) B = atof(argv[10]);
    if (argc > 11) C = atof(argv[11]);
    if (argc > 12) k_ = atof(argv[12]);
    if (argc > 13) omega = atof(argv[13]);

    init_param();
    init_wavefn();
    init_potential();
    init_prop();

    // Allocate GPU memory
    CUDA_CALL(cudaMalloc((void**)&psi_d, sizeof(double) * 2 * NX));
    CUDA_CALL(cudaMalloc((void**)&wrk_d, sizeof(double) * 2 * NX));
    CUDA_CALL(cudaMalloc((void**)&v_d, sizeof(double) * NX));
    CUDA_CALL(cudaMalloc((void**)&u_d, sizeof(double) * 2 * NX));
    CUDA_CALL(cudaMalloc((void**)&al_d, sizeof(double) * 4));
    CUDA_CALL(cudaMalloc((void**)&bux_d, sizeof(double) * 2 * NX * 2));
    CUDA_CALL(cudaMalloc((void**)&blx_d, sizeof(double) * 2 * NX * 2));

    // Copy data to device
    CUDA_CALL(cudaMemcpy(psi_d, psi, sizeof(double) * 2 * NX, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(v_d, v, sizeof(double) * NX, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(al_d, al, sizeof(double) * 4, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(bux_d, bux, sizeof(double) * 2 * NX * 2, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(blx_d, blx, sizeof(double) * 2 * NX * 2, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (NX + threadsPerBlock - 1) / threadsPerBlock;

    // Compute u(x)
    update_u_kernel << <blocks, threadsPerBlock >> > (v_d, u_d, DT);
    cudaDeviceSynchronize();

    // Output initial
    CUDA_CALL(cudaMemcpy(psi, psi_d, sizeof(double) * 2 * NX, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(v, v_d, sizeof(double) * NX, cudaMemcpyDeviceToHost));

    fprintf(fp, "timestamp: %.5f\n", 0.0);
    fprintf(fp, "params: %d %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf\n",
        NX, LX, DT, X0, S0, E0, option, A, B, C, k_, omega, BH, BW, EH);
    fprintf(fp, "psi_re: ");
    for (int i = 0; i < NX; i++) fprintf(fp, "%le ", psi[i][0]);
    fprintf(fp, "\npsi_im: ");
    for (int i = 0; i < NX; i++) fprintf(fp, "%le ", psi[i][1]);
    fprintf(fp, "\n");
    fprintf(fp, "pot: ");
    for (int i = 0; i < NX; i++) fprintf(fp, "%le ", v[i]);
    fprintf(fp, "\n\n");

    auto kin_step = [&](int t) {
        kin_prop_kernel << <blocks, threadsPerBlock >> > (t, psi_d, wrk_d, al_d, bux_d, blx_d);
        cudaDeviceSynchronize();
        copy_kernel << <(2 * NX + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (psi_d, wrk_d);
        cudaDeviceSynchronize();
        };

    for (int step = 1; step <= NSTEP; step++) {
        double time = step * DT;

        // If time-dependent potential, update it
        if (option > 0) {
            update_potential_kernel << <blocks, threadsPerBlock >> > (v_d, A, B, C, k_, omega, time, option);
            cudaDeviceSynchronize();
            // Update u(x)
            update_u_kernel << <blocks, threadsPerBlock >> > (v_d, u_d, DT);
            cudaDeviceSynchronize();
        }

        // pot half-step
        pot_prop_kernel << <blocks, threadsPerBlock >> > (psi_d, u_d);
        cudaDeviceSynchronize();

        // kin half-step: t=0
        kin_step(0);

        // kin full-step: t=1
        kin_step(1);

        // kin half-step: t=0 again
        kin_step(0);

        // pot half-step
        pot_prop_kernel << <blocks, threadsPerBlock >> > (psi_d, u_d);
        cudaDeviceSynchronize();

        if (step % OUTPUT_EVERY == 0) {
            CUDA_CALL(cudaMemcpy(psi, psi_d, sizeof(double) * 2 * NX, cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(v, v_d, sizeof(double) * NX, cudaMemcpyDeviceToHost));

            fprintf(fp, "timestamp: %.5f\n", time);
            fprintf(fp, "params: %d %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf\n",
                NX, LX, DT, X0, S0, E0, option, A, B, C, k_, omega, BH, BW, EH);

            fprintf(fp, "psi_re: ");
            for (int i = 0; i < NX; i++) fprintf(fp, "%le ", psi[i][0]);
            fprintf(fp, "\npsi_im: ");
            for (int i = 0; i < NX; i++) fprintf(fp, "%le ", psi[i][1]);
            fprintf(fp, "\n");

            fprintf(fp, "pot: ");
            for (int i = 0; i < NX; i++) fprintf(fp, "%le ", v[i]);
            fprintf(fp, "\n\n");
        }
    }

    fclose(fp);

    cudaFree(psi_d);
    cudaFree(wrk_d);
    cudaFree(v_d);
    cudaFree(u_d);
    cudaFree(al_d);
    cudaFree(bux_d);
    cudaFree(blx_d);

    return 0;
}
