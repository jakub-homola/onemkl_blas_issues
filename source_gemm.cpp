#include <cstdio>

#include <omp.h>
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>





int main()
{
    size_t count = 10;
    size_t n = 300;

    printf("Starting the program\n");

    std::vector<std::vector<double>> As(count);
    std::vector<std::vector<double>> Bs(count);
    std::vector<std::vector<double>> Cs(count);
#pragma omp parallel for
    for(size_t i = 0; i < count; i++)
    {
        As[i].resize(n * n);
        Bs[i].resize(n * n);
        Cs[i].resize(n * n);
    }



    {
        sycl::queue q(sycl::gpu_selector{});

        std::vector<std::unique_ptr<sycl::buffer<double,1>>> buf_As(count);
        std::vector<std::unique_ptr<sycl::buffer<double,1>>> buf_Bs(count);
        std::vector<std::unique_ptr<sycl::buffer<double,1>>> buf_Cs(count);
        
#pragma omp parallel for
        for(size_t i = 0; i < count; i++)
        {
            for(size_t r = 0; r < n; r++)
                for(size_t c = 0; c < n; c++)
                    As[i][r * n + c] = 0.2*r/n + 0.3*c/n + r%10/10.0 + c%15/15.0 + 0.1*i/count;
            for(size_t r = 0; r < n; r++)
                for(size_t c = 0; c < n; c++)
                    Bs[i][r * n + c] = 0.4*r/n + 0.1*c/n + r%10/10.0 + c%15/15.0 + 0.1*i/count;

            buf_As[i] = std::make_unique<sycl::buffer<double,1>>(As[i].data(), sycl::range<1>(n * n));
            buf_Bs[i] = std::make_unique<sycl::buffer<double,1>>(Bs[i].data(), sycl::range<1>(n * n));
            buf_Cs[i] = std::make_unique<sycl::buffer<double,1>>(Cs[i].data(), sycl::range<1>(n * n));

            printf("  submitting gemm for matrix %zu in thread %d\n", i, omp_get_thread_num());
            oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, n, n, n, 1.0, *buf_As[i], n, *buf_Bs[i], n, 0.0, *buf_Cs[i], n);
            printf("  submitted gemm for matrix %zu in thread %d\n", i, omp_get_thread_num());
        }
    }

    printf("Program finished OK\n\n");

    return 0;
}
