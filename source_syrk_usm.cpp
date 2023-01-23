#include <cstdio>

#include <omp.h>
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>





int main()
{
    size_t count = 10;
    size_t k = 1000;
    size_t n = 100;

    printf("Starting the program\n");
    
    sycl::queue q(sycl::gpu_selector{});

    using my_allocator_t = sycl::usm_allocator<double,sycl::usm::alloc::shared>;
    using my_vector_t = std::vector<double,my_allocator_t>;
    my_allocator_t allocator(q);
    std::vector<my_vector_t> As(count, my_vector_t(allocator));
    std::vector<my_vector_t> Cs(count, my_vector_t(allocator));

#pragma omp parallel for
    for(size_t i = 0; i < count; i++)
    {
        As[i].resize(n * k);
        Cs[i].resize(n * n);

        for(size_t r = 0; r < n; r++)
            for(size_t c = 0; c < k; c++)
                As[i][r * k + c] = 0.2*r/n + 0.3*c/k + r%10/10.0 + c%15/15.0 + 0.1*i/count;

        printf("  submitting syrk for matrix %zu in thread %d\n", i, omp_get_thread_num());
        oneapi::mkl::blas::row_major::syrk(q, oneapi::mkl::uplo::upper, oneapi::mkl::transpose::nontrans, n, k, 1.0, As[i].data(), k, 0.0, Cs[i].data(), n);
        printf("  submitted syrk for matrix %zu in thread %d\n", i, omp_get_thread_num());
    }

    printf("Waiting for kernels to finish\n");

    q.wait();

    printf("Program finished OK\n\n");

    return 0;
}
