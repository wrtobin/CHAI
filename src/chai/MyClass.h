#ifdef __CUDACC__

void verify_on_host(int value) {
   printf("verify_on_host: %d\n", value);
}

template <typename T>
void verify_on_host_templated(T value) {
   printf("verify_on_host_templated: %d\n", value);
}

__global__ void verify_on_device(int value) {
   printf("verify_on_device: %d\n", value);
}

template <typename T>
__global__ void verify_on_device_templated(T value) {
   printf("verify_on_device_templated: %d\n", value);
}

//template __global__ void verify_on_device_templated<int>(int value);

template <typename T>
class MyClass {
   public:
      __host__ __device__ MyClass() {}
      __host__ __device__ ~MyClass() {
#ifndef __CUDA_ARCH__
         cleanup();
#endif
      }

      __host__ void cleanup() {
         T test = 5;
         verify_on_host(test);
         verify_on_host_templated(test);
         verify_on_device<<<1, 1>>>(test);
         verify_on_device_templated<<<1, 1>>>(test);
      }
};

#endif
