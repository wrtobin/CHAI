#include "chai/ManagedArray.hpp"
#include "../util/forall.hpp"

#include <iostream>

int main(int CHAI_UNUSED_ARG(argc), char** CHAI_UNUSED_ARG(argv)) {

  std::cout << "Creating new arrays..." << std::endl;
  /*
   * Allocate two arrays with 10 elements of type float.
   */
  chai::ManagedArray<float> v1(10);
  chai::ManagedArray<float> v2(10);

  /*
   * Allocate an array on the device only
   */
  chai::ManagedArray<int> device_array(10, chai::GPU);

  std::cout << "Arrays created." << std::endl;

  std::cout << "Setting v1 on host." << std::endl;
  forall(sequential(), 0, 10, [=] (int i) {
      v1[i] = static_cast<float>(i * 1.0f);
  });

  std::cout << "v1 = [";
  forall(sequential(), 0, 10, [=] (int i) {
      std::cout << " " << v1[i];
  });
  std::cout << " ]" << std::endl;

  std::cout << "Setting v2 and device_array on device." << std::endl;
  forall(cuda(), 0, 10, [=] __device__ (int i) {
      v2[i] = v1[i]*2.0f;
      device_array[i] = i;
  });

  std::cout << "v2 = [";
  forall(sequential(), 0, 10, [=] (int i) {
      std::cout << " " << v2[i];
  });
  std::cout << " ]" << std::endl;

  std::cout << "Doubling v2 on device." << std::endl;
  forall(cuda(), 0, 10, [=] __device__ (int i) {
      v2[i] *= 2.0f;
  });

  std::cout << "Casting v2 to a pointer." << std::endl;
  float * raw_v2 = v2;

  std::cout << "raw_v2 = [";
  for (int i = 0; i < 10; i++ ) {
      std::cout << " " << raw_v2[i];
  }
  std::cout << " ]" << std::endl;

  std::cout << "Casting device_array to a pointer." << std::endl;
  int* raw_device_array = device_array;
  std::cout  << "device_array = [";
  for (int i = 0; i < 10; i++ ) {
      std::cout << " " << device_array[i];
  }
  std::cout << " ]" << std::endl;

  v1.free();
  v2.free();
  device_array.free();
}
