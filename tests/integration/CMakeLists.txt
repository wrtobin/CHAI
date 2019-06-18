# Test dependencies
set (chai_integration_test_depends
     chai umpire gtest)

if (ENABLE_CUDA)
  set (chai_integration_test_depends
       ${chai_integration_test_depends} cuda)
endif ()

# ManagedArray tests
blt_add_executable(
  NAME managed_array_tests
  SOURCES managed_array_tests.cpp
  DEPENDS_ON ${chai_integration_test_depends})

target_include_directories(
  managed_array_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_array_test
  COMMAND managed_array_tests)

# managed_ptr tests
blt_add_executable(
  NAME managed_ptr_tests
  SOURCES managed_ptr_tests.cpp
  DEPENDS_ON ${chai_integration_test_depends})

target_include_directories(
  managed_ptr_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_ptr_test
  COMMAND managed_ptr_tests)

