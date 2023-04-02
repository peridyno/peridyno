set(TEST_PROJECT Homework)

link_libraries(Core)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp *.cu)

add_executable(${TEST_PROJECT} ${TEST_SOURCES})

add_test(NAME ${TEST_PROJECT} COMMAND ${TEST_PROJECT})

set_target_properties(${TEST_PROJECT} PROPERTIES FOLDER "Tests")
set_target_properties(${TEST_PROJECT} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_FLAGS})

target_link_libraries(${TEST_PROJECT} PUBLIC gtest)