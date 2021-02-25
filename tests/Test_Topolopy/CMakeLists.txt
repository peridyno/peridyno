set(TEST_PROJECT Test_Topology)

link_libraries(Core Framework IO)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_executable(${TEST_PROJECT} ${TEST_SOURCES})

add_test(NAME ${TEST_PROJECT} COMMAND ${TEST_PROJECT})

set_target_properties(${TEST_PROJECT} PROPERTIES FOLDER "Tests")

target_link_libraries(${TEST_PROJECT} PUBLIC gtest)