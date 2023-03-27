set(TEST_PROJECT Test_Topology)

link_libraries(Core Framework Topology)

file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)

add_executable(${TEST_PROJECT} ${TEST_SOURCES})

add_test(NAME ${TEST_PROJECT} COMMAND ${TEST_PROJECT})

set_target_properties(${TEST_PROJECT} PROPERTIES FOLDER "Tests")

target_link_libraries(${TEST_PROJECT} PUBLIC gtest)

if(WIN32)
    set_target_properties(${TEST_PROJECT} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
elseif(UNIX)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set_target_properties(${TEST_PROJECT} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
    else()
        set_target_properties(${TEST_PROJECT} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
    endif()
endif()   