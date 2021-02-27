cmake_minimum_required(VERSION 3.10)

option(PERIDYNO_TESTS_CORE "Enable building peridyno tests for the core library" ON)
if(PERIDYNO_TESTS_CORE)
    add_subdirectory(Test_Array)
    add_subdirectory(Test_Core)
endif()

option(PERIDYNO_TESTS_FRAMEWORK "Enable building peridyno tests for the framework library" ON)
if(PERIDYNO_TESTS_FRAMEWORK)
    add_subdirectory(Test_Topolopy)
    add_subdirectory(Test_Collision)
endif()

option(PERIDYNO_TESTS_GRAPHICS "Enable building peridyno tests for the graphics library" ON)
if(PERIDYNO_TESTS_GRAPHICS)
    add_subdirectory(Test_Rendering)
endif()