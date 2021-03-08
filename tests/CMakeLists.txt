cmake_minimum_required(VERSION 3.10)

add_subdirectory(Test_Array)
add_subdirectory(Test_Core)

if(PERIDYNO_LIBRARY_FRAMEWORK)
    add_subdirectory(Test_Topolopy)
    add_subdirectory(Test_Collision)
endif()

if(PERIDYNO_LIBRARY_FRAMEWORK)
    add_subdirectory(Test_Rendering)
endif()