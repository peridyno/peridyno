cmake_minimum_required(VERSION 3.10)

add_subdirectory(Test_Array)
add_subdirectory(Test_Core)

if(PERIDYNO_LIBRARY_FRAMEWORK)
    add_subdirectory(Test_Topolopy)
    add_subdirectory(Test_Collision)
    add_subdirectory(Test_Field)
    add_subdirectory(Test_Pipeline)
    add_subdirectory(Test_CCD)
endif()

if(PERIDYNO_LIBRARY_VOLUME)
    add_subdirectory(Test_Volume)
endif()