if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
    add_subdirectory(Core)
endif() 

add_subdirectory(Engine)

if(PERIDYNO_LIBRARY_FRAMEWORK)
    add_subdirectory(GUI)
endif()