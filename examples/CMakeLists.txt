add_subdirectory(General) 

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
    add_subdirectory(Cuda)
endif()

if("${PERIDYNO_GPU_BACKEND}" STREQUAL "Vulkan")
    add_subdirectory(Vulkan) 
endif()