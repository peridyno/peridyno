cmake_minimum_required(VERSION 3.10)

#foreach(LIB_NAME IN ITEMS ${LIB_NAMES})
macro(build_library LIB_NAME LIB_DEPENDENCY)
    set(LIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}")

    file(                                                                           
        GLOB_RECURSE LIB_SRC
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${LIB_SRC_DIR}/*.c*"
        "${LIB_SRC_DIR}/*.h*"
    )

    if(WIN32)
        add_library(${LIB_NAME} STATIC ${LIB_SRC})
    elseif(UNIX)
        add_library(${LIB_NAME} SHARED ${LIB_SRC}) 
    endif()

    foreach(SRC IN ITEMS ${LIB_SRC}) 
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${LIB_SRC_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()

    if(WIN32)
        target_compile_options(${LIB_NAME} PRIVATE -Xcompiler "/wd 4819") 
    endif()
    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine/Dynamics")
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES "60")
    
    set_target_properties(${LIB_NAME} PROPERTIES
        OUTPUT_NAME "dyno${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")
    set_target_properties(${LIB_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF)

    target_include_directories(${LIB_NAME} PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/Dynamics>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Dynamics>)

    #To disable the warning "calling a constexpr __host__ function("***") from a __host__ __device__ function("***") is not allowed."
    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda>)

    target_link_libraries(${LIB_NAME} PUBLIC ${${LIB_DEPENDENCY}})

    install(TARGETS ${LIB_NAME}
        EXPORT ${LIB_NAME}Targets
        RUNTIME  DESTINATION  ${PERIDYNO_RUNTIME_INSTALL_DIR}
        LIBRARY  DESTINATION  ${PERIDYNO_LIBRARY_INSTALL_DIR}
        ARCHIVE  DESTINATION  ${PERIDYNO_ARCHIVE_INSTALL_DIR}
        )

    install(EXPORT ${LIB_NAME}Targets DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR}
        FILE ${LIB_NAME}Targets.cmake)

    #Append ${LIB_NAME}Targets.cmake to the global list, which will be include in PeridynoConfig.cmake
    get_property(LOCAL_CMAKES_NAMES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")
    list(APPEND LOCAL_CMAKES_NAMES "${LIB_NAME}Targets.cmake")    
    set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES ${LOCAL_CMAKES_NAMES})

    file(GLOB FILE_DYNAMICS_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/${LIB_NAME}/*.h")
    install(FILES ${FILE_DYNAMICS_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Dynamics/${LIB_NAME})
endmacro()

option(PERIDYNO_LIBRARY_PARTICLESYSTEM "Enable binding the particle system library" ON)
option(PERIDYNO_LIBRARY_PERIDYNAMICS "Enable binding the peridynamics library" ON)
option(PERIDYNO_LIBRARY_RIGIDBODY "Enable binding the rigid body library" ON)
option(PERIDYNO_LIBRARY_VOLUME "Enable binding the volume library" ON)
option(PERIDYNO_LIBRARY_HEIGHTFIELD "Enable binding the Height Field library" ON)

if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY Core IO Framework)
    build_library(ParticleSystem LIB_DEPENDENCY)

    if(PERIDYNO_LIBRARY_PERIDYNAMICS)
    set(LIB_DEPENDENCY ParticleSystem)
        build_library(Peridynamics LIB_DEPENDENCY)
    endif()
endif()

if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY Core IO Framework)
    build_library(RigidBody LIB_DEPENDENCY)
endif()

if(PERIDYNO_LIBRARY_VOLUME)
    set(LIB_DEPENDENCY Core IO Framework)
    build_library(Volume LIB_DEPENDENCY)
endif()


if(PERIDYNO_LIBRARY_HEIGHTFIELD)
    set(LIB_DEPENDENCY Core IO Framework cufft)
    build_library(HeightField LIB_DEPENDENCY)
endif()