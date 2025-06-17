macro(add_plugin LIB_NAME LIB_DEPENDENCY)
    set(LIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

    append_library(${LIB_NAME})

    file(                                                                           
        GLOB_RECURSE LIB_SRC
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${LIB_SRC_DIR}/*.c*"
        "${LIB_SRC_DIR}/*.h*"
    )

    add_library(${LIB_NAME} SHARED ${LIB_SRC}) 

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
    set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Plugins")
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_FLAGS})

    if(WIN32)
        set_target_properties(${LIB_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    elseif(UNIX)
        if (CMAKE_BUILD_TYPE MATCHES Debug)
            set_target_properties(${LIB_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
        else()
            set_target_properties(${LIB_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
        endif()
    endif()
    

    set_target_properties(${LIB_NAME} PROPERTIES
        OUTPUT_NAME "plugin-${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")
    set_target_properties(${LIB_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF)

    add_compile_definitions(PERIDYNO_API_EXPORTS)

    target_include_directories(${LIB_NAME} PUBLIC 
        $<BUILD_INTERFACE:${PERIDYNO_ROOT}/plugins>
        $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>/plugins)

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

    file(GLOB FILE_DYNAMICS_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
    install(FILES ${FILE_DYNAMICS_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/plugins/${LIB_NAME})
endmacro()

macro(add_example EXAMPLE_NAME GROUP_NAME LIB_DEPENDENCY)
    set(PROJECT_NAME ${EXAMPLE_NAME})

    get_property(LIB_NAMES GLOBAL PROPERTY PERIDYNO_LIBRARIES)
#   message("List ${LIB_NAMES}")
    foreach(LIB_NAME ${${LIB_DEPENDENCY}})
        string(FIND "${LIB_NAMES}" "${LIB_NAME}" TARGET_FOUND)
        if(TARGET_FOUND EQUAL -1)
            message("${LIB_NAME} not found! \n")
            return()
        endif()
    endforeach()

    file(  
        GLOB_RECURSE SRC_LIST 
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
    )

    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_LIST})

    add_executable(${PROJECT_NAME} ${SRC_LIST})                                                                 

    target_link_libraries(${PROJECT_NAME} 
        ${${LIB_DEPENDENCY}})

    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                  
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Examples/${GROUP_NAME}") 
    if("${PERIDYNO_GPU_BACKEND}" STREQUAL "CUDA")
        set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_FLAGS})
    endif()

    if(WIN32)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    elseif(UNIX)
        if (CMAKE_BUILD_TYPE MATCHES Debug)
            set_target_properties(${PROJECT_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
        else()
            set_target_properties(${PROJECT_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
        endif()
    endif()
endmacro()


macro(add_peridyno_test TEST_NAME LIB_DEPENDENCY)
    set(PROJECT_NAME ${TEST_NAME})

    file(  
        GLOB_RECURSE SRC_LIST 
        LIST_DIRECTORIES false
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
    )

    add_executable(${PROJECT_NAME} ${SRC_LIST})                                                                 

    target_link_libraries(${PROJECT_NAME} ${${LIB_DEPENDENCY}})

    file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")                  
    set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Tests")

    if(WIN32)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    elseif(UNIX)
        if (CMAKE_BUILD_TYPE MATCHES Debug)
            set_target_properties(${PROJECT_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
        else()
            set_target_properties(${PROJECT_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
        endif()
    endif()
endmacro()

macro(peridyno_install LIB_NAME)
    install(TARGETS ${LIB_NAME}
        EXPORT ${LIB_NAME}Targets
        RUNTIME  DESTINATION  ${PERIDYNO_RUNTIME_INSTALL_DIR}
        LIBRARY  DESTINATION  ${PERIDYNO_LIBRARY_INSTALL_DIR}
        ARCHIVE  DESTINATION  ${PERIDYNO_ARCHIVE_INSTALL_DIR}
        )

    install(EXPORT ${LIB_NAME}Targets DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR}
        FILE ${LIB_NAME}Targets.cmake)

    get_property(LOCAL_CMAKES_NAMES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")
    list(APPEND LOCAL_CMAKES_NAMES "${LIB_NAME}Targets.cmake")    
    set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES ${LOCAL_CMAKES_NAMES})


    file(GLOB FRAMEWORK_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.inl")
    install(FILES ${FRAMEWORK_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME})

    file(GLOB ITEMS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(ITEM in ${ITEMS})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${ITEM})
            file(GLOB FILE_DYNAMICS_MODULE "${CMAKE_CURRENT_SOURCE_DIR}/${ITEM}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/${ITEM}/*.inl")
            install(FILES ${FILE_DYNAMICS_MODULE}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}/${ITEM})
        endif()
    endforeach()
endmacro()

macro(append_library LIB_NAME)
    get_property(LIB_NAMES GLOBAL PROPERTY PERIDYNO_LIBRARIES)
    list(APPEND LIB_NAMES ${LIB_NAME})
    set_property(GLOBAL PROPERTY PERIDYNO_LIBRARIES ${LIB_NAMES})
endmacro()