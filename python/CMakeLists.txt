cmake_minimum_required(VERSION 3.10)

add_subdirectory(${PROJECT_SOURCE_DIR}/external/pybind11 pybind11)

set(PyBind_Name PyPeridyno)

file(GLOB_RECURSE PY_ALL_SOURCE_FILES "*.c*" "*.h*")

if(WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
elseif(UNIX)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Debug)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Debug)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Debug)
    else()
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Release)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Release)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/Release)
    endif()
endif()

link_libraries(Core Framework IO)
link_libraries(GlfwGUI)
link_libraries(ParticleSystem RigidBody Peridynamics GLRenderEngine)

file(COPY "Testing/" DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
file(COPY "Testing/" DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)

pybind11_add_module(${PyBind_Name} ${PY_ALL_SOURCE_FILES})

set_target_properties(${PyBind_Name} PROPERTIES
    OUTPUT_NAME "${PyBind_Name}-${PERIDYNO_LIBRARY_VERSION}")

install(TARGETS ${PyBind_Name}
    RUNTIME  DESTINATION  ${PERIDYNO_RUNTIME_INSTALL_DIR}
    LIBRARY  DESTINATION  ${PERIDYNO_LIBRARY_INSTALL_DIR}
    ARCHIVE  DESTINATION  ${PERIDYNO_ARCHIVE_INSTALL_DIR}
    )