set(LIB_NAME glad)
set(GLAD_GL_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/glad/glad.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/glad/glad.c")

add_library(glad SHARED ${GLAD_GL_SRC})

# Are we building a shared library?
get_target_property(library_type glad TYPE)
if (library_type STREQUAL SHARED_LIBRARY)
  # If so, define the macro GLAD_API_EXPORT on the command line.
  target_compile_definitions(glad PUBLIC GLAD_GLAPI_EXPORT PRIVATE GLAD_GLAPI_EXPORT_BUILD)
endif()

set_target_properties(glad PROPERTIES
    OUTPUT_NAME "glad-2.0.0")

set_target_properties(glad PROPERTIES FOLDER "ThirdParty")
set_target_properties(glad PROPERTIES C_STANDARD 99)

target_include_directories(glad PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

install(TARGETS ${LIB_NAME}
EXPORT ${LIB_NAME}Targets
RUNTIME  DESTINATION  ${PERIDYNO_RUNTIME_INSTALL_DIR}
LIBRARY  DESTINATION  ${PERIDYNO_LIBRARY_INSTALL_DIR}
ARCHIVE  DESTINATION  ${PERIDYNO_ARCHIVE_INSTALL_DIR}
)

install(EXPORT ${LIB_NAME}Targets DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR}
    FILE ${LIB_NAME}Targets.cmake)

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/glad/*.h")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/external/glad)