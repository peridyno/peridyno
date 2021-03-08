set(GLEW_HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/glew.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/glxew.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GL/wglew.h)

set(GLEW_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/glew.c)

add_library(glew STATIC ${GLEW_HEADER} ${GLEW_SOURCE})
set_target_properties(glew PROPERTIES
        OUTPUT_NAME "glew-1.10.0")

set_target_properties(glew PROPERTIES FOLDER "ThirdParty")
target_include_directories(glew PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(glew opengl32)

install(TARGETS glew
    EXPORT glewTargets
    RUNTIME  DESTINATION  ${PERIDYNO_RUNTIME_INSTALL_DIR}
    LIBRARY  DESTINATION  ${PERIDYNO_LIBRARY_INSTALL_DIR}
    ARCHIVE  DESTINATION  ${PERIDYNO_ARCHIVE_INSTALL_DIR}
    )

install(FILES ${GLEW_HEADER} DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Extern/glew-1.10.0/include)

install(EXPORT glewTargets DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR}
    FILE glewTargets.cmake)