set(LIB_NAME VtkApp)

# VTK
find_package(VTK COMPONENTS 
  RenderingCore
  #RenderingExternal
  RenderingOpenGL2
  InteractionStyle
  InteractionWidgets
  REQUIRED
)

file(GLOB_RECURSE SOURCES *.cpp *.h)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
add_library(${LIB_NAME} STATIC ${SOURCES})

target_link_libraries(${LIB_NAME} PUBLIC Core Framework VtkVisualModule ${VTK_LIBRARIES})
 vtk_module_autoinit(
    TARGETS ${LIB_NAME}
    MODULES ${VTK_LIBRARIES}
    )

set_target_properties(${LIB_NAME} PROPERTIES FOLDER "VTK")

#To resolve the error: Target "..." INTERFACE_INCLUDE_DIRECTORIES property contains path: "..." which is prefixed in the build directory.
target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}>)
