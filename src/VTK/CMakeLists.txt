set(LIB_NAME VtkVisualModule)

find_package(VTK COMPONENTS 
  RenderingOpenGL2
  RenderingExternal
  REQUIRED
)

file(GLOB_RECURSE SOURCES *.cpp *.h)
add_library(${LIB_NAME} STATIC ${SOURCES})

target_link_libraries(${LIB_NAME} PUBLIC Core Framework RenderEngine ${VTK_LIBRARIES})
vtk_module_autoinit(
	TARGETS ${LIB_NAME}
	MODULES ${VTK_LIBRARIES}
)

set_target_properties(${LIB_NAME} PROPERTIES FOLDER "VTK")
target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}>)

