set(LIB_NAME VtkRenderEngine)

find_package(VTK COMPONENTS 
  RenderingOpenGL2
  RenderingExternal
  FiltersSources
  REQUIRED
)

file(GLOB_RECURSE SOURCES *.cpp *.h)

#group source files
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${LIB_NAME} STATIC ${SOURCES})

target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/${LIB_NAME}>)
	
target_link_libraries(${LIB_NAME} PUBLIC 
	Core 
	Framework 
	RenderCore
	${VTK_LIBRARIES}
)

vtk_module_autoinit(
	TARGETS ${LIB_NAME}
	MODULES ${VTK_LIBRARIES}
)

set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Rendering/Engine")



