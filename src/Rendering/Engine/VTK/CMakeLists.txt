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

#To resolve the error: Target "..." INTERFACE_INCLUDE_DIRECTORIES property contains path: "..." which is prefixed in the build directory.
target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/VTK>)

#add_dependencies(${LIB_NAME} BuildShaderHeader)
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

file(GLOB CORE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${CORE_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/Engine/VTK)
