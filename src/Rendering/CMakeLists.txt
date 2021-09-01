add_library(Rendering INTERFACE Rendering.h)
set_target_properties(Rendering PROPERTIES FOLDER "Rendering")

target_include_directories(Rendering INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>)

add_subdirectory(Core)
add_subdirectory(Engine)

if(PERIDYNO_LIBRARY_FRAMEWORK AND PERIDYNO_LIBRARY_GUI)
    add_subdirectory(GUI)
endif()