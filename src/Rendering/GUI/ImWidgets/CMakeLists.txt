set(LIB_NAME ImWidgets)

file(GLOB_RECURSE LIB_SRC *.cpp *.h)

add_library(${LIB_NAME} STATIC ${LIB_SRC})
set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Rendering/GUI")

target_include_directories(${LIB_NAME} PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/GUI>)
	
target_link_libraries(${LIB_NAME} Core Framework Rendering imgui)

target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)