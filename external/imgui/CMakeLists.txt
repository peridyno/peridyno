file(GLOB_RECURSE IMGUI_SRC
     LIST_DIRECTORIES false
     CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
	 "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(imgui STATIC ${IMGUI_SRC})
set_target_properties(imgui PROPERTIES
    OUTPUT_NAME "imgui-1.8.3")

set_target_properties(imgui PROPERTIES FOLDER "ThirdParty")

target_include_directories(imgui PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(imgui glad)