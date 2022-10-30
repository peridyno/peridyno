
set(EXAMPLE_NAME App_ImGUI)

SET(EXAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "Generating project file for example in ${EXAMPLE_FOLDER}")
# Main
file(GLOB SOURCE *.cpp ${BASE_HEADERS} ${EXAMPLE_FOLDER}/*.cpp)
if(WIN32)
    add_executable(${EXAMPLE_NAME} WIN32 ${SOURCE} ${MAIN_HEADER})
    target_link_libraries(${EXAMPLE_NAME} Core ImGUI IO VkRenderEngine ${Vulkan_LIBRARY} ${WINLIBS})
else(WIN32)
    add_executable(${EXAMPLE_NAME} ${SOURCE} ${MAIN_HEADER} )
    target_link_libraries(${EXAMPLE_NAME} Core ImGUI IO VkRenderEngine)
endif(WIN32)

set_target_properties(${EXAMPLE_NAME} PROPERTIES FOLDER "Examples")