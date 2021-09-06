cmake_minimum_required(VERSION 3.10)

#set(LIB_NAMES GlutGUI QtGUI)

include_directories(
	${CMAKE_SOURCE_DIR}/Source
	${CMAKE_CURRENT_SOURCE_DIR}
)                                                              

add_subdirectory(ImWidgets)
add_subdirectory(GlfwGUI)

if(PERIDYNO_Qt5_GUI)
    add_subdirectory(QtGUI)
endif()

file(GLOB FILE_GUI_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${FILE_GUI_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/GUI)