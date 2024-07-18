CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

set(LIB_NAME WtGUI)
set(LIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

#set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)

# external include directory
include_directories(${PROJECT_SOURCE_DIR}/external)

# add sources
file(GLOB_RECURSE SOURCES "${LIB_SRC_DIR}/*.cpp" "${LIB_SRC_DIR}/*.h" "${LIB_SRC_DIR}/*.hpp")
add_library(${LIB_NAME} SHARED ${SOURCES})

foreach(SRC IN ITEMS ${SOURCES})
    get_filename_component(SRC_PATH "${SRC}" PATH)
    file(RELATIVE_PATH SRC_PATH_REL "${LIB_SRC_DIR}" "${SRC_PATH}")
    string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
    source_group("${GROUP_PATH}" FILES "${SRC}")
endforeach()

target_link_libraries(${LIB_NAME} PRIVATE glfw)

# nvJPEG
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${LIB_NAME} PRIVATE CUDA::nvjpeg)

# Wt
find_package(wt REQUIRED)
set_target_properties(Wt::Wt PROPERTIES MAP_IMPORTED_CONFIG_RELEASE RelWithDebInfo)
set_target_properties(Wt::HTTP PROPERTIES MAP_IMPORTED_CONFIG_RELEASE RelWithDebInfo)
target_link_libraries(${LIB_NAME} PRIVATE Wt::Wt Wt::HTTP)

# link dyno
target_link_libraries(${LIB_NAME} PUBLIC Core Framework Topology ImWidgets GLRenderEngine)

# for python scripting
# Jian: put pybind11 at end since there could be header conflict for node.h
add_subdirectory(${PROJECT_SOURCE_DIR}/external/pybind11 pybind11)
target_link_libraries(${LIB_NAME} PRIVATE pybind11::embed)

if(MSVC)
    set_target_properties(${LIB_NAME} 
		PROPERTIES 
			#COMPILE_FLAGS "/W3 /wd4251 /wd4275" # /Wp64
			VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
			#VS_DEBUGGER_COMMAND_ARGUMENTS "--docroot ${CMAKE_SOURCE_DIR}/docroot --http-listen 0.0.0.0:5000"
			)
			
	# copy dependency dlls
	add_custom_command(TARGET ${LIB_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different 
						$<TARGET_FILE:${LIB_NAME}>		# in case no dlls
						$<TARGET_RUNTIME_DLLS:${LIB_NAME}>
						${CMAKE_BINARY_DIR}
                        #$<TARGET_FILE_DIR:${target}>
		COMMAND_EXPAND_LISTS
	)

	# set as startup project for visual studio
	set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${LIB_NAME})
endif(MSVC) 

set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Rendering/GUI")
set_target_properties(${LIB_NAME} PROPERTIES
    OUTPUT_NAME "dyno${LIB_NAME}-${PERIDYNO_LIBRARY_VERSION}")
    
if(WIN32)
    set_target_properties(${LIB_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
elseif(UNIX)
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set_target_properties(${LIB_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
    else()
        set_target_properties(${LIB_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
    endif()
endif()

target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/GUI>
    $<BUILD_INTERFACE:${PERIDYNO_ROOT}/src/Rendering/GUI/WtGUI>	
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}>
    $<INSTALL_INTERFACE:${PERIDYNO_INC_INSTALL_DIR}/Rendering/GUI>)

install(TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}Targets
    RUNTIME  DESTINATION  ${PERIDYNO_RUNTIME_INSTALL_DIR}
    LIBRARY  DESTINATION  ${PERIDYNO_LIBRARY_INSTALL_DIR}
    ARCHIVE  DESTINATION  ${PERIDYNO_ARCHIVE_INSTALL_DIR}
    )

install(EXPORT ${LIB_NAME}Targets DESTINATION ${PERIDYNO_CMAKE_CONFIG_INSTALL_DIR}
    FILE ${LIB_NAME}Targets.cmake)

# install dependency files
install(FILES "dependency/bin/wtd.dll" "dependency/bin/wthttpd.dll"
	TYPE BIN CONFIGURATIONS Debug OPTIONAL)
install(FILES "dependency/bin/wt.dll" "dependency/bin/wthttp.dll"
	TYPE BIN CONFIGURATIONS Release OPTIONAL)
install(DIRECTORY dependency/bin DESTINATION . OPTIONAL
	FILES_MATCHING PATTERN "python*.*")

#Append ${LIB_NAME}Targets.cmake to the global list, which will be include in PeridynoConfig.cmake
get_property(LOCAL_CMAKES_NAMES GLOBAL PROPERTY "GLOBAL_CMAKES_NAMES")
list(APPEND LOCAL_CMAKES_NAMES "${LIB_NAME}Targets.cmake")    
set_property(GLOBAL PROPERTY GLOBAL_CMAKES_NAMES ${LOCAL_CMAKES_NAMES})

file(GLOB FILE_QTGUI_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${FILE_QTGUI_HEADER}  DESTINATION ${PERIDYNO_INC_INSTALL_DIR}/Rendering/GUI/WtGUI)
