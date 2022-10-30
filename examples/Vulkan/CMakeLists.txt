cmake_minimum_required(VERSION 3.10)

#get all sub directories
macro(GET_SUBDIR_LIST result curdir)
    file(GLOB children RELATIVE ${curdir} ${curdir}/*)
    set(dirlist "")
    foreach (child ${children})
        if (IS_DIRECTORY ${curdir}/${child})
            list(APPEND dirlist ${child})
        endif ()
    endforeach ()
    set(${result} ${dirlist})
endmacro()

# call add_subdirectory, add_dependencies for every sub directory
GET_SUBDIR_LIST(SubDirs ${CMAKE_CURRENT_SOURCE_DIR})
foreach (SubDir ${SubDirs})
    add_subdirectory(${SubDir})
    add_dependencies(${SubDir} CompileShader)
endforeach ()