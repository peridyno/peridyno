
# create the header file
file(WRITE ${SHADER_HEADER_FILE} "#pragma once\n")

file(APPEND ${SHADER_HEADER_FILE} "#include <map>\n")
file(APPEND ${SHADER_HEADER_FILE} "#include <string>\n\n")

# open variable
file(APPEND ${SHADER_HEADER_FILE} "const std::map<std::string, std::string> ShaderSource = {\n\n")    
# merge variables...
set(_IN_FILES ${SHADER_FILES} ${ARGN}) 

foreach(SHADER_FILE ${_IN_FILES})	
	# get variable name from file name
	get_filename_component(VAR_NAME ${SHADER_FILE} NAME)
	file(APPEND ${SHADER_HEADER_FILE} "//// ${VAR_NAME} ////\n")
	file(APPEND ${SHADER_HEADER_FILE} "{ \"${VAR_NAME}\",\n")
	
	file(STRINGS ${SHADER_FILE} LINES)
    foreach(LINE ${LINES})
		string(REPLACE "\"" "\\\"" LINE "${LINE}")
		file(APPEND ${SHADER_HEADER_FILE} "\"")
		file(APPEND ${SHADER_HEADER_FILE} "${LINE}")
		file(APPEND ${SHADER_HEADER_FILE} "\\n\"\n")
	endforeach()
	file(APPEND ${SHADER_HEADER_FILE} "},\n\n")	
endforeach()	
# close variable
file(APPEND ${SHADER_HEADER_FILE} "};\n\n")
	