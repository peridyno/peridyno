
# create the header file
file(WRITE ${SHADER_HEADER_FILE} "#pragma once\n")

file(APPEND ${SHADER_HEADER_FILE} "#include <map>\n")
file(APPEND ${SHADER_HEADER_FILE} "#include <string>\n\n")

# open variable
file(APPEND ${SHADER_HEADER_FILE} "const std::map<std::string, std::string> ShaderSource = {\n\n")    
# merge variables...
set(_IN_FILES ${SHADER_FILES} ${ARGN}) 

function(load_shader FILE)
	file(STRINGS ${FILE} lines_)
	get_filename_component(name_ ${FILE} NAME)
	get_filename_component(dir_ ${FILE} DIRECTORY)
	set(content_name_ content_${name_})

	get_property(property_content_ DIRECTORY PROPERTY ${content_name_})
	if(NOT DEFINED property_content_)
		set(out_)
		foreach(line_ ${lines_})
			string(REGEX MATCH "#include \"([^\"]+)\"" _ "${line_}")
			if(CMAKE_MATCH_COUNT GREATER 0)
				set(include_path_ ${CMAKE_MATCH_1})
				get_filename_component(include_name_ ${include_path_} NAME)
				set(include_ content_${include_name_})
				load_shader("${dir_}/${include_path_}")
				get_property(content_include_ DIRECTORY PROPERTY ${include_})
				string(APPEND out_ "${content_include_}\n")
			else()
				string(APPEND out_ "${line_}\n")
			endif()
		endforeach()
		set_property(DIRECTORY PROPERTY ${content_name_} "${out_}")
	endif()
endfunction()

foreach(SHADER_FILE ${_IN_FILES})	
	# get variable name from file name
	get_filename_component(name_ ${SHADER_FILE} NAME)
	file(APPEND ${SHADER_HEADER_FILE} "//// ${name_} ////\n")
	file(APPEND ${SHADER_HEADER_FILE} "{ \"${name_}\",\n")

	load_shader(${SHADER_FILE})

	get_property(content_ DIRECTORY PROPERTY content_${name_})
	string(LENGTH "${content_}" len_)
	set(pos_ 0)
	while(${pos_} LESS ${len_})
		math(EXPR next_len_ "${len_} - ${pos_}")
		# fix msvc C2026
		if(${next_len_} GREATER 16380)
			set(next_len_ 16380)
		endif()

		string(SUBSTRING "${content_}" ${pos_} ${next_len_} sub_content_)
		file(APPEND ${SHADER_HEADER_FILE} "R\"(${sub_content_})\" ")
		math(EXPR pos_ "${pos_} + ${next_len_}")
	endwhile()
	file(APPEND ${SHADER_HEADER_FILE} "\n},\n\n")
endforeach()	
# close variable
file(APPEND ${SHADER_HEADER_FILE} "};\n\n")
