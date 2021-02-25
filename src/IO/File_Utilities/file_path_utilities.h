/*
 * @file file_path_utilities.h 
 * @brief Some universal functions when processing files' name.
 * @author LiYou Xu, Fei Zhu
 * @acknowledge Jernej Barbic, author of VegaFEM
 *
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_UTILITIES_FILE_UTILITIES_FILE_PATH_UTILITIES_H_
#define PHYSIKA_CORE_UTILITIES_FILE_UTILITIES_FILE_PATH_UTILITIES_H_

#include<string>

namespace dyno{

namespace FileUtilities{

//extract the path of a file's directory out of its path. if the path doesn't have a directory, it will return string('.').
std::string dirName(const std::string &path);

//extract filename in a path of a file
std::string filenameInPath(const std::string &path);

//extract the file extension out of its path. if the file doesn't have a file extension, it will return empty string.  ex. fileExtension(string("five.txt"))  will return string(".txt") .
std::string fileExtension(const std::string &path);

std::string removeFileExtension(const std::string &path);

} //end of namespace File_Utilities

} //end of namespace dyno

#endif //PHYSIKA_CORE_UTILITIES_FILE_UTILITIES_FILE_PATH_UTILITIES_H_
