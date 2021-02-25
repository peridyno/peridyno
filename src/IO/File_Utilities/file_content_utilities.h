/*
 * @file file_content_utilities.h 
 * @brief Some universal functions when preprocessing file content.
 * @author Fei Zhu, LiYou Xu
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

#ifndef PHYSIKA_CORE_UTILITIES_FILE_UTILITIES_FILE_CONTENT_UTILITIES_H_
#define PHYSIKA_CORE_UTILITIES_FILE_UTILITIES_FILE_CONTENT_UTILITIES_H_

#include<string>

namespace dyno{

namespace FileUtilities{

//remove abundant whitespaces
//replace each whitespace squence with a squence of fixed number of white spaces 
std::string removeWhitespaces(const std::string &line, unsigned int num_retained_spaces = 0);

} //end of namespace FileUtilities

} //end of namespace dyno

#endif //PHYSIKA_CORE_UTILITIES_FILE_UTILITIES_FILE_CONTENT_UTILITIES_H_
