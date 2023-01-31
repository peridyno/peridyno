/*
 * @file parseline.cpp
 * @brief Some universal functions when processing file content.
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


#include "File_Utilities/file_content_utilities.h"
using std::string;

namespace dyno{

namespace FileUtilities{

string removeWhitespaces(const string &line, unsigned int num_retained_spaces)
{
    string::size_type pos;
    string new_line = line;
    string whitespace(" "), retained_whitespaces(" ");
    for(unsigned int i = 0; i < num_retained_spaces; ++i)
        retained_whitespaces += whitespace;
    while(new_line[0] == ' ')
        new_line = new_line.substr(1);
    while((pos=new_line.find(retained_whitespaces)) != string::npos)
        new_line.erase(pos,1);
    return new_line;
}

} //end of namespace FileUtilities

} //end of namespace dyno

