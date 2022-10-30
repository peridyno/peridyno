/*
 * @file file_path_utilities.cpp
 * @brief Some universal functions when processing files' path.
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


#include "File_Utilities/file_path_utilities.h"
using std::string;

namespace dyno{

namespace FileUtilities{

string dirName(const string &path)
{
    string::size_type pos1 = path.rfind('/');
    string::size_type pos2 = path.rfind('\\');              //find the position of the last '/' or '\'
    if(pos1 != string::npos && pos2 != string::npos && pos1 < pos2) pos1 = pos2; //and store it in variable pos1
    else if(pos1 == string::npos && pos2 == string::npos) return string(".");
    else if(pos1 == string::npos) pos1 = pos2;
    if(path[pos1] == '\\') return path.substr(0,pos1-1);  //if pos1 stores char '\', we should return path.substr(0,pos1-1) because '\'always apear in pairs
    return path.substr(0,pos1);
}
string filenameInPath(const string &path)
{
    string::size_type pos1 = path.rfind('/');
    string::size_type pos2 = path.rfind('\\');
    if(pos1 != string::npos && pos2 != string::npos && pos1 < pos2) pos1 = pos2;
    else if(pos1 == string::npos && pos2 == string::npos) return path;
    else if(pos1 == string::npos) pos1 = pos2;
    return path.substr(pos1+1);
}
string fileExtension(const string &path)
{
    string::size_type pos = path.rfind('.');
    if(pos != string::npos)
        return path.substr(pos);
    else return string("");
}
string removeFileExtension(const std::string &path)
{
    string::size_type pos = path.rfind('.');
    if(pos != string::npos)
        return path.substr(0,pos);
    else
        return path;
}

} //end of namespace FilePathUtilities

} //end of namespace dyno

