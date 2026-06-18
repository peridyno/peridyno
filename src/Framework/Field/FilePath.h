#pragma once

#include <ghc/fs_std.hpp>

namespace dyno 
{
	class  FilePath
	{
	public:
		FilePath(std::string s = "",std::string filter = "")
		{
			fs_path = s;
			file_filter = filter;
		}

		//return the full string of the path
		const std::string string() { return fs_path.string(); }
		fs::path& path() { return fs_path; }

		inline bool operator== (const std::string& s) const {
			return fs_path == s;
		}

		inline bool operator!= (const std::string& s) const {
			return fs_path != s;
		}

		bool is_path() { return _is_path; }

		std::vector<std::string>& extensions() { return exts; }

		void add_extension(const std::string ext) { exts.push_back(ext); }

		void set_as_path(bool b) { _is_path = b; }

		void set_path(std::string s) { fs_path = s; }

		std::string getFileFilter() { return file_filter; }

		void setFilter(std::string filter) { file_filter = filter; }

	private:
		bool _is_path = false;

		fs::path fs_path;
		std::vector<std::string> exts;
		std::string file_filter = "";
	};

	class SaveFilePath : public FilePath
	{
	public:
		SaveFilePath(std::string s,std::string filter = "") :FilePath(s, filter)
		{
		
		}

	private:

	};


}

#include "FilePath.inl"