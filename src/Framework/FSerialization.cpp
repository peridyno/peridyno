#include "Field.h"
#include "DeclareEnum.h"
#include "Vector.h"

namespace dyno 
{
	template<>
	std::string FVar<bool>::serialize()
	{
		if (isEmpty())
			return "";

		bool b = this->getValue();
		return b ? "true" : "false";
	}

	template<>
	bool FVar<bool>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		bool b = str == std::string("true") ? true : false;
		this->setValue(b);

		return true;
	}

	template<>
	std::string FVar<int>::serialize()
	{
		if (isEmpty())
			return "";

		int val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<int>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		int val = std::stoi(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<uint>::serialize()
	{
		if (isEmpty())
			return "";

		uint val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<uint>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		uint val = std::stoi(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<float>::serialize()
	{
		if (isEmpty())
			return "";

		float val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<float>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		float val = std::stof(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<double>::serialize()
	{
		if (isEmpty())
			return "";

		double val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<double>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		double val = std::stod(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<Vec3f>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3f val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	bool FVar<Vec3f>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;
		
		ss >> substr;
		float x = std::stof(substr);

		ss >> substr;
		float y = std::stof(substr);

		ss >> substr;
		float z = std::stof(substr);


		this->setValue(Vec3f(x, y, z));

		return true;
	}

	template<>
	std::string FVar<Vec3i>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3i val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	bool FVar<Vec3i>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		int x = std::stoi(substr);

		ss >> substr;
		int y = std::stoi(substr);

		ss >> substr;
		int z = std::stoi(substr.c_str());

		this->setValue(Vec3i(x, y, z));

		return true;
	}

	template<>
	std::string FVar<Vec3d>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3d val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	bool FVar<Vec3d>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		double x = std::stod(substr);

		ss >> substr;
		double y = std::stod(substr);

		ss >> substr;
		double z = std::stod(substr);

		this->setValue(Vec3d(x, y, z));

		return true;
	}

// 	template<>
// 	std::string FVar<FilePath>::serialize()
// 	{
// 		if (isEmpty())
// 			return "";
// 
// 		FilePath val = this->getValue();
// 
// 		return val.string();
// 	}
// 
// 	template<>
// 	bool FVar<FilePath>::deserialize(const std::string& str)
// 	{
// 		if (str.empty())
// 			return false;
// 
// 		this->setValue(str);
// 
// 		return true;
// 	}

	template<>
	std::string FVar<PEnum>::serialize()
	{
		if (isEmpty())
			return "";

		int key = this->getDataPtr()->currentKey();

		std::stringstream ss;
		ss << key;

		return ss.str();
	}

	template<>
	bool FVar<PEnum>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		int key = std::stod(str);

		this->getDataPtr()->setCurrentKey(key);

		return true;
	}

	template<>
	std::string FVar<std::string>::serialize()
	{
		if (isEmpty())
			return "";

		std::string val = this->getValue();

		return val;
	}

	template<>
	bool FVar<std::string>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		this->setValue(str);

		return true;
	}	
}
