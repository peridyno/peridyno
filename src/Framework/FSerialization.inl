#include "Field.h"

namespace dyno 
{
	template<>
	inline std::string FVar<bool>::serialize()
	{
		if (isEmpty())
			return "";

		bool b = this->getValue();
		return b ? "true" : "false";
	}

	template<>
	inline bool FVar<bool>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		bool b = str == std::string("true") ? true : false;
		this->setValue(b);

		return true;
	}

	template<>
	inline std::string FVar<int>::serialize()
	{
		if (isEmpty())
			return "";

		int val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	inline bool FVar<int>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		int val = std::stoi(str);
		this->setValue(val);

		return true;
	}

	template<>
	inline std::string FVar<uint>::serialize()
	{
		if (isEmpty())
			return "";

		uint val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	inline bool FVar<uint>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		uint val = std::stoi(str);
		this->setValue(val);

		return true;
	}

	template<>
	inline std::string FVar<float>::serialize()
	{
		if (isEmpty())
			return "";

		float val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	inline bool FVar<float>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		float val = std::stof(str);
		this->setValue(val);

		return true;
	}

	template<>
	inline std::string FVar<double>::serialize()
	{
		if (isEmpty())
			return "";

		double val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	inline bool FVar<double>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		double val = std::stod(str);
		this->setValue(val);

		return true;
	}

	template<>
	inline std::string FVar<Vec3f>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3f val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	inline bool FVar<Vec3f>::deserialize(const std::string& str)
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
	inline std::string FVar<Vec3i>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3i val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	inline bool FVar<Vec3i>::deserialize(const std::string& str)
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
	inline std::string FVar<Vec3d>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3d val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	inline bool FVar<Vec3d>::deserialize(const std::string& str)
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
	inline std::string FVar<std::string>::serialize()
	{
		if (isEmpty())
			return "";

		std::string val = this->getValue();

		return val;
	}

	template<>
	inline bool FVar<std::string>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		this->setValue(str);

		return true;
	}

	template<>
	inline std::string FVar<Vec2f>::serialize()
	{
		if (isEmpty())
			return "";

		Vec2f val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y ;

		return ss.str();
	}

	template<>
	inline bool FVar<Vec2f>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		float x = std::stof(substr);

		ss >> substr;
		float y = std::stof(substr);

		this->setValue(Vec2f(x, y));

		return true;
	}

	template<>
	inline std::string FVar<Vec2d>::serialize()
	{
		if (isEmpty())
			return "";

		Vec2d val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y;

		return ss.str();
	}

	template<>
	inline bool FVar<Vec2d>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		double x = std::stod(substr);

		ss >> substr;
		double y = std::stod(substr);

		this->setValue(Vec2d(x, y));

		return true;
	}

	template<>
	inline std::string FVar<Quat1f>::serialize()
	{
		if (isEmpty())
			return "";

		Quat1f val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z << " " << val.w;

		return ss.str();
	}

	template<>
	inline bool FVar<Quat1f>::deserialize(const std::string& str)
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
		
		ss >> substr;
		double w = std::stod(substr);
		
		this->setValue(Quat1f(x, y, z, w));

		return true;
	}

	template<>
	inline std::string FVar<Quat1d>::serialize()
	{
		if (isEmpty())
			return "";

		Quat1d val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z << " " << val.w;

		return ss.str();
	}

	template<>
	inline bool FVar<Quat1d>::deserialize(const std::string& str)
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

		ss >> substr;
		double w = std::stod(substr);

		this->setValue(Quat1d(x, y, z, w));

		return true;
	}

	template<>
	inline std::string FVar<Mat3f>::serialize()
	{
		if (isEmpty())
			return "";

		Mat3f val = this->getValue();

		std::stringstream ss;
		ss << val(0, 0) << " " << val(0, 1) << " " << val(0, 2) << " "
			<< val(1, 0) << " " << val(1, 1) << " " << val(1, 2) << " "
			<< val(2, 0) << " " << val(2, 1) << " " << val(2, 2);

		return ss.str();
	}

	template<>
	inline bool FVar<Mat3f>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		double m00 = std::stod(substr);

		ss >> substr;
		double m01 = std::stod(substr);

		ss >> substr;
		double m02 = std::stod(substr);

		ss >> substr;
		double m10 = std::stod(substr);

		ss >> substr;
		double m11 = std::stod(substr);

		ss >> substr;
		double m12 = std::stod(substr);
		
		ss >> substr;
		double m20 = std::stod(substr);

		ss >> substr;
		double m21 = std::stod(substr);

		ss >> substr;
		double m22 = std::stod(substr);

		this->setValue(
			Mat3f(m00, m01, m02, 
				m10, m11, m12, 
				m20, m21, m22)
		);

		return true;
	}

	template<>
	inline std::string FVar<Mat3d>::serialize()
	{
		if (isEmpty())
			return "";

		Mat3d val = this->getValue();

		std::stringstream ss;
		ss << val(0, 0) << " " << val(0, 1) << " " << val(0, 2) << " "
			<< val(1, 0) << " " << val(1, 1) << " " << val(1, 2) << " "
			<< val(2, 0) << " " << val(2, 1) << " " << val(2, 2);

		return ss.str();
	}

	template<>
	inline bool FVar<Mat3d>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		double m00 = std::stod(substr);

		ss >> substr;
		double m01 = std::stod(substr);

		ss >> substr;
		double m02 = std::stod(substr);

		ss >> substr;
		double m10 = std::stod(substr);

		ss >> substr;
		double m11 = std::stod(substr);

		ss >> substr;
		double m12 = std::stod(substr);

		ss >> substr;
		double m20 = std::stod(substr);

		ss >> substr;
		double m21 = std::stod(substr);

		ss >> substr;
		double m22 = std::stod(substr);

		this->setValue(
			Mat3d(m00, m01, m02,
				m10, m11, m12,
				m20, m21, m22)
		);

		return true;
	}

	template<>
	inline std::string FVar<Mat4f>::serialize()
	{
		if (isEmpty())
			return "";

		Mat4f val = this->getValue();

		std::stringstream ss;
		ss << val(0, 0) << " " << val(0, 1) << " " << val(0, 2) << " " << val(0,3) << " "
			<< val(1, 0) << " " << val(1, 1) << " " << val(1, 2) << " " << val(1, 3) << " "
			<< val(2, 0) << " " << val(2, 1) << " " << val(2, 2) << " " << val(2, 3) << " "
			<< val(3, 0) << " " << val(3, 1) << " " << val(3, 2) << " " << val(3, 3) ;

		return ss.str();
	}

	template<>
	inline bool FVar<Mat4f>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		double m00 = std::stod(substr);

		ss >> substr;
		double m01 = std::stod(substr);

		ss >> substr;
		double m02 = std::stod(substr);

		ss >> substr;
		double m03 = std::stod(substr);

		ss >> substr;
		double m10 = std::stod(substr);

		ss >> substr;
		double m11 = std::stod(substr);

		ss >> substr;
		double m12 = std::stod(substr);

		ss >> substr;
		double m13 = std::stod(substr);

		ss >> substr;
		double m20 = std::stod(substr);

		ss >> substr;
		double m21 = std::stod(substr);

		ss >> substr;
		double m22 = std::stod(substr);

		ss >> substr;
		double m23 = std::stod(substr);

		ss >> substr;
		double m30 = std::stod(substr);

		ss >> substr;
		double m31 = std::stod(substr);

		ss >> substr;
		double m32 = std::stod(substr);

		ss >> substr;
		double m33 = std::stod(substr);

		this->setValue(
			Mat4f(m00, m01, m02, m03, 
				m10, m11, m12, m13,
				m20, m21, m22, m23,
				m30, m31, m32, m33)
		);

		return true;
	}

	template<>
	inline std::string FVar<Mat4d>::serialize()
	{
		if (isEmpty())
			return "";

		Mat4d val = this->getValue();

		std::stringstream ss;
		ss << val(0, 0) << " " << val(0, 1) << " " << val(0, 2) << " " << val(0, 3) << " "
			<< val(1, 0) << " " << val(1, 1) << " " << val(1, 2) << " " << val(1, 3) << " "
			<< val(2, 0) << " " << val(2, 1) << " " << val(2, 2) << " " << val(2, 3) << " "
			<< val(3, 0) << " " << val(3, 1) << " " << val(3, 2) << " " << val(3, 3);

		return ss.str();
	}

	template<>
	inline bool FVar<Mat4d>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		double m00 = std::stod(substr);

		ss >> substr;
		double m01 = std::stod(substr);

		ss >> substr;
		double m02 = std::stod(substr);

		ss >> substr;
		double m03 = std::stod(substr);

		ss >> substr;
		double m10 = std::stod(substr);

		ss >> substr;
		double m11 = std::stod(substr);

		ss >> substr;
		double m12 = std::stod(substr);

		ss >> substr;
		double m13 = std::stod(substr);

		ss >> substr;
		double m20 = std::stod(substr);

		ss >> substr;
		double m21 = std::stod(substr);

		ss >> substr;
		double m22 = std::stod(substr);

		ss >> substr;
		double m23 = std::stod(substr);

		ss >> substr;
		double m30 = std::stod(substr);

		ss >> substr;
		double m31 = std::stod(substr);

		ss >> substr;
		double m32 = std::stod(substr);

		ss >> substr;
		double m33 = std::stod(substr);

		this->setValue(
			Mat4d(m00, m01, m02, m03,
				m10, m11, m12, m13,
				m20, m21, m22, m23,
				m30, m31, m32, m33)
		);

		return true;
	}

	template<>
	inline std::string FVar<Transform3f>::serialize()
	{
		if (isEmpty())
			return "";

		Transform3f trans = this->getValue();
		auto t = trans.translation();
		auto val = trans.rotation();
		auto s = trans.scale();

		std::stringstream ss;
		ss << t.x << " " << t.y << " " << t.z << " "
			<< val(0, 0) << " " << val(0, 1) << " " << val(0, 2) << " "
			<< val(1, 0) << " " << val(1, 1) << " " << val(1, 2) << " "
			<< val(2, 0) << " " << val(2, 1) << " " << val(2, 2) << " "
			<< s.x << " " << s.y << " " << s.z;

		return ss.str();
	}

	template<>
	inline bool FVar<Transform3f>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		double tx = std::stod(substr);

		ss >> substr;
		double ty = std::stod(substr);

		ss >> substr;
		double tz = std::stod(substr);

		ss >> substr;
		double m00 = std::stod(substr);

		ss >> substr;
		double m01 = std::stod(substr);

		ss >> substr;
		double m02 = std::stod(substr);

		ss >> substr;
		double m10 = std::stod(substr);

		ss >> substr;
		double m11 = std::stod(substr);

		ss >> substr;
		double m12 = std::stod(substr);

		ss >> substr;
		double m20 = std::stod(substr);

		ss >> substr;
		double m21 = std::stod(substr);

		ss >> substr;
		double m22 = std::stod(substr);

		ss >> substr;
		double sx = std::stod(substr);

		ss >> substr;
		double sy = std::stod(substr);

		ss >> substr;
		double sz = std::stod(substr);

		this->setValue(
			Transform3f(Vec3f(tx, ty, tz), Mat3f(m00, m01, m02, m10, m11, m12, m20, m21, m22), Vec3f(sx, sy, sz))
		);

		return true;
	}

	template<>
	inline std::string FVar<Transform3d>::serialize()
	{
		if (isEmpty())
			return "";

		Transform3d trans = this->getValue();
		auto t = trans.translation();
		auto val = trans.rotation();
		auto s = trans.scale();

		std::stringstream ss;
		ss << t.x << " " << t.y << " " << t.z << " "
			<< val(0, 0) << " " << val(0, 1) << " " << val(0, 2) << " "
			<< val(1, 0) << " " << val(1, 1) << " " << val(1, 2) << " "
			<< val(2, 0) << " " << val(2, 1) << " " << val(2, 2) << " "
			<< s.x << " " << s.y << " " << s.z;

		return ss.str();
	}

	template<>
	inline bool FVar<Transform3d>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		double tx = std::stod(substr);

		ss >> substr;
		double ty = std::stod(substr);

		ss >> substr;
		double tz = std::stod(substr);

		ss >> substr;
		double m00 = std::stod(substr);

		ss >> substr;
		double m01 = std::stod(substr);

		ss >> substr;
		double m02 = std::stod(substr);

		ss >> substr;
		double m10 = std::stod(substr);

		ss >> substr;
		double m11 = std::stod(substr);

		ss >> substr;
		double m12 = std::stod(substr);

		ss >> substr;
		double m20 = std::stod(substr);

		ss >> substr;
		double m21 = std::stod(substr);

		ss >> substr;
		double m22 = std::stod(substr);

		ss >> substr;
		double sx = std::stod(substr);

		ss >> substr;
		double sy = std::stod(substr);

		ss >> substr;
		double sz = std::stod(substr);

		this->setValue(
			Transform3d(Vec3d(tx, ty, tz), Mat3d(m00, m01, m02, m10, m11, m12, m20, m21, m22), Vec3d(sx, sy, sz))
		);

		return true;
	}

	template class FVar<bool>;
	template class FVar<int>;
	template class FVar<uint>;
	template class FVar<float>;
	template class FVar<double>;
	template class FVar<Vec3f>;
	template class FVar<Vec3d>;
	template class FVar<Vec3i>;
	template class FVar<std::string>;
	template class FVar<Vec2f>;
	template class FVar<Vec2d>;
	template class FVar<Quat1f>;
	template class FVar<Quat1d>;
	template class FVar<Mat3f>;
	template class FVar<Mat3d>;
	template class FVar<Mat4f>;
	template class FVar<Mat4d>;
	template class FVar<Transform3f>;
	template class FVar<Transform3d>;

}
