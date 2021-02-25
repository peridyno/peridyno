#include <limits>
#include <glm/gtx/norm.hpp>

namespace dyno {

	template <typename T>
	DYN_FUNC Vector<T, 3>::Vector()
		:Vector(0) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>::Vector(T x)
		: Vector(x, x, x) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>::Vector(T x, T y, T z)
		: data_(x, y, z)
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>::Vector(const Vector<T, 3>& vec)
		: data_(vec.data_)
	{

	}

	template <typename T>
	DYN_FUNC Vector<T, 3>::~Vector()
	{
	}

	template <typename T>
	DYN_FUNC T& Vector<T, 3>::operator[] (unsigned int idx)
	{
		return const_cast<T &> (static_cast<const Vector<T, 3> &>(*this)[idx]);
	}

	template <typename T>
	DYN_FUNC const T& Vector<T, 3>::operator[] (unsigned int idx) const
	{
		return data_[idx];
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator+ (const Vector<T, 3> &vec2) const
	{
		return Vector<T, 3>(*this) += vec2;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator+= (const Vector<T, 3> &vec2)
	{
		data_ += vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator- (const Vector<T, 3> &vec2) const
	{
		return Vector<T, 3>(*this) -= vec2;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator-= (const Vector<T, 3> &vec2)
	{
		data_ -= vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator*(const Vector<T, 3> &vec2) const
	{
		return Vector<T, 3>(data_[0] * vec2[0], data_[1] * vec2[1], data_[2] * vec2[2]);
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator*=(const Vector<T, 3> &vec2)
	{
		data_[0] *= vec2.data_[0];	data_[1] *= vec2.data_[1];	data_[2] *= vec2.data_[2];
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator/(const Vector<T, 3> &vec2) const
	{
		return Vector<T, 3>(data_[0] / vec2[0], data_[1] / vec2[1], data_[2] / vec2[2]);
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator/=(const Vector<T, 3> &vec2)
	{
		data_[0] /= vec2.data_[0];	data_[1] /= vec2.data_[1];	data_[2] /= vec2.data_[2];
		return *this;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator=(const Vector<T, 3> &vec2)
	{
		data_ = vec2.data_;
		return *this;
	}


	template <typename T>
	DYN_FUNC bool Vector<T, 3>::operator== (const Vector<T, 3> &vec2) const
	{
		return data_ == vec2.data_;
	}

	template <typename T>
	DYN_FUNC bool Vector<T, 3>::operator!= (const Vector<T, 3> &vec2) const
	{
		return !((*this) == vec2);
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator+(T value) const
	{
		return Vector<T, 3>(*this) += value;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator+= (T value)
	{
		data_ += value;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator-(T value) const
	{
		return Vector<T, 3>(*this) -= value;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator-= (T value)
	{
		data_ -= value;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator* (T scale) const
	{
		return Vector<T, 3>(*this) *= scale;
	}

	template <typename T>
	Vector<T, 3>& Vector<T, 3>::operator*= (T scale)
	{
		data_ *= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator/ (T scale) const
	{
		return Vector<T, 3>(*this) /= scale;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator/= (T scale)
	{
		data_ /= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator-(void) const
	{
		Vector<T, 3> res;
		res.data_ = -data_;
		return res;
	}

	template <typename T>
	DYN_FUNC T Vector<T, 3>::norm() const
	{
		return glm::length(data_);
	}

	template <typename T>
	DYN_FUNC T Vector<T, 3>::normSquared() const
	{
		return glm::length2(data_);
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::normalize()
	{
		data_ = glm::length(data_) > glm::epsilon<T>() ? glm::normalize(data_) : glm::tvec3<T>(0, 0, 0);
		return *this;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3> Vector<T, 3>::cross(const Vector<T, 3>& vec2) const
	{
		Vector<T, 3> res;
		res.data_ = glm::cross(data_, vec2.data_);
		return res;
	}

	template <typename T>
	DYN_FUNC T Vector<T, 3>::dot(const Vector<T, 3>& vec2) const
	{
		return glm::dot(data_, vec2.data_);
	}

	template <typename T>
	DYN_FUNC Vector<T, 3> Vector<T, 3>::minimum(const Vector<T, 3>& vec2) const
	{
		Vector<T, 3> res;
		res[0] = data_[0] < vec2[0] ? data_[0] : vec2[0];
		res[1] = data_[1] < vec2[1] ? data_[1] : vec2[1];
		res[2] = data_[2] < vec2[2] ? data_[2] : vec2[2];
		return res;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3> Vector<T, 3>::maximum(const Vector<T, 3>& vec2) const
	{
		Vector<T, 3> res;
		res[0] = data_[0] > vec2[0] ? data_[0] : vec2[0];
		res[1] = data_[1] > vec2[1] ? data_[1] : vec2[1];
		res[2] = data_[2] > vec2[2] ? data_[2] : vec2[2];
		return res;
	}

	//make * operator commutative
	template <typename S, typename T>
	DYN_FUNC const Vector<T, 3> operator *(S scale, const Vector<T, 3> &vec)
	{
		return vec * (T)scale;
	}

}
