#include <limits>
#include <glm/gtx/norm.hpp>

namespace dyno
{

	template <typename T>
	DYN_FUNC Vector<T, 2>::Vector()
		:Vector(0) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>::Vector(T x)
		: Vector(x, x) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>::Vector(T x, T y)
		: data_(x, y)
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>::Vector(const Vector<T, 2>& vec)
		: data_(vec.data_)
	{

	}

	template <typename T>
	DYN_FUNC Vector<T, 2>::~Vector()
	{

	}

	template <typename T>
	DYN_FUNC T& Vector<T, 2>::operator[] (unsigned int idx)
	{
		return const_cast<T &>(static_cast<const Vector<T, 2> &>(*this)[idx]);
	}

	template <typename T>
	DYN_FUNC const T& Vector<T, 2>::operator[] (unsigned int idx) const
	{
		return data_[idx];
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> Vector<T, 2>::operator+ (const Vector<T, 2> &vec2) const
	{
		return Vector<T, 2>(*this) += vec2;
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>& Vector<T, 2>::operator+= (const Vector<T, 2> &vec2)
	{
		data_ += vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> Vector<T, 2>::operator- (const Vector<T, 2> &vec2) const
	{
		return Vector<T, 2>(*this) -= vec2;
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>& Vector<T, 2>::operator-= (const Vector<T, 2> &vec2)
	{
		data_ -= vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> Vector<T, 2>::operator*(const Vector<T, 2> &vec2) const
	{
		return Vector<T, 2>(data_[0] * vec2[0], data_[1] * vec2[1]);
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>& Vector<T, 2>::operator*=(const Vector<T, 2> &vec2)
	{
		data_[0] *= vec2.data_[0];	data_[1] *= vec2.data_[1];
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> Vector<T, 2>::operator/(const Vector<T, 2> &vec2) const
	{
		return Vector<T, 2>(data_[0] / vec2[0], data_[1] / vec2[1]);
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>& Vector<T, 2>::operator/=(const Vector<T, 2> &vec2)
	{
		data_[0] /= vec2.data_[0];	data_[1] /= vec2.data_[1];;
		return *this;
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>& Vector<T, 2>::operator=(const Vector<T, 2> &vec2)
	{
		data_ = vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC bool Vector<T, 2>::operator== (const Vector<T, 2> &vec2) const
	{
		return data_ == vec2.data_;
	}

	template <typename T>
	DYN_FUNC bool Vector<T, 2>::operator!= (const Vector<T, 2> &vec2) const
	{
		return !((*this) == vec2);
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> Vector<T, 2>::operator+(T value) const
	{
		return Vector<T, 2>(*this) += value;
	}


	template <typename T>
	DYN_FUNC Vector<T, 2>& Vector<T, 2>::operator+= (T value)
	{
		data_ += value;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> Vector<T, 2>::operator-(T value) const
	{
		return Vector<T, 2>(*this) -= value;
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>& Vector<T, 2>::operator-= (T value)
	{
		data_ -= value;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> Vector<T, 2>::operator* (T scale) const
	{
		return Vector<T, 2>(*this) *= scale;
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>& Vector<T, 2>::operator*= (T scale)
	{
		data_ *= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> Vector<T, 2>::operator/ (T scale) const
	{
		return Vector<T, 2>(*this) /= scale;
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>& Vector<T, 2>::operator/= (T scale)
	{
		data_ /= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> Vector<T, 2>::operator-(void) const
	{
		Vector<T, 2> res;
		res.data_ = -data_;
		return res;
	}

	template <typename T>
	DYN_FUNC T Vector<T, 2>::norm() const
	{
		return glm::length(data_);
	}

	template <typename T>
	DYN_FUNC T Vector<T, 2>::normSquared() const
	{
		return glm::length2(data_);
	}

	template <typename T>
	DYN_FUNC Vector<T, 2>& Vector<T, 2>::normalize()
	{
		data_ = glm::length(data_) > glm::epsilon<T>() ? glm::normalize(data_) : glm::tvec2<T>(0, 0);
		return *this;
	}

	template <typename T>
	DYN_FUNC T Vector<T, 2>::cross(const Vector<T, 2>& vec2) const
	{
		return (*this)[0] * vec2[1] - (*this)[1] * vec2[0];
	}

	template <typename T>
	DYN_FUNC T Vector<T, 2>::dot(const Vector<T, 2>& vec2) const
	{
		return glm::dot(data_, vec2.data_);
	}

	template <typename T>
	DYN_FUNC Vector<T, 2> Vector<T, 2>::minimum(const Vector<T, 2> & vec2) const
	{
		Vector<T, 2> res;
		res[0] = data_[0] < vec2[0] ? data_[0] : vec2[0];
		res[1] = data_[1] < vec2[1] ? data_[1] : vec2[1];
		return res;
	}

	template <typename T>
	DYN_FUNC Vector<T, 2> Vector<T, 2>::maximum(const Vector<T, 2> & vec2) const
	{
		Vector<T, 2> res;
		res[0] = data_[0] > vec2[0] ? data_[0] : vec2[0];
		res[1] = data_[1] > vec2[1] ? data_[1] : vec2[1];
		return res;
	}

	//make * operator commutative
	template <typename S, typename T>
	DYN_FUNC const Vector<T, 2> operator *(S scale, const Vector<T, 2> &vec)
	{
		return vec * (T)scale;
	}

} //end of namespace dyno
