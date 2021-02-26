#include <limits>
#include <glm/gtx/norm.hpp>

namespace dyno 
{
	template <typename T>
	DYN_FUNC Vector<T, 4>::Vector()
		:Vector(0) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>::Vector(T x)
		: Vector(x, x, x, x) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>::Vector(T x, T y, T z, T w)
		: data_(x, y, z, w)
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>::Vector(const Vector<T, 4>& vec2)
		: data_(vec2.data_)
	{

	}

	template <typename T>
	DYN_FUNC Vector<T, 4>::~Vector()
	{

	}

	template <typename T>
	DYN_FUNC T& Vector<T, 4>::operator[] (unsigned int idx)
	{
		return const_cast<T &> (static_cast<const Vector<T, 4> &>(*this)[idx]);
	}

	template <typename T>
	DYN_FUNC const T& Vector<T, 4>::operator[] (unsigned int idx) const
	{
		// #ifndef __CUDA_ARCH__
		//     if(idx>=4)
		//         throw PhysikaException("Vector index out of range!");
		// #endif
		return data_[idx];
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator+ (const Vector<T, 4> &vec2) const
	{
		return Vector<T, 4>(*this) += vec2;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator+= (const Vector<T, 4> &vec2)
	{
		data_ += vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator- (const Vector<T, 4> &vec2) const
	{
		return Vector<T, 4>(*this) -= vec2;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator-= (const Vector<T, 4> &vec2)
	{
		data_ -= vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator*(const Vector<T, 4> &vec2) const
	{
		return Vector<T, 4>(data_[0] * vec2.data_[0], data_[1] * vec2.data_[1], data_[2] * vec2.data_[2], data_[3] * vec2.data_[3]);
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator*=(const Vector<T, 4> &vec2)
	{
		data_[0] *= vec2.data_[0];	data_[1] *= vec2.data_[1];	data_[2] *= vec2.data_[2];	data_[3] *= vec2.data_[3];
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator/(const Vector<T, 4> &vec2) const
	{
		return Vector<T, 4>(data_[0] / vec2[0], data_[1] / vec2[1], data_[2] / vec2[2], data_[3] / vec2[3]);
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator/=(const Vector<T, 4> &vec2)
	{
		data_[0] /= vec2.data_[0];	data_[1] /= vec2.data_[1];	data_[2] /= vec2.data_[2];	data_[3] /= vec2.data_[3];
		return *this;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator=(const Vector<T, 4> & vec2)
	{
		data_ = vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC bool Vector<T, 4>::operator== (const Vector<T, 4> &vec2) const
	{
		return data_ == vec2.data_;
	}

	template <typename T>
	DYN_FUNC bool Vector<T, 4>::operator!= (const Vector<T, 4> &vec2) const
	{
		return !((*this) == vec2);
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator+(T value) const
	{
		return Vector<T, 4>(*this) += value;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator+= (T value)
	{
		data_ += value;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator-(T value) const
	{
		return Vector<T, 4>(*this) -= value;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator-= (T value)
	{
		data_ -= value;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator* (T scale) const
	{
		return Vector<T, 4>(*this) *= scale;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator*= (T scale)
	{
		data_ *= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator/ (T scale) const
	{
		return Vector<T, 4>(*this) /= scale;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator/= (T scale)
	{
		data_ /= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator-(void) const
	{
		Vector<T, 4> res;
		res.data_ = -data_;
		return res;
	}

	template <typename T>
	DYN_FUNC T Vector<T, 4>::norm() const
	{
		return glm::length(data_);
	}

	template <typename T>
	DYN_FUNC T Vector<T, 4>::normSquared() const
	{
		return glm::length2(data_);
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::normalize()
	{
		data_ = glm::length(data_) > glm::epsilon<T>() ? glm::normalize(data_) : glm::tvec4<T>(0, 0, 0, 0);
		return *this;
	}

	template <typename T>
	DYN_FUNC T Vector<T, 4>::dot(const Vector<T, 4>& vec2) const
	{
		return glm::dot(data_, vec2.data_);
	}

	template <typename T>
	DYN_FUNC Vector<T, 4> Vector<T, 4>::minimum(const Vector<T, 4>& vec2) const
	{
		Vector<T, 4> res;
		res[0] = data_[0] < vec2[0] ? data_[0] : vec2[0];
		res[1] = data_[1] < vec2[1] ? data_[1] : vec2[1];
		res[2] = data_[2] < vec2[2] ? data_[2] : vec2[2];
		res[3] = data_[3] < vec2[3] ? data_[3] : vec2[3];
		return res;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4> Vector<T, 4>::maximum(const Vector<T, 4>& vec2) const
	{
		Vector<T, 4> res;
		res[0] = data_[0] > vec2[0] ? data_[0] : vec2[0];
		res[1] = data_[1] > vec2[1] ? data_[1] : vec2[1];
		res[2] = data_[2] > vec2[2] ? data_[2] : vec2[2];
		res[3] = data_[3] > vec2[3] ? data_[3] : vec2[3];
		return res;
	}

	template <typename S, typename T>
	DYN_FUNC  const Vector<T, 4> operator *(S scale, const Vector<T, 4> &vec)
	{
		return vec * (T)scale;
	}
}
