#include "Math/SimpleMath.h"

namespace dyno
{
	template<typename Integer>
	DYN_FUNC Fraction<Integer, 3>::Fraction() :
		mDenominator(1)
	{
		mNumerator = Vector<Integer, 3>(0);
	}

	template <typename Integer>
	DYN_FUNC Fraction<Integer, 3>::Fraction(Integer num0, Integer num1, Integer num2)
	{
		mNumerator = Vector<Integer, 3>(num0, num1, num2);
		mDenominator = Integer(1);
	}

	template <typename Integer>
	DYN_FUNC Fraction<Integer, 3>::Fraction(Vector<Integer, 3> numerator, Integer denominator)
	{
		mNumerator = denominator > Integer(0) ? numerator : -numerator;
		mDenominator = denominator > Integer(0) ? denominator : -denominator;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 3>::Fraction(Integer num0, Integer num1, Integer num2, Integer denominator)
	{
		mNumerator = denominator > Integer(0) ? Vector<Integer, 3>(num0, num1, num2) : Vector<Integer, 3>(-num0, -num1, -num2);
		mDenominator = denominator > Integer(0) ? denominator : -denominator;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 3>& Fraction<Integer, 3>::operator=(const Fraction<Integer, 3>& other)
	{
		mNumerator = other.mNumerator;
		mDenominator = other.mDenominator;

		return *this;
	}

	template <typename Integer>
	DYN_FUNC bool Fraction<Integer, 3>::operator== (const Fraction<Integer, 3>& other) const
	{
		return mNumerator * other.mDenominator() == other.mNumerator * mDenominator;
	}

	template <typename Integer>
	DYN_FUNC bool Fraction<Integer, 3>::operator!= (const Fraction<Integer, 3>& other) const
	{
		return !((*this) == other);
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 3> Fraction<Integer, 3>::operator+(const Integer& v) const
	{
		Fraction<Integer, 3> addition;
		addition.mNumerator = mNumerator + v * Vector<Integer, 3>(mDenominator);
		addition.mDenominator = mDenominator;

		return addition;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 3> Fraction<Integer, 3>::operator-(const Integer& v) const
	{
		Fraction<Integer, 3> sub;
		sub.mNumerator = mNumerator - v * Vector<Integer, 3>(mDenominator);
		sub.mDenominator = mDenominator;

		return sub;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 3> Fraction<Integer, 3>::operator*(const Integer& v) const
	{
		return Fraction<Integer, 3>(v * mNumerator, mDenominator);
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 3> Fraction<Integer, 3>::operator/(const Integer& v) const
	{
		return Fraction<Integer, 3>(mNumerator, v * mDenominator);
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 3>& Fraction<Integer, 3>::operator+=(const Integer& v)
	{
		mNumerator += v * mDenominator;
		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 3>& Fraction<Integer, 3>::operator-=(const Integer& v)
	{
		mNumerator -= v * mDenominator;
		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 3>& Fraction<Integer, 3>::operator*=(const Integer& v)
	{
		mNumerator *= v;
		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 3>& Fraction<Integer, 3>::operator/=(const Integer& v)
	{
		mDenominator *= v;
		return *this;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 3> Fraction<Integer, 3>::operator-(void) const
	{
		Fraction<Integer, 3> neg;
		neg.mNumerator = -mNumerator;
		neg.mDenominator = mDenominator;

		return neg;
	}

	template <typename Integer>
	DYN_FUNC Fraction<Integer, 1> Fraction<Integer, 3>::dot(const Fraction<Integer, 3>& other) const
	{
		return Fraction<Integer, 1>(mNumerator.dot(other.numerator()), mDenominator * other.mDenominator);
	}
}

