#include "Math/SimpleMath.h"

namespace dyno
{
	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>::Fraction() :
		mNumerator(0),
		mDenominator(1)
	{
	}

	template <typename Integer>
	DYN_FUNC Fraction<Integer, 1>::Fraction(Integer numerator)
	{
		mNumerator = numerator;
		mDenominator = 1;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>::Fraction(Integer numerator, Integer denominator)
	{
		Integer divisor = gcd(abs(numerator), abs(denominator));

		Integer sign = denominator >= 0 ? 1 : -1;

		mNumerator = sign * numerator / divisor;
		mDenominator = sign * denominator / divisor;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 1> Fraction<Integer, 1>::operator+ (const Fraction<Integer, 1>& other) const
	{
		Integer numer = mNumerator * other.mDenominator + mDenominator * other.mNumerator;
		Integer denom = mDenominator * other.mDenominator;

		return Fraction<Integer, 1>(numer, denom);
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 1> Fraction<Integer, 1>::operator- (const Fraction<Integer, 1>& other) const
	{
		Integer numer = mNumerator * other.mDenominator - mDenominator * other.mNumerator;
		Integer denom = mDenominator * other.mDenominator;

		return Fraction<Integer, 1>(numer, denom);
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 1> Fraction<Integer, 1>::operator* (const Fraction<Integer, 1>& other) const
	{
		return Fraction<Integer, 1>(mNumerator * other.mNumerator, mDenominator * other.mDenominator);
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 1> Fraction<Integer, 1>::operator/(const Fraction<Integer, 1>& other) const
	{
		return Fraction<Integer, 1>(mNumerator * other.mDenominator, mDenominator * other.mNumerator);
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>& Fraction<Integer, 1>::operator+=(const Fraction<Integer, 1>& other)
	{
		Integer numer = mNumerator * other.mDenominator + mDenominator * other.mNumerator;
		Integer denom = mDenominator * other.mDenominator;

		Integer divisor = gcd(abs(numer), abs(denom));

		Integer sign = denom >= 0 ? 1 : -1;

		mNumerator = sign * numer / divisor;
		mDenominator = sign * denom / divisor;

		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>& Fraction<Integer, 1>::operator-=(const Fraction<Integer, 1>& other)
	{
		Integer numer = mNumerator * other.mDenominator - mDenominator * other.mNumerator;
		Integer denom = mDenominator * other.mDenominator;

		Integer divisor = gcd(abs(numer), abs(denom));

		Integer sign = denom >= 0 ? 1 : -1;

		mNumerator = sign * numer / divisor;
		mDenominator = sign * denom / divisor;

		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>& Fraction<Integer, 1>::operator*=(const Fraction<Integer, 1>& other)
	{
		Integer numer = mNumerator * other.mNumerator;
		Integer denom = mDenominator * other.mDenominator;

		Integer divisor = gcd(abs(numer), abs(denom));

		Integer sign = denom >= 0 ? 1 : -1;

		mNumerator = sign * numer / divisor;
		mDenominator = sign * denom / divisor;

		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>& Fraction<Integer, 1>::operator/=(const Fraction<Integer, 1>& other)
	{
		Integer numer = mNumerator * other.mDenominator;
		Integer denom = mDenominator * other.mNumerator;

		Integer divisor = gcd(abs(numer), abs(denom));

		Integer sign = denom >= 0 ? 1 : -1;

		mNumerator = sign * numer / divisor;
		mDenominator = sign * denom / divisor;

		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>& Fraction<Integer, 1>::operator=(const Fraction<Integer, 1>& other)
	{
		mNumerator = other.mNumerator;
		mDenominator = other.mDenominator;

		return *this;
	}

	template<typename Integer>
	DYN_FUNC bool Fraction<Integer, 1>::operator==(const Fraction<Integer, 1>& other) const
	{
		return mNumerator * other.mDenominator == mDenominator * other.mNumerator;
	}

	template<typename Integer>
	DYN_FUNC bool Fraction<Integer, 1>::operator!=(const Fraction<Integer, 1>& other) const
	{
		return !(*this == other);
	}

	template <typename Integer>
	DYN_FUNC bool Fraction<Integer, 1>::operator>(const Fraction<Integer, 1>& other) const
	{
		return mNumerator * other.mDenominator > mDenominator * other.mNumerator;
	}

	template <typename Integer>
	DYN_FUNC bool Fraction<Integer, 1>::operator<(const Fraction<Integer, 1>& other) const
	{
		return mNumerator * other.mDenominator < mDenominator * other.mNumerator;
	}

	template <typename Integer>
	DYN_FUNC bool Fraction<Integer, 1>::operator>=(const Fraction<Integer, 1>& other) const
	{
		return mNumerator * other.mDenominator >= mDenominator * other.mNumerator;
	}

	template <typename Integer>
	DYN_FUNC bool Fraction<Integer, 1>::operator<=(const Fraction<Integer, 1>& other) const
	{
		return mNumerator * other.mDenominator <= mDenominator * other.mNumerator;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 1> Fraction<Integer, 1>::operator+(const Integer& v) const
	{
		Fraction<Integer, 1> addition;
		addition.mNumerator = mNumerator + v * mDenominator;
		addition.mDenominator = mDenominator;
		return addition;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 1> Fraction<Integer, 1>::operator-(const Integer& v) const
	{
		Fraction<Integer, 1> sub;
		sub.mNumerator = mNumerator - v * mDenominator;
		sub.mDenominator = mDenominator;
		return sub;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 1> Fraction<Integer, 1>::operator*(const Integer& v) const
	{
		return Fraction<Integer, 1>(v * mNumerator, mDenominator);
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 1> Fraction<Integer, 1>::operator/(const Integer& v) const
	{
		return Fraction<Integer, 1>(mNumerator, v * mDenominator);
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>& Fraction<Integer, 1>::operator+=(const Integer& v)
	{
		mNumerator += v * mDenominator;
		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>& Fraction<Integer, 1>::operator-=(const Integer& v)
	{
		mNumerator -= v * mDenominator;
		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>& Fraction<Integer, 1>::operator*=(const Integer& v)
	{
		(*this) = Fraction<Integer, 1>(v * mNumerator, mDenominator);

		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer, 1>& Fraction<Integer, 1>::operator/=(const Integer& v)
	{
		(*this) = Fraction<Integer, 1>(mNumerator, v * mDenominator);

		return *this;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer, 1> Fraction<Integer, 1>::operator-(void) const
	{
		Fraction<Integer, 1> neg;
		neg.mNumerator = -mNumerator;
		neg.mDenominator = mDenominator;
		return neg;
	}
}

