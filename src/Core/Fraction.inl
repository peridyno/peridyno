#include "Math/SimpleMath.h"

namespace dyno
{
	template<typename Integer>
	DYN_FUNC Fraction<Integer>::Fraction() :
		mNumerator(0),
		mDenominator(1)
	{
	}

	template <typename Integer>
	DYN_FUNC Fraction<Integer>::Fraction(Integer numerator)
	{
		mNumerator = numerator;
		mDenominator = 1;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer>::Fraction(Integer numerator, Integer denominator)
	{
		Integer divisor = gcd(abs(numerator), abs(denominator));

		Integer sign = denominator >= 0 ? 1 : -1;

		mNumerator = sign * numerator / divisor;
		mDenominator = sign * denominator / divisor;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer> Fraction<Integer>::operator+ (const Fraction<Integer>& other) const 
	{
		Integer numer = mNumerator * other.mDenominator + mDenominator * other.mNumerator;
		Integer denom = mDenominator * other.mDenominator;

		return Fraction<Integer>(numer, denom);
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer> Fraction<Integer>::operator- (const Fraction<Integer>& other) const
	{
		Integer numer = mNumerator * other.mDenominator - mDenominator * other.mNumerator;
		Integer denom = mDenominator * other.mDenominator;

		return Fraction<Integer>(numer, denom);
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer> Fraction<Integer>::operator* (const Fraction<Integer>& other) const
	{
		return Fraction<Integer>(mNumerator * other.mNumerator, mDenominator * other.mDenominator);
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer> Fraction<Integer>::operator/(const Fraction<Integer>& other) const
	{
		return Fraction<Integer>(mNumerator * other.mDenominator, mDenominator * other.mNumerator);
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer>& Fraction<Integer>::operator+=(const Fraction<Integer>& other)
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
	DYN_FUNC Fraction<Integer>& Fraction<Integer>::operator-=(const Fraction<Integer>& other)
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
	DYN_FUNC Fraction<Integer>& Fraction<Integer>::operator*=(const Fraction<Integer>& other)
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
	DYN_FUNC Fraction<Integer>& Fraction<Integer>::operator/=(const Fraction<Integer>& other)
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
	DYN_FUNC Fraction<Integer>& Fraction<Integer>::operator=(const Fraction<Integer>& other)
	{
		mNumerator = other.mNumerator;
		mDenominator = other.mDenominator;

		return *this;
	}

	template<typename Integer>
	DYN_FUNC bool Fraction<Integer>::operator==(const Fraction<Integer>& other) const
	{
		return mNumerator * other.mDenominator == mDenominator * other.mNumerator;
	}

	template<typename Integer>
	DYN_FUNC bool Fraction<Integer>::operator!=(const Fraction<Integer>& other) const
	{
		return !(*this == other);
	}

	template <typename Integer>
	DYN_FUNC bool Fraction<Integer>::operator>(const Fraction<Integer>& other) const
	{
		return mNumerator * other.mDenominator > mDenominator * other.mNumerator;
	}

	template <typename Integer>
	DYN_FUNC bool Fraction<Integer>::operator<(const Fraction<Integer>& other) const
	{
		return mNumerator * other.mDenominator < mDenominator * other.mNumerator;
	}

	template <typename Integer>
	DYN_FUNC bool Fraction<Integer>::operator>=(const Fraction<Integer>& other) const
	{
		return mNumerator * other.mDenominator >= mDenominator * other.mNumerator;
	}

	template <typename Integer>
	DYN_FUNC bool Fraction<Integer>::operator<=(const Fraction<Integer>& other) const
	{
		return mNumerator * other.mDenominator <= mDenominator * other.mNumerator;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer> Fraction<Integer>::operator+(const Integer& v) const
	{
		Fraction<Integer> addition;
		addition.mNumerator = mNumerator + v * mDenominator;
		addition.mDenominator = mDenominator;
		return addition;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer> Fraction<Integer>::operator-(const Integer& v) const
	{
		Fraction<Integer> sub;
		sub.mNumerator = mNumerator - v * mDenominator;
		sub.mDenominator = mDenominator;
		return sub;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer> Fraction<Integer>::operator*(const Integer& v) const
	{
		return Fraction<Integer>(v * mNumerator, mDenominator);
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer> Fraction<Integer>::operator/(const Integer& v) const
	{
		return Fraction<Integer>(mNumerator, v * mDenominator);
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer>& Fraction<Integer>::operator+=(const Integer& v)
	{
		mNumerator += v * mDenominator;
		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer>& Fraction<Integer>::operator-=(const Integer& v)
	{
		mNumerator -= v * mDenominator;
		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer>& Fraction<Integer>::operator*=(const Integer& v)
	{
		(*this) = Fraction<Integer>(v * mNumerator, mDenominator);

		return *this;
	}

	template<typename Integer>
	DYN_FUNC Fraction<Integer>& Fraction<Integer>::operator/=(const Integer& v)
	{
		(*this) = Fraction<Integer>(mNumerator, v * mDenominator);

		return *this;
	}

	template<typename Integer>
	DYN_FUNC const Fraction<Integer> Fraction<Integer>::operator-(void) const
	{
		Fraction<Integer> neg;
		neg.mNumerator = -mNumerator;
		neg.mDenominator = mDenominator;
		return neg;
	}
}

