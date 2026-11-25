/**
 * Copyright 2026 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "Platform.h"

#include "FractionBase.h"

namespace dyno
{
	/**
	 * @brief A representation of fraction combining two numerator and a denominator, i.e., (numerator0 / denominator, numerator1 / denominator)
	 *			denominator is guaranteed to be non-negative
	 */
	template <typename Integer>
	class Fraction<Integer, 2>
	{
	public:
		DYN_FUNC Fraction();
		DYN_FUNC Fraction(Integer num0, Integer num1);
		DYN_FUNC Fraction(Vector<Integer, 2> numerator, Integer denominator);
		DYN_FUNC explicit Fraction(Integer num0, Integer num1, Integer denominator);

		DYN_FUNC Integer numerator(int i) const { return mNumerator[i]; }
		DYN_FUNC Vector<Integer, 2> numerator() const { return mNumerator; }
		DYN_FUNC Integer denominator() const { return mDenominator; }

		DYN_FUNC Integer& numerator(int i) { return mNumerator[i]; }
		DYN_FUNC Vector<Integer, 2>& numerator() { return mNumerator; }
		DYN_FUNC Integer& denominator() { return mDenominator; }

		DYN_FUNC Fraction<Integer, 2>& operator= (const Fraction<Integer, 2>& other);

		DYN_FUNC bool operator== (const Fraction<Integer, 2>&) const;
		DYN_FUNC bool operator!= (const Fraction<Integer, 2>&) const;

		DYN_FUNC const Fraction<Integer, 2> operator+ (const Integer& v) const;
		DYN_FUNC const Fraction<Integer, 2> operator- (const Integer& v) const;
		DYN_FUNC const Fraction<Integer, 2> operator* (const Integer& v) const;
		DYN_FUNC const Fraction<Integer, 2> operator/ (const Integer& v) const;

		DYN_FUNC Fraction<Integer, 2>& operator+= (const Integer& v);
		DYN_FUNC Fraction<Integer, 2>& operator-= (const Integer& v);
		DYN_FUNC Fraction<Integer, 2>& operator*= (const Integer& v);
		DYN_FUNC Fraction<Integer, 2>& operator/= (const Integer& v);

		DYN_FUNC Fraction<Integer, 1> dot(const Fraction<Integer, 2>& other) const;

		DYN_FUNC const Fraction<Integer, 2> operator - (void) const;

		DYN_FUNC const bool isValid() { return mDenominator != 0; }

	protected:
		Vector<Integer, 2> mNumerator;
		Integer mDenominator;
	};
}

#include "Fraction2D.inl"
