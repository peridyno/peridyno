/**
 * Copyright 2025 Xiaowei He
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

namespace dyno
{
	/**
	 * @brief A representation of fraction combining a numerator and a denominator, i.e., numerator / denominator
	 *			denominator is guaranteed to be non-negative
	 */
	template <typename Integer>
	class Fraction
	{
	public:
		DYN_FUNC Fraction();
		DYN_FUNC Fraction(Integer numerator);
		DYN_FUNC explicit Fraction(Integer numerator, Integer denominator);

		DYN_FUNC Integer numerator() const { return mNumerator; }
		DYN_FUNC Integer denominator() const { return mDenominator; }

		DYN_FUNC Integer& numerator() { return mNumerator; }
		DYN_FUNC Integer& denominator() { return mDenominator; }

		DYN_FUNC const Fraction<Integer> operator+ (const Fraction<Integer>&other) const;
		DYN_FUNC const Fraction<Integer> operator- (const Fraction<Integer>&other) const;
		DYN_FUNC const Fraction<Integer> operator* (const Fraction<Integer>&other) const;
		DYN_FUNC const Fraction<Integer> operator/ (const Fraction<Integer>&other) const;

		DYN_FUNC Fraction<Integer>& operator+= (const Fraction<Integer>&other);
		DYN_FUNC Fraction<Integer>& operator-= (const Fraction<Integer>&other);
		DYN_FUNC Fraction<Integer>& operator*= (const Fraction<Integer>&other);
		DYN_FUNC Fraction<Integer>& operator/= (const Fraction<Integer>&other);

		DYN_FUNC Fraction<Integer>& operator= (const Fraction<Integer>&other);

		DYN_FUNC bool operator== (const Fraction<Integer>&other) const;
		DYN_FUNC bool operator!= (const Fraction<Integer>&other) const;
		DYN_FUNC bool operator> (const Fraction<Integer>& other) const;
		DYN_FUNC bool operator< (const Fraction<Integer>& other) const;
		DYN_FUNC bool operator>= (const Fraction<Integer>& other) const;
		DYN_FUNC bool operator<= (const Fraction<Integer>& other) const;

		DYN_FUNC const Fraction<Integer> operator+ (const Integer& v) const;
		DYN_FUNC const Fraction<Integer> operator- (const Integer& v) const;
		DYN_FUNC const Fraction<Integer> operator* (const Integer& v) const;
		DYN_FUNC const Fraction<Integer> operator/ (const Integer& v) const;

		DYN_FUNC Fraction<Integer>& operator+= (const Integer& v);
		DYN_FUNC Fraction<Integer>& operator-= (const Integer& v);
		DYN_FUNC Fraction<Integer>& operator*= (const Integer& v);
		DYN_FUNC Fraction<Integer>& operator/= (const Integer& v);

		DYN_FUNC const Fraction<Integer> operator - (void) const;

		DYN_FUNC const bool isValid() { return mDenominator != 0; }

	protected:
		Integer mNumerator;
		Integer mDenominator;
	};
}

#include "Fraction.inl"
