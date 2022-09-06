#pragma once
#include "Platform.h"

namespace dyno {

	/*
	 * Interval class is defined for C++ floating-point types.
	 */

	template <typename Real>
	class Interval
	{
	public:
		DYN_FUNC Interval();
		DYN_FUNC Interval(Real min_val, Real max_val, bool lOpen = false, bool rOpen = false);
		DYN_FUNC Interval(const Interval<Real> &interval);
		DYN_FUNC Interval<Real>& operator= (const Interval<Real> &interval);
		DYN_FUNC bool operator== (const Interval<Real> &interval);
		DYN_FUNC bool operator!= (const Interval<Real> &interval);
		DYN_FUNC ~Interval();

		DYN_FUNC Real size() const;

		inline DYN_FUNC Real leftLimit() const { return v0; }
		inline DYN_FUNC Real rightLimit() const { return v1; }

		DYN_FUNC bool isLeftOpen() const;
		DYN_FUNC bool isRightOpen() const;

		DYN_FUNC void setLeftLimit(Real val, bool bOpen = false);
		DYN_FUNC void setRightLimit(Real val, bool bOpen = false);

		DYN_FUNC bool inside(Real val) const;
		DYN_FUNC bool outside(Real val) const;

		DYN_FUNC Interval<Real> intersect(const Interval<Real>& itv) const;

		DYN_FUNC bool isEmpty() const;

		DYN_FUNC static Interval<Real> unitInterval(); //[0,1]
	private:
		Real v0, v1;
		bool leftOpen, rightOpen;
	};

}  //end of namespace dyno

#include "Interval.inl"