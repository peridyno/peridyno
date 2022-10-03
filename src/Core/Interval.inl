#include "Interval.h"
#include "Algorithm/SimpleMath.h"

namespace dyno {

	template <typename Real>
	DYN_FUNC Interval<Real>::Interval() :
		v0(0),
		v1(0),
		leftOpen(true),
		rightOpen(true)
	{
	}

	template <typename Real>
	DYN_FUNC Interval<Real>::Interval(Real min_val, Real max_val, bool lOpen, bool rOpen) :
		v0(min_val),
		v1(max_val),
		leftOpen(lOpen),
		rightOpen(rOpen)
	{
	}

	template <typename Real>
	DYN_FUNC Interval<Real>::Interval(const Interval<Real> &interval) :
		v0(interval.v0),
		v1(interval.v1),
		leftOpen(interval.leftOpen),
		rightOpen(interval.rightOpen)
		
	{
	}

	template <typename Real>
	DYN_FUNC Interval<Real>& Interval<Real>::operator= (const Interval<Real> &interval)
	{
		v0 = interval.v0;
		v1 = interval.v1;
		leftOpen = interval.leftOpen;
		rightOpen = interval.rightOpen;
		return *this;
	}

	template <typename Real>
	DYN_FUNC bool Interval<Real>::operator==(const Interval<Real> &interval)
	{
		return (v0 == interval.v0) 
			&& (v1 == interval.v0)
			&& (leftOpen == interval.leftOpen)
			&& (rightOpen == interval.rightOpen);
	}

	template <typename Real>
	DYN_FUNC bool Interval<Real>::operator!=(const Interval<Real> &interval)
	{
		return !((*this) == interval);
	}

	template <typename Real>
	DYN_FUNC Interval<Real>::~Interval()
	{
	}

	template <typename Real>
	DYN_FUNC Real Interval<Real>::size() const
	{
		return (v1 - v0);
	}

	template <typename Real>
	DYN_FUNC bool Interval<Real>::isLeftOpen() const
	{
		return leftOpen;
	}

	template <typename Real>
	DYN_FUNC bool Interval<Real>::isRightOpen() const
	{
		return rightOpen;
	}

	template <typename Real>
	DYN_FUNC void Interval<Real>::setLeftLimit(Real val, bool bOpen)
	{
		v0 = val;
		leftOpen = bOpen;
	}

	template <typename Real>
	DYN_FUNC void Interval<Real>::setRightLimit(Real val, bool bOpen)
	{
		v1 = val;
		rightOpen = bOpen;
	}

	template <typename Real>
	DYN_FUNC bool Interval<Real>::inside(Real val) const
	{
		if (isEmpty())
		{
			return false;
		}

		if (val > v0 && val < v1)
			return true;
		else if ((val == v0 && leftOpen == false) || (val == v1 && rightOpen == false))
			return true;
		else
			return false;
	}

	template <typename Real>
	DYN_FUNC bool Interval<Real>::outside(Real val) const
	{
		return !inside(val);
	}

	template <typename Real>
	DYN_FUNC Interval<Real> Interval<Real>::intersect(const Interval<Real>& itv) const
	{
		Interval<Real> ret;

		ret.v0 = maximum(v0, itv.v0);
		ret.v1 = minimum(v1, itv.v1);
		ret.leftOpen = outside(ret.v0)||itv.outside(ret.v0);
		ret.rightOpen = outside(ret.v1) || itv.outside(ret.v1);

		return ret;
	}

	template <typename Real>
	DYN_FUNC bool Interval<Real>::isEmpty() const
	{
		return v0 > v1 || (v0 == v1 && (leftOpen == true || rightOpen == true));
	}

	template <typename Real>
	DYN_FUNC Interval<Real> Interval<Real>::unitInterval()
	{
		return Interval<Real>(0, 1, false, false);
	}
} //end of namespace dyno
