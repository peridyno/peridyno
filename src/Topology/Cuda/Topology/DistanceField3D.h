/**
 * @file DistanceField3D.h
 * @author Xiaowei He (xiaowei@iscas.ac.cn)
 * @brief GPU supported signed distance field
 * @version 0.1
 * @date 2019-05-31
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once

#include <string>
#include "Platform.h"
#include "Array/Array3D.h"

namespace dyno {

#define FARWAY_DISTANCE 10^6

	template<typename TDataType>
	class DistanceField3D {
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		DistanceField3D();

		DistanceField3D(std::string filename);

		/*!
		*	\brief	Should not release data here, call release() explicitly.
		*/
		~DistanceField3D();

		/**
		 * @brief Release m_distance
		 * Should be explicitly called before destruction to avoid GPU memory leak.
		 */
		void release();

		/**
		 * @brief Translate the distance field with a displacement
		 * 
		 * @param t displacement
		 */
		void translate(const Coord& t);

		/**
		 * @brief Scale the distance field
		 * 
		 * @param s scaling factor
		 */
		void scale(const Real s);

		/**
		 * @brief Query the signed distance for p
		 * 
		 * @param p position
		 * @param d return the signed distance at position p
		 * @param normal return the normal at position p
		 */
		GPU_FUNC void getDistance(const Coord &p, Real &d, Coord &normal);

		DYN_FUNC uint nx() { return mDistances.nx(); }

		DYN_FUNC uint ny() { return mDistances.ny(); }

		DYN_FUNC uint nz() { return mDistances.nz(); }

	public:
		/**
		 * @brief load signed distance field from a file
		 * 
		 * @param filename 
		 * @param inverted indicated whether the signed distance field should be inverted after initialization
		 */
		void loadSDF(std::string filename, bool inverted = false);

		/**
		 * @brief load signed distance field from a Box (lo, hi)
		 * 
		 * @param inverted indicated whether the signed distance field should be positive in outside. default: +[---]+
		 */
		void loadBox(Coord& lo, Coord& hi, bool inverted = false);

		void loadCylinder(Coord& center, Real radius, Real height, int axis, bool inverted = false);

		void loadSphere(Coord& center, Real radius, bool inverted = false);

		void setSpace(const Coord p0, const Coord p1, Real h);

		inline Coord lowerBound() { return mOrigin; }

		inline Coord upperBound() { return Coord(mOrigin[0] + (mDistances.nx() - 1) * mH, mOrigin[1] + (mDistances.ny() - 1) * mH, mOrigin[2] + (mDistances.nz() - 1) * mH); }


		void assign(DistanceField3D<TDataType>& sdf) {
			mOrigin = sdf.mOrigin;
			mH = sdf.mH;
			mInverted = sdf.mInverted;
			mDistances.assign(sdf.mDistances);
		}

		DArray3D<Real>& distances() { return mDistances; }

		void setDistance(CArray3D<Real> distance) {
			mDistances.assign(distance);
		}

		Real getGridSpacing() { return mH; }

		/**
		 * @brief Invert the signed distance field
		 *
		 */
		void invertSDF();
		
	private:
		GPU_FUNC inline Real lerp(Real a, Real b, Real alpha) const {
			return (1.0f - alpha)*a + alpha *b;
		}

		/**
		 * @brief Lower left corner
		 * 
		 */
		Coord mOrigin;

		/**
		 * @brief grid spacing
		 * 
		 */
		Real mH;

		bool mInverted = false;

		/**
		 * @brief Storing the signed distance field as a 3D array.
		 * 
		 */
		DArray3D<Real> mDistances;
	};

	template<typename TDataType>
	GPU_FUNC void DistanceField3D<TDataType>::getDistance(const Coord &p, Real &d, Coord &normal)
	{
		// get cell and lerp values
		Coord fp = (p - mOrigin) / mH;
		const int i = (int)floor(fp[0]);
		const int j = (int)floor(fp[1]);
		const int k = (int)floor(fp[2]);
		if (i < 0 || i >= mDistances.nx() - 1 || j < 0 || j >= mDistances.ny() - 1 || k < 0 || k >= mDistances.nz() - 1) {
			if (mInverted) d = -FARWAY_DISTANCE;
			else d = FARWAY_DISTANCE;
			normal = Coord(0);
			return;
		}
		Coord ip = Coord(i, j, k);

		Coord alphav = fp - ip;
		Real alpha = alphav[0];
		Real beta = alphav[1];
		Real gamma = alphav[2];

		Real d000 = mDistances(i, j, k);
		Real d100 = mDistances(i + 1, j, k);
		Real d010 = mDistances(i, j + 1, k);
		Real d110 = mDistances(i + 1, j + 1, k);
		Real d001 = mDistances(i, j, k + 1);
		Real d101 = mDistances(i + 1, j, k + 1);
		Real d011 = mDistances(i, j + 1, k + 1);
		Real d111 = mDistances(i + 1, j + 1, k + 1);

		Real dx00 = lerp(d000, d100, alpha);
		Real dx10 = lerp(d010, d110, alpha);
		Real dxy0 = lerp(dx00, dx10, beta);

		Real dx01 = lerp(d001, d101, alpha);
		Real dx11 = lerp(d011, d111, alpha);
		Real dxy1 = lerp(dx01, dx11, beta);

		Real d0y0 = lerp(d000, d010, beta);
		Real d0y1 = lerp(d001, d011, beta);
		Real d0yz = lerp(d0y0, d0y1, gamma);

		Real d1y0 = lerp(d100, d110, beta);
		Real d1y1 = lerp(d101, d111, beta);
		Real d1yz = lerp(d1y0, d1y1, gamma);

		Real dx0z = lerp(dx00, dx01, gamma);
		Real dx1z = lerp(dx10, dx11, gamma);

		normal[0] = d0yz - d1yz;
		normal[1] = dx0z - dx1z;
		normal[2] = dxy0 - dxy1;

		Real l = normal.norm();
		if (l < 0.0001f) normal = Coord(0);
		else normal = normal.normalize();

		d = (1.0f - gamma) * dxy0 + gamma * dxy1;
	}
}
