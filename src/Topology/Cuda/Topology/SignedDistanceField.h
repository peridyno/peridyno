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
#include "Module/TopologyModule.h"
#include "DistanceField3D.h"

namespace dyno {

	template<typename TDataType>
	class SignedDistanceField : public TopologyModule {
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;


		DistanceField3D<TDataType>& getSDF() { return distanceField; }

		void setSDF(DistanceField3D<TDataType> dis) { distanceField = dis;}

	private:
		DistanceField3D<TDataType> distanceField;


	};
}
