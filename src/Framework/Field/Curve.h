#pragma once
#include <vector>
#include <memory>
#include <string>
#include "Vector/Vector2D.h"
#include "Vector/Vector3D.h"

#include "Canvas.h"

namespace dyno {

	class Curve :public Canvas
	{
	public:

		Curve();
		Curve(CurveMode mode) {mClose = int(mode);}
		Curve(const Curve& curve);

		~Curve() { };

		/**
		 * @brief Resample Bezier curve.
		 */
		void updateResampleBezierCurve(std::vector<Coord2D>& myBezierPoint_H);
		/**
		 * @brief Updating the data of a Field
		 */
		void UpdateFieldFinalCoord() override;

		bool isSquard()override { return true; };

	private:

	};

}
#include "Curve.inl"
