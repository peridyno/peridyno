#pragma once
#include <vector>
#include <memory>
#include <string>
#include "Vector/Vector2D.h"
#include "Vector/Vector3D.h"

#include "Canvas.h"

namespace dyno {


	class Ramp : public Canvas
	{
	public:
		Ramp();
		Ramp(const Ramp& ramp);

		~Ramp() { ; };

		/**
		 * @brief Get the value value"Y" of the curve by value"X" .
		 */
		float getCurveValueByX(float inputX);
		/**
		 * @brief Update the data of the Bezier curve points.
		 */
		void updateBezierCurve()override;

		double calculateLengthForPointSet(std::vector<Coord2D> BezierPtSet);
		/**
		 * @brief Resample Bezier curve.
		 */
		void updateResampleBezierCurve();
		/**
		 * @brief Reordering points on canvas boundaries.
		 */		
		void borderCloseResort();
		/**
		 * @brief Updating the data of a Field
		 */
		void UpdateFieldFinalCoord() override;

		bool isSquard()override { return false; };

	public:


		std::vector<Coord2D> myBezierPoint_H;//

		std::vector<Coord2D> FE_MyCoord;
		std::vector<Coord2D> FE_HandleCoord;


	private:



	};

}

#include "Ramp.inl"