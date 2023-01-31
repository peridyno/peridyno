/**
 * Copyright 2021 Yue Chang
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
#include "Node.h"

namespace dyno
{
	/*!
	*	\class	ParticleEimitter
	*	\brief	
	*/
	template<typename TDataType>
	class ParticleEmitter : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleEmitter(std::string name = "particle emitter");
		virtual ~ParticleEmitter();

		virtual void generateParticles();

		uint sizeOfParticles() { return mPosition.size(); }

		DArray<Coord>& getPositions() { return mPosition; }
		DArray<Coord>& getVelocities() { return mVelocity; }

		std::string getNodeType() override;

	public:
		DEF_VAR(Vec3f, Location, 0, "Node location");
		DEF_VAR(Vec3f, Rotation, 0, "Node rotation");
//		DEF_VAR(Vec3f, Scale, 0, "Node scale");
	
		DEF_VAR(Real, VelocityMagnitude, 1, "Emitter Velocity");
		DEF_VAR(Real, SamplingDistance, 0.005, "Emitter Sampling Distance");
		DEF_VAR(Coord, InitialVelocity, Coord(0, -1, 0), "Initial velocity");

	protected:
		void updateStates() final;

		inline SquareMatrix<Real, 3> rotationMatrix()
		{
			auto center = this->varLocation()->getData();
			auto rot_vec = this->varRotation()->getData();

			Quat<Real> quat = Quat<float>::identity();
			float x_rad = rot_vec[0] / 180.0f * M_PI;
			float y_rad = rot_vec[1] / 180.0f * M_PI;
			float z_rad = rot_vec[2] / 180.0f * M_PI;

			quat = quat * Quat<Real>(x_rad, Coord(1, 0, 0));
			quat = quat * Quat<Real>(y_rad, Coord(0, 1, 0));
			quat = quat * Quat<Real>(z_rad, Coord(0, 0, 1));

			return quat.toMatrix3x3();
		}

	protected:
		DArray<Coord> mPosition;
		DArray<Coord> mVelocity;
	};
}