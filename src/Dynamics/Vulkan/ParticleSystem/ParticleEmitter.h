/**
 * Copyright 2023 Xiaowei He
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
#include "Node/ParametricModel.h"

namespace dyno
{
	/*!
	*	\class	ParticleEimitter
	*	\brief	
	*/
	class ParticleEmitter : public ParametricModel<DataType3f>
	{
	public:
		ParticleEmitter();
		virtual ~ParticleEmitter();

		uint sizeOfParticles() { return mPosition.size(); }

		DArray<Vec3f>& getPositions() { return mPosition; }
		DArray<Vec3f>& getVelocities() { return mVelocity; }

		std::string getNodeType() override;

	public:
		DEF_VAR(float, VelocityMagnitude, 1.0f, "Emitter Velocity");
		DEF_VAR(float, SamplingDistance, 0.005f, "Emitter Sampling Distance");

	protected:
		void updateStates() final;
		
		virtual void generateParticles();

		inline SquareMatrix<float, 3> rotationMatrix()
		{
			auto center = this->varLocation()->getData();
			auto rot_vec = this->varRotation()->getData();

			Quat<float> quat = Quat<float>::identity();
			float x_rad = rot_vec[0] / 180.0f * M_PI;
			float y_rad = rot_vec[1] / 180.0f * M_PI;
			float z_rad = rot_vec[2] / 180.0f * M_PI;

			quat = quat * Quat<float>(x_rad, Vec3f(1, 0, 0));
			quat = quat * Quat<float>(y_rad, Vec3f(0, 1, 0));
			quat = quat * Quat<float>(z_rad, Vec3f(0, 0, 1));

			return quat.toMatrix3x3();
		}

	protected:
		DArray<Vec3f> mPosition;
		DArray<Vec3f> mVelocity;
	};
}