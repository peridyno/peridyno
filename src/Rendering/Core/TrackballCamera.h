/**
 * Copyright 2017-2021 Xiaowei HE
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
#include "Camera.h"

namespace dyno
{
	class TrackballCamera : public Camera {

	public:
		TrackballCamera();
		~TrackballCamera() {};

		void reset();

		void registerPoint(float x, float y) override;

		void rotateToPoint(float x, float y) override;
		void translateToPoint(float x, float y) override;
		void zoomToPoint(float x, float y) override;
		void zoom(float amount) override;

		//TODO: implement
		void setEyePos(const Vec3f& p) override { mCameraPos = p; };

		//TODO: implement
		void setTargetPos(const Vec3f& p) override { mCameraTarget = p; };

		//TODO: implement
		Vec3f getEyePos() const override { return mCameraPos; };

		//TODO: implement
		Vec3f getTargetPos() const override { return mCameraTarget;};

		glm::mat4 getViewMat() override;
		glm::mat4 getProjMat() override;

	public:
		float mRegX;
		float mRegY;

		Vec3f mCameraPos;
		Vec3f mCameraTarget;
		Vec3f mCameraUp;
	};

}

