/**
 * Copyright 2017-2021 Jian SHI
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
#include "GLPhotorealisticRender.h"

namespace dyno
{
	class GLPhotorealisticInstanceRender : public GLPhotorealisticRender
	{
		DECLARE_CLASS(GLPhotorealisticInstanceRender)
	public:
		GLPhotorealisticInstanceRender();
		~GLPhotorealisticInstanceRender() override;

	public:
		virtual std::string caption() override;

		DEF_ARRAYLIST_IN(Transform3f, Transform, DeviceType::GPU, "");

	protected:
		void updateImpl() override;

		void paintGL(const RenderParams& rparams) override;
		void updateGL() override;
		bool initializeGL() override;
		void releaseGL() override;

	private:
		CArray<uint> mOffset;
		CArray<List<Transform3f>> mLists;

		XBuffer<Transform3f> mGLTransform;

	};

};