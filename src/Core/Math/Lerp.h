/**
 * Copyright 2024 Xiaowei He
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
#include "Vector.h"

#include "Array/Array.h"
#include "Array/Array2D.h"
#include "Array/Array3D.h"


namespace dyno
{
	/**
	 * Linear interpolation functions for Array, Array2D and Array3D 
	 */

	enum LerpMode
	{
		REPEAT,
		CLAMP_TO_BORDER
	};

	template<typename T, DeviceType deviceType>
	DYN_FUNC T lerp(Array<T, deviceType>& array1d, float x, LerpMode mode = LerpMode::REPEAT)
	{
		const uint nx = array1d.size();

		int i0 = (int)floor(x);

		int i1 = i0 + 1;

		const float fx = x - i0;

		i0 = mode == LerpMode::REPEAT ? i0 % nx : (i0 < 0 ? 0 : (i0 >= nx ? nx - 1 : nx));
		i1 = mode == LerpMode::REPEAT ? i1 % nx : (i1 < 0 ? 0 : (i1 >= nx ? nx - 1 : nx));

		const float w0 = (1.0f - fx);
		const float w1 = fx;

		return w0 * array1d[i0] + w1 * array1d[i1];
	};

	template<typename T, DeviceType deviceType>
	DYN_FUNC T bilinear(Array2D<T, deviceType>& array2d, float x, float y, LerpMode mode = LerpMode::REPEAT)
	{
		const uint nx = array2d.nx();
		const uint ny = array2d.ny();

		int i0 = (int)floor(x);
		int j0 = (int)floor(y);

		int i1 = i0 + 1;
		int j1 = j0 + 1;

		const float fx = x - i0;
		const float fy = y - j0;

		i0 = mode == LerpMode::REPEAT ? i0 % nx : (i0 < 0 ? 0 : (i0 >= nx ? nx - 1 : nx));
		j0 = mode == LerpMode::REPEAT ? j0 % ny : (j0 < 0 ? 0 : (j0 >= ny ? ny - 1 : ny));

		i1 = mode == LerpMode::REPEAT ? i1 % nx : (i1 < 0 ? 0 : (i1 >= nx ? nx - 1 : nx));
		j1 = mode == LerpMode::REPEAT ? j1 % ny : (j1 < 0 ? 0 : (j1 >= ny ? ny - 1 : ny));

		const float w00 = (1.0f - fx) * (1.0f - fy);
		const float w10 = fx * (1.0f - fy);
		const float w01 = (1.0f - fx) * fy;
		const float w11 = fx * fy;

		return w00 * array2d(i0, j0) + w01 * array2d(i0, j1) + w10 * array2d(i1, j0) * w11 * array2d(i1, j1);
	};

	template<typename T, DeviceType deviceType>
	DYN_FUNC T trilinear(Array3D<T, deviceType>& array3d, float x, float y, LerpMode mode = LerpMode::REPEAT)
	{
		const uint nx = array3d.nx();
		const uint ny = array3d.ny();
		const uint nz = array3d.nz();

		int i0 = (int)floor(x);
		int j0 = (int)floor(y);
		int k0 = (int)floor(z);

		int i1 = i0 + 1;
		int j1 = j0 + 1;
		int k1 = k0 + 1;

		const float fx = x - i0;
		const float fy = y - j0;
		const float fz = z - k0;

		i0 = mode == LerpMode::REPEAT ? i0 % nx : (i0 < 0 ? 0 : (i0 >= nx ? nx - 1 : nx));
		j0 = mode == LerpMode::REPEAT ? j0 % ny : (j0 < 0 ? 0 : (j0 >= ny ? ny - 1 : ny));
		k0 = mode == LerpMode::REPEAT ? k0 % nz : (k0 < 0 ? 0 : (k0 >= nz ? nz - 1 : nz));

		i1 = mode == LerpMode::REPEAT ? i1 % nx : (i1 < 0 ? 0 : (i1 >= nx ? nx - 1 : nx));
		j1 = mode == LerpMode::REPEAT ? j1 % ny : (j1 < 0 ? 0 : (j1 >= ny ? ny - 1 : ny));
		k1 = mode == LerpMode::REPEAT ? k1 % nz : (k1 < 0 ? 0 : (k1 >= nz ? nz - 1 : nz));

		const float w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
		const float w100 = fx * (1.0f - fy) * (1.0f - fz);
		const float w010 = (1.0f - fx) * fy * (1.0f - fz);
		const float w001 = (1.0f - fx) * (1.0f - fy) * fz;
		const float w111 = fx * fy * fz;
		const float w011 = (1.0f - fx) * fy * fz;
		const float w101 = fx * (1.0f - fy) * fz;
		const float w110 = fx * fy * (1.0f - fz);

		return w000 * array3d(i0, j0, k0) + w100 * array3d(i1, j0, k0) + w010 * array3d(i0, j1, k0) + w001 * array3d(i0, j0, k1)
			+ w111 * array3d(i1, j1, k1) + w011 * array3d(i0, j1, k1) + w101 * array3d(i1, j0, k1) + w110 * array3d(i1, j1, k0);
	};
}