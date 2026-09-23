/**
 * Copyright 2026 Yuzhong Guo
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

#include "GLDiscreteElementJointVisualModule.h"

#include <glad/glad.h>
#include <math.h>

#include "ShaderStruct.h"
#include "Utility.h"

#include "surface.vert.h"
#include "surface.frag.h"
#include "surface.geom.h"

// HingeJoint
#define HINGE_AXIS_HALF_LEN   Real(0.1)
#define HINGE_AXIS_THICK      Real(0.02)
#define HINGE_PLATE_HALF_LEN  Real(0.04)
#define HINGE_PLATE_HALF_W    Real(0.075)
#define HINGE_PLATE_THICK     Real(0.005)   // Plate thickness (thinner than axis)

// Connector (shared across joint types)
#define CONNECTOR_THICK       Real(0.01)

// SliderJoint
#define SLIDER_RAIL_THICK     Real(0.01)
#define SLIDER_RAIL_HALF_LEN  Real(0.3)
#define SLIDER_BLOCK_SIZE     Real(0.03)
#define SLIDER_MARKER_SIZE    Real(0.02)

// BallAndSocketJoint
#define BALL_SPHERE_RADIUS    Real(0.03)
#define BALL_BOX_ROT_C       Real(0.7071067811865476)
#define BALL_BOX_ROT_S       Real(0.7071067811865476)
#define BALL_PLATE_HALF_SIZE  Real(0.04)   // Square plate half-size

// FixedJoint / PointJoint
#define JOINT_BOX_SIZE        Real(0.02)
#define POINT_SPHERE_RADIUS  Real(0.02)
#define FIXED_FORK_LEN        Real(0.15)

// Global joint scale multipliers
#define JOINT_THICKNESS_SCALE  Real(5.0)
#define JOINT_LENGTH_SCALE    Real(0.3)

#define COLOR_HINGE_AXIS       Vec3f(1.0f, 0.0f, 0.0f)   // Red
#define COLOR_HINGE_CONNECTOR_A Vec3f(0.7f, 0.0f, 0.0f)   // Hinge body1 connector (bright)
#define COLOR_HINGE_CONNECTOR_B Vec3f(0.2f, 0.0f, 0.0f)  // Hinge body2 connector (dark)
#define COLOR_SLIDER           Vec3f(0.0f, 0.2f, 1.0f)   // Blue
#define COLOR_SLIDER_MARKER    Vec3f(0.0f, 0.0f, 0.5f)   // Dark blue
#define COLOR_BALL             Vec3f(0.0f, 1.0f, 0.0f)   // Green (sphere)
#define COLOR_BALL_CONNECTOR_A Vec3f(0.0f, 1.0f, 0.0f)    // Ball body1 connector (bright)
#define COLOR_BALL_CONNECTOR_B Vec3f(0.0f, 0.3f, 0.0f)   // Ball body2 connector (dark)
#define COLOR_BALL_PLATE_A     Vec3f(0.0f, 0.8f, 0.8f)   // Ball body1 plate (cyan)
#define COLOR_BALL_PLATE_B     Vec3f(0.0f, 0.3f, 0.3f)   // Ball body2 plate (dark cyan)
#define COLOR_DISTANCE         Vec3f(1.0f, 1.0f, 0.0f)   // Yellow
#define COLOR_FIXED            Vec3f(0.5f, 0.0f, 0.5f)   // Purple (fork)
#define COLOR_FIXED_CONNECTOR_A Vec3f(0.5f, 0.0f, 0.5f)  // Fixed body1 connector (bright)
#define COLOR_FIXED_CONNECTOR_B Vec3f(0.2f, 0.0f, 0.2f)  // Fixed body2 connector (dark)
#define COLOR_POINT            Vec3f(1.0f, 0.5f, 0.0f)   // Orange

namespace dyno
{
	// ---------------------------------------------------------------------------
	// Helper functions (copied from DiscreteElementsJointToInstance.cu)
	// ---------------------------------------------------------------------------

	template<typename Coord>
	__device__ void computePerpAxes(const Coord& d, Coord& v, Coord& w)
	{
		Coord other;
		auto ax = fabs(d[0]), ay = fabs(d[1]), az = fabs(d[2]);
		if (ay <= ax && ay <= az)
			other = Coord(0, 1, 0);
		else if (ax <= ay && ax <= az)
			other = Coord(1, 0, 0);
		else
			other = Coord(0, 0, 1);

		auto parallel = dot(other, d);
		v = other - d * parallel;
		auto vLen = sqrt(dot(v, v));

		if (vLen < Real(1e-6))
		{
			other = Coord(1, 0, 0);
			parallel = dot(other, d);
			v = other - d * parallel;
			vLen = sqrt(dot(v, v));
		}

		v = v / vLen;

		w = Coord(
			d[1] * v[2] - d[2] * v[1],
			d[2] * v[0] - d[0] * v[2],
			d[0] * v[1] - d[1] * v[0]
		);
	}

	template<typename Coord>
	__device__ void computePerpAxesWithRef(const Coord& d, const Coord& refDir, Coord& v, Coord& w)
	{
		v = Coord(
			d[1] * refDir[2] - d[2] * refDir[1],
			d[2] * refDir[0] - d[0] * refDir[2],
			d[0] * refDir[1] - d[1] * refDir[0]
		);
		auto vLen = sqrt(dot(v, v));

		if (vLen < Real(1e-6))
		{
			computePerpAxes<Coord>(d, v, w);
			return;
		}

		v = v / vLen;

		w = Coord(
			d[1] * v[2] - d[2] * v[1],
			d[2] * v[0] - d[0] * v[2],
			d[0] * v[1] - d[1] * v[0]
		);
	}

	template<typename Coord, typename Real>
	__device__ Transform3f makeBoxTransform(
		const Coord& center, const Coord& u, const Coord& v, const Coord& w,
		Real eu, Real ev, Real ew)
	{
		SquareMatrix<Real, 3> rot;
		rot(0, 0) = u[0]; rot(0, 1) = v[0]; rot(0, 2) = w[0];
		rot(1, 0) = u[1]; rot(1, 1) = v[1]; rot(1, 2) = w[1];
		rot(2, 0) = u[2]; rot(2, 1) = v[2]; rot(2, 2) = w[2];

		Coord scale(2 * eu, 2 * ev, 2 * ew);

		return Transform3f(center, rot, scale);
	}

	template<typename Coord, typename Real>
	__device__ Transform3f makeSphereTransform(
		const Coord& center, Real radius)
	{
		SquareMatrix<Real, 3> rot;
		rot(0, 0) = 1; rot(0, 1) = 0; rot(0, 2) = 0;
		rot(1, 0) = 0; rot(1, 1) = 1; rot(1, 2) = 0;
		rot(2, 0) = 0; rot(2, 1) = 0; rot(2, 2) = 1;

		Coord scale(2 * radius, 2 * radius, 2 * radius);

		return Transform3f(center, rot, scale);
	}

	template<typename Coord, typename Real>
	__device__ Transform3f makeConnectorTransform(
		const Coord& A, const Coord& B, Real thickness)
	{
		Coord dir = B - A;
		auto len = sqrt(dot(dir, dir));
		Coord dirN = len > Real(1e-6) ? dir / len : Coord(1, 0, 0);
		Real halfLen = len > Real(1e-6) ? len * Real(0.5) : Real(0);

		Coord perp1, perp2;
		computePerpAxes<Coord>(dirN, perp1, perp2);

		Coord center = (A + B) * Real(0.5);
		return makeBoxTransform<Coord, Real>(center, dirN, perp1, perp2, halfLen, thickness, thickness);
	}

	template<typename Coord, typename Real>
	__device__ Transform3f makeConnectorTransformWithRef(
		const Coord& A, const Coord& B, Real thickness, const Coord& refAxis)
	{
		Coord dir = B - A;
		auto len = sqrt(dot(dir, dir));
		Coord dirN = len > Real(1e-6) ? dir / len : Coord(1, 0, 0);
		Real halfLen = len > Real(1e-6) ? len * Real(0.5) : Real(0);

		Coord perp1, perp2;
		computePerpAxesWithRef<Coord>(dirN, refAxis, perp1, perp2);

		Coord center = (A + B) * Real(0.5);
		return makeBoxTransform<Coord, Real>(center, dirN, perp1, perp2, halfLen, thickness, thickness);
	}

	// Joint kernels (copied from DiscreteElementsJointToInstance.cu)
	template<typename Coord, typename Matrix, typename Real, typename HingeJoint>
	__global__ void buildHingeJointTransforms(
		DArray<HingeJoint> joints,
		DArray<Matrix> rotMat,
		DArray<Coord> pos,
		DArray<Transform3f> transforms,
		DArray<Vec3f> colors,
		Real length,
		Real thicknessScale,
		uint offset)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= joints.size()) return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		Coord r1 = rotMat[idx1] * joints[tId].r1;
		Coord anchor = pos[idx1] + r1;

		Coord axis = rotMat[idx1] * joints[tId].hingeAxisBody1;
		axis.normalize();

		Coord perp1, perp2;
		computePerpAxes<Coord>(axis, perp1, perp2);

		// 5 instances per hinge joint: axis, 2 connectors, 2 cross plates
		uint base = offset + tId * 5;

		Real axisHalfLen = length * HINGE_AXIS_HALF_LEN * thicknessScale;
		Real axisThick = length * HINGE_AXIS_THICK * thicknessScale;

		// 0: Axis (long thin rod along hinge axis)
		transforms[base + 0] = makeBoxTransform<Coord, Real>(
			anchor, axis, perp1, perp2, axisHalfLen, axisThick, axisThick);
		colors[base + 0] = COLOR_HINGE_AXIS;

		Real connThick = length * CONNECTOR_THICK * thicknessScale;

		// 1: Connector to body1 (亮)
		transforms[base + 1] = makeConnectorTransformWithRef<Coord, Real>(
			anchor, pos[idx1], connThick, axis);
		colors[base + 1] = COLOR_HINGE_CONNECTOR_A;

		// 2: Connector to body2 (暗)
		if (idx2 != INVALID)
		{
			transforms[base + 2] = makeConnectorTransformWithRef<Coord, Real>(
				anchor, pos[idx2], connThick, axis);
		}
		else
		{
			transforms[base + 2] = makeBoxTransform<Coord, Real>(
				anchor, axis, perp1, perp2, Real(0), Real(0), Real(0));
		}
		colors[base + 2] = COLOR_HINGE_CONNECTOR_B;

		// 3 & 4: Two hinge leaves (合页叶片) on each side of the axis.
		// Each leaf extends from the axis along its connector direction,
		// as wide as the axis (along axis), thin in the remaining direction.
		// This forms a hinge/door-knuckle structure to to distinguish hinge joints.
		Real plateHalfLen = length * HINGE_PLATE_HALF_LEN * thicknessScale;
		Real plateHalfW = length * HINGE_PLATE_HALF_W * thicknessScale;
		Real plateThick = length * HINGE_PLATE_THICK * thicknessScale;

		// Connector directions (used for both leaves)
		Coord dir1 = pos[idx1] - anchor;
		Coord dir2 = idx2 != INVALID ? (pos[idx2] - anchor) : (Coord(0) - dir1);

		// Leaf 1: extends along connector to body1
		{
			auto len1 = sqrt(dot(dir1, dir1));
			Coord u1 = len1 > Real(1e-6) ? dir1 / len1 : perp1; // length dir
			Coord w1 = Coord(
				u1[1] * axis[2] - u1[2] * axis[1],
				u1[2] * axis[0] - u1[0] * axis[2],
				u1[0] * axis[1] - u1[1] * axis[0]); // thickness dir = u1 x axis
			auto wLen = sqrt(dot(w1, w1));
			w1 = wLen > Real(1e-6) ? w1 / wLen : perp2;
			Coord center1 = anchor + u1 * plateHalfLen;
			transforms[base + 3] = makeBoxTransform<Coord, Real>(
				center1, u1, axis, w1, plateHalfLen, plateHalfW, plateThick);
			colors[base + 3] = COLOR_HINGE_AXIS;
		}

		// Leaf 2: extends along connector to body2 (or opposite of body1)
		{
			auto len2 = sqrt(dot(dir2, dir2));
			Coord u2 = len2 > Real(1e-6) ? dir2 / len2 : perp2;
			Coord w2 = Coord(
				u2[1] * axis[2] - u2[2] * axis[1],
				u2[2] * axis[0] - u2[0] * axis[2],
				u2[0] * axis[1] - u2[1] * axis[0]);
			auto wLen2 = sqrt(dot(w2, w2));
			w2 = wLen2 > Real(1e-6) ? w2 / wLen2 : perp1;
			Coord center2 = anchor + u2 * plateHalfLen;
			transforms[base + 4] = makeBoxTransform<Coord, Real>(
				center2, u2, axis, w2, plateHalfLen, plateHalfW, plateThick);
			colors[base + 4] = COLOR_HINGE_AXIS;
		}
	}

	template<typename Coord, typename Matrix, typename Real, typename SliderJoint>
	__global__ void buildSliderJointTransforms(
		DArray<SliderJoint> joints,
		DArray<Matrix> rotMat,
		DArray<Coord> pos,
		DArray<Transform3f> transforms,
		DArray<Vec3f> colors,
		Real length,
		Real thicknessScale,
		uint offset)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= joints.size()) return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		Coord r1 = rotMat[idx1] * joints[tId].r1;
		Coord anchor = pos[idx1] + r1;

		Coord baseAnchor;
		if (idx2 != INVALID)
			baseAnchor = pos[idx2] + rotMat[idx2] * joints[tId].r2;
		else
			baseAnchor = anchor;

		Coord axis = rotMat[idx1] * joints[tId].sliderAxis;
		axis.normalize();

		Coord perp1, perp2;
		computePerpAxes<Coord>(axis, perp1, perp2);

		uint base = offset + tId * 4;

		Real railThick = length * SLIDER_RAIL_THICK * thicknessScale;

		if (joints[tId].useRange)
		{
			Coord posMin = baseAnchor + axis * joints[tId].d_min;
			Coord posMax = baseAnchor + axis * joints[tId].d_max;
			Coord railCenter = (posMin + posMax) * Real(0.5);
			Real railHalfLen = (joints[tId].d_max - joints[tId].d_min) * Real(0.5);
			transforms[base + 0] = makeBoxTransform<Coord, Real>(
				railCenter, axis, perp1, perp2, railHalfLen, railThick, railThick);
		}
		else
		{
			Real railHalfLen = length * SLIDER_RAIL_HALF_LEN;
			transforms[base + 0] = makeBoxTransform<Coord, Real>(
				baseAnchor, axis, perp1, perp2, railHalfLen, railThick, railThick);
		}
		colors[base + 0] = COLOR_SLIDER;

		Real sliderSize = length * SLIDER_BLOCK_SIZE * thicknessScale;
		transforms[base + 1] = makeBoxTransform<Coord, Real>(
			anchor, axis, perp1, perp2, sliderSize, sliderSize, sliderSize);
		colors[base + 1] = COLOR_SLIDER;

		Real markerSize = length * SLIDER_MARKER_SIZE * thicknessScale;
		if (joints[tId].useRange)
		{
			Coord posMin = baseAnchor + axis * joints[tId].d_min;
			Coord posMax = baseAnchor + axis * joints[tId].d_max;

			transforms[base + 2] = makeBoxTransform<Coord, Real>(
				posMin, axis, perp1, perp2, markerSize, markerSize, markerSize);
			transforms[base + 3] = makeBoxTransform<Coord, Real>(
				posMax, axis, perp1, perp2, markerSize, markerSize, markerSize);
		}
		else
		{
			transforms[base + 2] = makeBoxTransform<Coord, Real>(
				anchor, axis, perp1, perp2, Real(0), Real(0), Real(0));
			transforms[base + 3] = makeBoxTransform<Coord, Real>(
				anchor, axis, perp1, perp2, Real(0), Real(0), Real(0));
		}
		colors[base + 2] = COLOR_SLIDER_MARKER;
		colors[base + 3] = COLOR_SLIDER_MARKER;
	}

	// Ball-and-socket joint is split into two kernels:
	//  - buildBallAndSocketConnectorTransforms: 2 box-shaped connectors (rendered with Box template)
	//  - buildBallAndSocketSphereTransforms:    1 sphere (rendered with Sphere template)
	// Splitting avoids the issue where a Sphere Transform fed into a Box-template pass
	// still renders as a Box.
	template<typename Coord, typename Matrix, typename Real, typename BallAndSocketJoint>
	__global__ void buildBallAndSocketConnectorTransforms(
		DArray<BallAndSocketJoint> joints,
		DArray<Matrix> rotMat,
		DArray<Coord> pos,
		DArray<Transform3f> transforms,
		DArray<Vec3f> colors,
		Real length,
		Real thicknessScale,
		uint offset)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= joints.size()) return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		Coord r1 = rotMat[idx1] * joints[tId].r1;
		Coord anchor = pos[idx1] + r1;

		// 4 instances per joint: 2 connectors + 2 square plates
		uint base = offset + tId * 4;
		Real connThick = length * CONNECTOR_THICK * thicknessScale;
		Real plateHalfSize = length * BALL_PLATE_HALF_SIZE * thicknessScale;

		// 0: Connector to body1 (亮)
		transforms[base + 0] = makeConnectorTransform<Coord, Real>(
			anchor, pos[idx1], connThick);
		colors[base + 0] = COLOR_BALL_CONNECTOR_A;

		// 1: Connector to body2 (暗)
		if (idx2 != INVALID)
		{
			transforms[base + 1] = makeConnectorTransform<Coord, Real>(
				anchor, pos[idx2], connThick);
		}
		else
		{
			Coord axis = pos[idx1] - anchor;
			auto len = sqrt(dot(axis, axis));
			axis = len > Real(1e-6) ? axis / len : Coord(1, 0, 0);
			Coord perp1, perp2;
			computePerpAxes<Coord>(axis, perp1, perp2);
			transforms[base + 1] = makeBoxTransform<Coord, Real>(
				anchor, axis, perp1, perp2, Real(0), Real(0), Real(0));
		}
		colors[base + 1] = COLOR_BALL_CONNECTOR_B;

		// 2: Plate on body1 side, at connector1 midpoint, orientation follows body1's rotation.
		{
			Coord bx = rotMat[idx1] * Coord(1, 0, 0);
			Coord by = rotMat[idx1] * Coord(0, 1, 0);
			Coord bz = rotMat[idx1] * Coord(0, 0, 1);
			Coord mid1 = (anchor + pos[idx1]) * Real(0.5);
			transforms[base + 2] = makeBoxTransform<Coord, Real>(
				mid1, by, bz, bx, plateHalfSize, plateHalfSize, connThick);
			colors[base + 2] = COLOR_BALL_CONNECTOR_A;
		}

		// 3: Plate on body2 side, at connector2 midpoint, orientation follows body2's rotation.
		{
			Coord bx = idx2 != INVALID
				? (rotMat[idx2] * Coord(1, 0, 0))
				: (rotMat[idx1] * Coord(1, 0, 0));
			Coord by = idx2 != INVALID
				? (rotMat[idx2] * Coord(0, 1, 0))
				: (rotMat[idx1] * Coord(0, 1, 0));
			Coord bz = idx2 != INVALID
				? (rotMat[idx2] * Coord(0, 0, 1))
				: (rotMat[idx1] * Coord(0, 0, 1));
			Coord mid2 = idx2 != INVALID ? ((anchor + pos[idx2]) * Real(0.5)) : anchor;
			transforms[base + 3] = makeBoxTransform<Coord, Real>(
				mid2, by, bz, bx, plateHalfSize, plateHalfSize, connThick);
			colors[base + 3] = COLOR_BALL_CONNECTOR_B;
		}
	}

	template<typename Coord, typename Matrix, typename Real, typename BallAndSocketJoint>
	__global__ void buildBallAndSocketSphereTransforms(
		DArray<BallAndSocketJoint> joints,
		DArray<Matrix> rotMat,
		DArray<Coord> pos,
		DArray<Transform3f> transforms,
		DArray<Vec3f> colors,
		Real length,
		Real thicknessScale)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= joints.size()) return;

		int idx1 = joints[tId].bodyId1;

		Coord r1 = rotMat[idx1] * joints[tId].r1;
		Coord anchor = pos[idx1] + r1;

		Real sphereRadius = length * BALL_SPHERE_RADIUS * thicknessScale;

		transforms[tId] = makeSphereTransform<Coord, Real>(
			anchor, sphereRadius);
		colors[tId] = COLOR_BALL;
	}

	// FixedJoint: 2 connectors + 3 long boxes forming a fork (叉子) structure
	// The 3 prongs splay outward from the anchor at 120° in the plane perpendicular
	// to the r1 direction (from body1 to anchor), so the fork faces away from body1.
	template<typename Coord, typename Matrix, typename Real, typename FixedJoint>
	__global__ void buildFixedJointTransforms(
		DArray<FixedJoint> joints,
		DArray<Matrix> rotMat,
		DArray<Coord> pos,
		DArray<Transform3f> transforms,
		DArray<Vec3f> colors,
		Real length,
		Real thicknessScale,
		uint offset)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= joints.size()) return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;
		Coord r1 = rotMat[idx1] * joints[tId].r1;
		Coord anchor = pos[idx1] + r1;

		// Axis from body1 to anchor (fork faces this direction)
		Coord axis = r1;
		auto axisLen = sqrt(dot(axis, axis));
		axis = axisLen > Real(1e-6) ? axis / axisLen : Coord(1, 0, 0);

		// Use body1's local axes to construct the fork plane so that the fork
		// twists with body1's spin (not a fixed arbitrary perpendicular frame).
		Coord b1x = rotMat[idx1] * Coord(1, 0, 0);
		Coord b1y = rotMat[idx1] * Coord(0, 1, 0);
		Coord b1z = rotMat[idx1] * Coord(0, 0, 1);
		// Project body1 local X onto the plane perpendicular to axis as perp1.
		Coord perp1 = b1x - axis * dot(b1x, axis);
		auto p1len = sqrt(dot(perp1, perp1));
		if (p1len < Real(1e-6))
		{
			// b1x is parallel to axis; fall back to local Y.
			perp1 = b1y - axis * dot(b1y, axis);
			p1len = sqrt(dot(perp1, perp1));
		}
		perp1 = p1len > Real(1e-6) ? perp1 / p1len : Coord(0, 1, 0);
		Coord perp2 = cross(axis, perp1);

		Real forkLen = length * FIXED_FORK_LEN;
		Real forkThick = length * JOINT_BOX_SIZE * thicknessScale;
		Real connThick = length * CONNECTOR_THICK * thicknessScale;

		// 5 instances per joint: 2 connectors + 3 fork prongs
		uint base = offset + tId * 5;

		// 0: Connector to body1 (亮紫)
		transforms[base + 0] = makeConnectorTransform<Coord, Real>(
			anchor, pos[idx1], connThick);
		colors[base + 0] = COLOR_FIXED_CONNECTOR_A;

		// 1: Connector to body2 (暗紫)
		if (idx2 != INVALID)
		{
			transforms[base + 1] = makeConnectorTransform<Coord, Real>(
				anchor, pos[idx2], connThick);
		}
		else
		{
			transforms[base + 1] = makeBoxTransform<Coord, Real>(
				anchor, axis, perp1, perp2, Real(0), Real(0), Real(0));
		}
		colors[base + 1] = COLOR_FIXED_CONNECTOR_B;

		// 2,3,4: 3 prongs at 0°, 120°, 240° around the axis, each a long box.
		// Perp frame follows body1 local axes, so fork twists with body1 spin.
		for (int i = 0; i < 3; i++)
		{
			Real angle = Real(2.0 * 3.14159265358979323846 * i / 3.0);
			Real cosA = cos(angle);
			Real sinA = sin(angle);
			Coord dir = perp1 * cosA + perp2 * sinA;

			Coord center = anchor + dir * forkLen;
			Coord perpA, perpB;
			computePerpAxes<Coord>(dir, perpA, perpB);

			transforms[base + 2 + i] = makeBoxTransform<Coord, Real>(
				center, dir, perpA, perpB, forkLen, forkThick, forkThick);
			colors[base + 2 + i] = COLOR_FIXED;
		}
	}

	// PointJoint: rendered as a sphere (goes into mJointSphereTransforms)
	// Note: Coord is not a template parameter here because it does not appear in
	// the argument list (would be non-deducible). Use auto/decltype instead.
	template<typename Real, typename PointJoint>
	__global__ void buildPointJointSphereTransforms(
		DArray<PointJoint> joints,
		DArray<Transform3f> transforms,
		DArray<Vec3f> colors,
		Real length,
		Real thicknessScale,
		uint offset)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= joints.size()) return;

		auto anchor = joints[tId].anchorPoint;

		Real sphereRadius = length * POINT_SPHERE_RADIUS * thicknessScale;

		transforms[offset + tId] = makeSphereTransform<decltype(anchor), Real>(
			anchor, sphereRadius);
		colors[offset + tId] = COLOR_POINT;
	}

	template<typename Coord, typename Matrix, typename Real, typename DistanceJoint>
	__global__ void buildDistanceJointTransforms(
		DArray<DistanceJoint> joints,
		DArray<Matrix> rotMat,
		DArray<Coord> pos,
		DArray<Transform3f> transforms,
		DArray<Vec3f> colors,
		Real length,
		Real thicknessScale,
		uint offset)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= joints.size()) return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		Coord worldR1 = pos[idx1] + rotMat[idx1] * joints[tId].r1;

		if (idx2 != INVALID)
		{
			Coord worldR2 = pos[idx2] + rotMat[idx2] * joints[tId].r2;
			Real connThick = length * CONNECTOR_THICK * thicknessScale;
			transforms[offset + tId] = makeConnectorTransform<Coord, Real>(
				worldR1, worldR2, connThick);
		}
		else
		{
			Coord u = Coord(1, 0, 0);
			Coord v = Coord(0, 1, 0);
			Coord w = Coord(0, 0, 1);
			transforms[offset + tId] = makeBoxTransform<Coord, Real>(
				worldR1, u, v, w, Real(0), Real(0), Real(0));
		}
		colors[offset + tId] = COLOR_DISTANCE;
	}

	// Constructor
	template<typename TDataType>
	GLDiscreteElementJointVisualModule<TDataType>::GLDiscreteElementJointVisualModule()
	{
		this->setName("discrete_element_joint_visual");

		// Set ranges for scale variables
		this->varLength()->setRange(0, 10);
		this->varThicknessScale()->setRange(0, 10);
		this->varBaseColor()->setValue(Color(0.5, 0.721, 1));

		this->varAlpha()->setValue(0.5);

	}

	template<typename TDataType>
	std::string GLDiscreteElementJointVisualModule<TDataType>::caption()
	{
		return "GLDiscreteElementJointVisualModule";
	}

	// updateImpl()
	template<typename TDataType>
	void GLDiscreteElementJointVisualModule<TDataType>::updateImpl()
	{
		auto topo = this->inDiscreteElements()->constDataPtr();

		// Get joint arrays
		auto& hingeJoints = topo->hingeJoints();
		auto& sliderJoints = topo->sliderJoints();
		auto& ballJoints = topo->ballAndSocketJoints();
		auto& fixedJoints = topo->fixedJoints();
		auto& pointJoints = topo->pointJoints();
		auto& distanceJoints = topo->distanceJoints();

		auto& pos = topo->position();
		auto& rotMat = topo->rotation();

		// Generate joint instance transforms
		uint hingeCount = this->varShowHingeJoint()->getValue() ? hingeJoints.size() : 0;
		uint sliderCount = this->varShowSliderJoint()->getValue() ? sliderJoints.size() : 0;
		uint ballCount = this->varShowBallAndSocketJoint()->getValue() ? ballJoints.size() : 0;
		uint fixedCount = this->varShowFixedJoint()->getValue() ? fixedJoints.size() : 0;
		uint pointCount = this->varShowPointJoint()->getValue() ? pointJoints.size() : 0;
		uint distanceCount = this->varShowDistanceJoint()->getValue() ? distanceJoints.size() : 0;

		uint jointTotal = hingeCount * 5 + sliderCount * 4 + ballCount * 4
			+ fixedCount * 5 + distanceCount;

		mJointTransforms.resize(jointTotal);
		mJointColors.resize(jointTotal);

		// Sphere-template joint instances (BallAndSocket ball + PointJoint sphere)
		// rendered via mJointSpherePass instead of the Box-template mJointPass.
		uint jointSphereTotal = ballCount + pointCount;
		mJointSphereTransforms.resize(jointSphereTotal);
		mJointSphereColors.resize(jointSphereTotal);

		Real len = this->varLength()->getValue() * JOINT_LENGTH_SCALE;
		Real thicknessScale = this->varThicknessScale()->getValue() * JOINT_THICKNESS_SCALE;

		uint offset = 0;

		if (hingeCount > 0)
		{
			cuExecuteNoSync(hingeCount,
				buildHingeJointTransforms,
				hingeJoints, rotMat, pos,
				mJointTransforms, mJointColors, len, thicknessScale, offset);
			offset += hingeCount * 5;
		}

		if (sliderCount > 0)
		{
			cuExecuteNoSync(sliderCount,
				buildSliderJointTransforms,
				sliderJoints, rotMat, pos,
				mJointTransforms, mJointColors, len, thicknessScale, offset);
			offset += sliderCount * 4;
		}

		if (ballCount > 0)
		{
			cuExecuteNoSync(ballCount,
				buildBallAndSocketConnectorTransforms,
				ballJoints, rotMat, pos,
				mJointTransforms, mJointColors, len, thicknessScale, offset);
			offset += ballCount * 4;

			// Sphere part rendered with Sphere template via mJointSpherePass
			cuExecuteNoSync(ballCount,
				buildBallAndSocketSphereTransforms,
				ballJoints, rotMat, pos,
				mJointSphereTransforms, mJointSphereColors, len, thicknessScale);
		}

		if (fixedCount > 0)
		{
			Real fixJointLen = len * 0.25;
			cuExecuteNoSync(fixedCount,
				buildFixedJointTransforms,
				fixedJoints, rotMat, pos,
				mJointTransforms, mJointColors, fixJointLen, thicknessScale, offset);
			offset += fixedCount * 5;
		}

		// PointJoint spheres are appended after BallAndSocket spheres
		// in mJointSphereTransforms (offset = ballCount).
		if (pointCount > 0)
		{
			cuExecuteNoSync(pointCount,
				buildPointJointSphereTransforms,
				pointJoints,
				mJointSphereTransforms, mJointSphereColors, len, thicknessScale, ballCount);
		}

		if (distanceCount > 0)
		{
			cuExecuteNoSync(distanceCount,
				buildDistanceJointTransforms,
				distanceJoints, rotMat, pos,
				mJointTransforms, mJointColors, len, thicknessScale, offset);
		}

		// Synchronize all kernel launches before loading data into buffers
		cuSynchronize();
	}

	// ---------------------------------------------------------------------------
	// GL functions
	// ---------------------------------------------------------------------------

	template<typename TDataType>
	bool GLDiscreteElementJointVisualModule<TDataType>::initializeGL()
	{
		// create shader program
		mShaderProgram = Program::createProgramSPIRV(
			SURFACE_VERT, sizeof(SURFACE_VERT),
			SURFACE_FRAG, sizeof(SURFACE_FRAG),
			SURFACE_GEOM, sizeof(SURFACE_GEOM));

		// create shader uniform buffer
		mRenderParamsUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		mPBRMaterialUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		// create instance passes
		auto initPass = [](InstancePass& pass) {
			pass.vao.create();
			pass.vertexIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			pass.vertexPosition.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			pass.instanceTransform.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			pass.instanceColor.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		};
		initPass(mJointPass);
		initPass(mJointSpherePass);

		// Create unit box [-0.5, 0.5]^3: 8 vertices, 12 triangles
		std::vector<Vec3f> pts;
		pts.resize(8);
		pts[0] = Vec3f(-0.5f, -0.5f, -0.5f);
		pts[1] = Vec3f(0.5f, -0.5f, -0.5f);
		pts[2] = Vec3f(0.5f, 0.5f, -0.5f);
		pts[3] = Vec3f(-0.5f, 0.5f, -0.5f);
		pts[4] = Vec3f(-0.5f, -0.5f, 0.5f);
		pts[5] = Vec3f(0.5f, -0.5f, 0.5f);
		pts[6] = Vec3f(0.5f, 0.5f, 0.5f);
		pts[7] = Vec3f(-0.5f, 0.5f, 0.5f);
		mStandardBox.setPoints(pts);
		pts.clear();

		std::vector<Topology::Triangle> tris;
		tris.resize(12);
		tris[0] = Topology::Triangle(0, 1, 2);
		tris[1] = Topology::Triangle(0, 2, 3);
		tris[2] = Topology::Triangle(0, 4, 5);
		tris[3] = Topology::Triangle(0, 5, 1);
		tris[4] = Topology::Triangle(4, 7, 6);
		tris[5] = Topology::Triangle(4, 6, 5);
		tris[6] = Topology::Triangle(1, 5, 6);
		tris[7] = Topology::Triangle(1, 6, 2);
		tris[8] = Topology::Triangle(2, 6, 7);
		tris[9] = Topology::Triangle(2, 7, 3);
		tris[10] = Topology::Triangle(0, 3, 7);
		tris[11] = Topology::Triangle(0, 7, 4);
		mStandardBox.setTriangles(tris);
		tris.clear();

		// Load sphere from OBJ file
		mStandardSphere.loadObjFile(getAssetPath() + "standard/standard_icosahedron.obj");

		return true;
	}

	template<typename TDataType>
	void GLDiscreteElementJointVisualModule<TDataType>::releaseGL()
	{
		// release shader resources
		if (mShaderProgram) {
			mShaderProgram->release();
			delete mShaderProgram;
			mShaderProgram = nullptr;
		}
		mRenderParamsUBlock.release();
		mPBRMaterialUBlock.release();

		// release instance passes
		auto releasePass = [](InstancePass& pass) {
			pass.vao.release();
			pass.vertexIndex.release();
			pass.vertexPosition.release();
			pass.instanceTransform.release();
			pass.instanceColor.release();
		};
		releasePass(mJointPass);
		releasePass(mJointSpherePass);
	}

	template<typename TDataType>
	void GLDiscreteElementJointVisualModule<TDataType>::updateGL()
	{
		// 1. Load template mesh data once (into instance passes)
		if (!mTemplateLoaded)
		{
			bool allLoaded = true;
			auto loadTemplate = [&allLoaded](InstancePass& pass,
				DArray<Topology::Triangle>& templateIndices,
				DArray<Vec3f>& templateVertices)
			{
				pass.numTriangles = templateIndices.size();
				if (pass.numTriangles == 0) { allLoaded = false; return; }
				pass.vertexIndex.load(templateIndices);
				pass.vertexPosition.load(templateVertices);
				pass.vertexIndex.updateGL();
				pass.vertexPosition.updateGL();
			};
			// Joint pass uses Box template for connectors/hinge axis/slider rails/fixed/point/distance boxes
			loadTemplate(mJointPass,  mStandardBox.triangleIndices(),    mStandardBox.getPoints());
			// Joint sphere pass uses Sphere template for the ball of ball-and-socket joints
			loadTemplate(mJointSpherePass, mStandardSphere.triangleIndices(), mStandardSphere.getPoints());
			if (allLoaded) mTemplateLoaded = true;
		}

		// 2. Update instance data (transforms + colors) every frame
		auto updateInstance = [](InstancePass& pass,
			DArray<Transform3f>& transforms,
			DArray<Vec3f>& colors)
		{
			pass.instanceCount = transforms.size();
			if (pass.instanceCount == 0 || pass.numTriangles == 0) return;

			pass.instanceTransform.load(transforms);
			pass.instanceColor.load(colors);
			pass.instanceTransform.updateGL();
			pass.instanceColor.updateGL();

			// Re-bind VAO only when instance buffer changes
			pass.vao.bind();
			pass.vertexIndex.bind();
			glEnableVertexAttribArray(0);
			glVertexAttribIPointer(0, 1, GL_INT, sizeof(int), (void*)0);
			pass.vao.bindVertexBuffer(&pass.instanceTransform, 3, 3, GL_FLOAT, sizeof(Transform3f), 0, 1);
			pass.vao.bindVertexBuffer(&pass.instanceTransform, 4, 3, GL_FLOAT, sizeof(Transform3f), sizeof(Vec3f), 1);
			pass.vao.bindVertexBuffer(&pass.instanceTransform, 5, 3, GL_FLOAT, sizeof(Transform3f), 2 * sizeof(Vec3f), 1);
			pass.vao.bindVertexBuffer(&pass.instanceTransform, 6, 3, GL_FLOAT, sizeof(Transform3f), 3 * sizeof(Vec3f), 1);
			pass.vao.bindVertexBuffer(&pass.instanceTransform, 7, 3, GL_FLOAT, sizeof(Transform3f), 4 * sizeof(Vec3f), 1);
			pass.vao.bindVertexBuffer(&pass.instanceColor, 8, 3, GL_FLOAT, sizeof(Vec3f), 0, 1);
			pass.vao.unbind();
		};

		updateInstance(mJointPass,  mJointTransforms,  mJointColors);
		updateInstance(mJointSpherePass, mJointSphereTransforms, mJointSphereColors);

		glCheckError();
	}

	template<typename TDataType>
	void GLDiscreteElementJointVisualModule<TDataType>::paintGL(const RenderParams& rparams)
	{
		if (mJointPass.instanceCount == 0 && mJointSpherePass.instanceCount == 0)
			return;

		mShaderProgram->use();

		mShaderProgram->setInt("uVertexNormal", 0);
		mShaderProgram->setInt("uColorMode", 0);

		mRenderParamsUBlock.load((void*)&rparams, sizeof(RenderParams));
		mRenderParamsUBlock.bindBufferBase(0);

		// Helper: load PBR material with given alpha
		auto loadPBR = [&](float alpha) {
			PBRMaterial pbr;
			auto color = this->varBaseColor()->getValue();
			pbr.color = glm::vec3{ color.r, color.g, color.b };
			pbr.metallic = this->varMetallic()->getValue();
			pbr.roughness = this->varRoughness()->getValue();
			pbr.alpha = alpha;
			pbr.useAOTex = 0;
			pbr.useMetallicTex = 0;
			pbr.useRoughnessTex = 0;
			mPBRMaterialUBlock.load((void*)&pbr, sizeof(pbr));
			mPBRMaterialUBlock.bindBufferBase(1);
		};

		// Draw joint instance passes with varAlpha
		//    - mJointPass:        Box-template instances (connectors, hinge axis, slider rails, fixed/point/distance)
		//    - mJointSpherePass:  Sphere-template instances (ball of ball-and-socket)
		{
			loadPBR(float(this->varAlpha()->getValue()));
			auto drawPass = [&](InstancePass& pass) {
				if (pass.instanceCount == 0 || pass.numTriangles == 0) return;
				pass.vertexPosition.bindBufferBase(8);
				pass.vao.bind();
				mShaderProgram->setInt("uInstanced", 1);
				glDrawArraysInstanced(GL_TRIANGLES, 0, pass.numTriangles * 3, pass.instanceCount);
				pass.vao.unbind();
			};

			drawPass(mJointPass);
			drawPass(mJointSpherePass);
		}

		glCheckError();
	}

	DEFINE_CLASS(GLDiscreteElementJointVisualModule);

}
