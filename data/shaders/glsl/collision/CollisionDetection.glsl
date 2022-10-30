#ifndef COLLISIONDETECTION_H
#define COLLISIONDETECTION_H
#extension GL_EXT_debug_printf : enable
#include "../rigidbody/SharedData.glsl"
#include "../math/Primitive3D.glsl"
#include "../math/Quat.glsl"

struct Transform
{
	vec3 position;
	mat3 rotation;
};


struct Manifold
{
	vec3 normal;				// From A to B
	vec3 position[8];				// World coordinate of contact
	float penetration[8];			// Depth of penetration from collision, its value is negative
	int contactCount;
};

float fsign(float v)
{
	return v < 0 ? -1.0f : 1.0f;
}

vec3 mul(const Transform tx, const vec3 v)
{
	return vec3(tx.rotation * v + tx.position);
}

vec3 mul(const mat3 rot, const vec3 trans, const vec3 v)
{
	return vec3(rot * v + trans);
}

vec3 mul_t(const Transform tx, const vec3 v)
{
	return transpose(tx.rotation) * (v - tx.position);
}

//--------------------------------------------------------------------------------------------------
vec3 mul_t(const mat3 r, const vec3 v)
{
	return transpose(r) * v;
}

//--------------------------------------------------------------------------------------------------
// qBoxtoBox
//--------------------------------------------------------------------------------------------------
bool trackFaceAxis(inout int axis, inout float sMax, inout vec3 axisNormal, int n, float s, const vec3 normal)
{
	if (s > float(0.0))
		return true;

	if (s > sMax)
	{
		sMax = s;
		axis = n;
		axisNormal = normal;
	}

	return false;
}

//--------------------------------------------------------------------------------------------------
bool trackEdgeAxis(inout int axis, inout float sMax, inout vec3 axisNormal, int n, float s, const vec3 normal)
{
	if (s > float(0.0))
		return true;

	float l = float(1.0) / length(normal);
	s *= l;

	if (s > sMax)
	{
		sMax = s;
		axis = n;
		axisNormal = normal * l;
	}

	return false;
}


//--------------------------------------------------------------------------------------------------
void computeReferenceEdgesAndBasis(inout uint outEdge[4], inout mat3 basis, inout vec3 e, const vec3 eR, const Transform rtx, vec3 n, int axis)
{
	n = mul_t(rtx.rotation, n);

	if (axis >= 3)
		axis -= 3;

	mat3 rot_t = rtx.rotation;
	mat3 outB;

	switch (axis)
	{
	case 0:
		if (n.x > float(0.0))
		{
			outEdge[0] = 1;
			outEdge[1] = 8;
			outEdge[2] = 7;
			outEdge[3] = 9;

			e = vec3(eR.y, eR.z, eR.x);
			outB[0] = rot_t[1];
			outB[1] = rot_t[2];
			outB[2] = rot_t[0];
		}
		else
		{
			outEdge[0] = 11;
			outEdge[1] = 3;
			outEdge[2] = 10;
			outEdge[3] = 5;

			e = vec3(eR.z, eR.y, eR.x);
			outB[0] = rot_t[2];
			outB[1] = rot_t[1];
			outB[2] = -rot_t[0];
		}
		break;

	case 1:
		if (n.y > float(0.0))
		{
			outEdge[0] = 0;
			outEdge[1] = 1;
			outEdge[2] = 2;
			outEdge[3] = 3;

			e = vec3(eR.z, eR.x, eR.y);
			outB[0] = rot_t[2];
			outB[1] = rot_t[0];
			outB[2] = rot_t[1];
		}
		else
		{
			outEdge[0] = 4;
			outEdge[1] = 5;
			outEdge[2] = 6;
			outEdge[3] = 7;

			e = vec3(eR.z, eR.x, eR.y);
			outB[0] = rot_t[2];
			outB[1] = -rot_t[0];
			outB[2] = -rot_t[1];
		}
		break;

	case 2:
		if (n.z > float(0.0))
		{
			outEdge[0] = 11;
			outEdge[1] = 4;
			outEdge[2] = 8;
			outEdge[3] = 0;

			e = vec3(eR.y, eR.x, eR.z);
			outB[0] = -rot_t[1];
			outB[1] = rot_t[0];
			outB[2] = rot_t[2];
		}
		else
		{
			outEdge[0] = 6;
			outEdge[1] = 10;
			outEdge[2] = 2;
			outEdge[3] = 9;

			e = vec3(eR.y, eR.x, eR.z);
			outB[0] = -rot_t[1];
			outB[1] = -rot_t[0];
			outB[2] = -rot_t[2];
		}
		break;
	}

	basis = outB;
}

//--------------------------------------------------------------------------------------------------
void computeIncidentFace(inout vec3 outVert[4], const Transform itx, const vec3 e, vec3 n)
{
	n = -mul_t(itx.rotation, n);
	vec3 absN = abs(n);

	if (absN.x > absN.y && absN.x > absN.z)
	{
		if (n.x > float(0.0))
		{
			outVert[0] = vec3(e.x, e.y, -e.z);
			outVert[1] = vec3(e.x, e.y, e.z);
			outVert[2] = vec3(e.x, -e.y, e.z);
			outVert[3] = vec3(e.x, -e.y, -e.z);
		}
		else
		{
			outVert[0] = vec3(-e.x, -e.y, e.z);
			outVert[1] = vec3(-e.x, e.y, e.z);
			outVert[2] = vec3(-e.x, e.y, -e.z);
			outVert[3] = vec3(-e.x, -e.y, -e.z);
		}
	}
	else if (absN.y > absN.x && absN.y > absN.z)
	{
		if (n.y > float(0.0))
		{
			outVert[0] = vec3(-e.x, e.y, e.z);
			outVert[1] = vec3(e.x, e.y, e.z);
			outVert[2] = vec3(e.x, e.y, -e.z);
			outVert[3] = vec3(-e.x, e.y, -e.z);
		}
		else
		{
			outVert[0] = vec3(e.x, -e.y, e.z);
			outVert[1] = vec3(-e.x, -e.y, e.z);
			outVert[2] = vec3(-e.x, -e.y, -e.z);
			outVert[3] = vec3(e.x, -e.y, -e.z);
		}
	}
	else
	{
		if (n.z > float(0.0))
		{
			outVert[0] = vec3(-e.x, e.y, e.z);
			outVert[1] = vec3(-e.x, -e.y, e.z);
			outVert[2] = vec3(e.x, -e.y, e.z);
			outVert[3] = vec3(e.x, e.y, e.z);
		}
		else
		{
			outVert[0] = vec3(e.x, -e.y, -e.z);
			outVert[1] = vec3(-e.x, -e.y, -e.z);
			outVert[2] = vec3(-e.x, e.y, -e.z);
			outVert[3] = vec3(e.x, e.y, -e.z);
		}
	}

	for (int i = 0; i < 4; ++i)
		outVert[i] = mul(itx, outVert[i]);
}

//--------------------------------------------------------------------------------------------------
#define InFront( a ) \
	((a) < float( 0.0 ))

#define Behind( a ) \
	((a) >= float( 0.0 ))

#define On( a ) \
	((a) < float( 0.005 ) && (a) > -float( 0.005 ))

int orthographic(inout vec3 outVert[8], float sign, float e, int axis, uint clipEdge, vec3 inVert[8], int inCount)
{
	int outCount = 0;
	vec3 a = inVert[inCount - 1];

	for (int i = 0; i < inCount; ++i)
	{
		vec3 b = inVert[i];

		float da = sign * a[axis] - e;
		float db = sign * b[axis] - e;

		vec3 cv;

		// B
		if (((InFront(da) && InFront(db)) || On(da) || On(db)))
		{
			outVert[outCount++] = b;
		}
		// I
		else if (InFront(da) && Behind(db))
		{
			cv = a + (b - a) * (da / (da - db));
			outVert[outCount++] = cv;
		}
		// I, B
		else if (Behind(da) && InFront(db))
		{
			cv = a + (b - a) * (da / (da - db));
			outVert[outCount++] = cv;
			outVert[outCount++] = b;
		}

		a = b;
	}

	return outCount;
}

int clip(inout vec3 outVerts[8], inout float outDepths[8], const vec3 rPos, const vec3 e, uint clipEdges[4], const mat3 basis, vec3 incident[4])
{
	int inCount = 4;
	int outCount;
	vec3 inV[8];
	vec3 outV[8];

	for (int i = 0; i < 4; ++i)
		inV[i] = mul_t(basis, incident[i] - rPos);

	outCount = orthographic(outV, float(1.0), e.x, 0, clipEdges[0], inV, inCount);

	if (outCount == 0)
		return 0;

	inCount = orthographic(inV, float(1.0), e.y, 1, clipEdges[1], outV, outCount);

	if (inCount == 0)
		return 0;

	outCount = orthographic(outV, float(-1.0), e.x, 0, clipEdges[2], inV, inCount);

	if (outCount == 0)
		return 0;

	inCount = orthographic(inV, float(-1.0), e.y, 1, clipEdges[3], outV, outCount);

	// Keep incident vertices behind the reference face
	outCount = 0;
	for (int i = 0; i < inCount; ++i)
	{
		float d = inV[i].z - e.z;

		if (d <= float(0.0))
		{
			outVerts[outCount] = basis * inV[i] + rPos;
//			outVerts[outCount].f = in[i].f;
			outDepths[outCount++] = d;
		}
	}

	//assert(outCount <= 8);

	return outCount;
}

//--------------------------------------------------------------------------------------------------
void edgesContact(inout vec3 CA, inout vec3 CB, const vec3 PA, const vec3 QA, const vec3 PB, const vec3 QB)
{
	vec3 DA = QA - PA;
	vec3 DB = QB - PB;
	vec3 r = PA - PB;
	float a = dot(DA, DA);
	float e = dot(DB, DB);
	float f = dot(DB, r);
	float c = dot(DA, r);

	float b = dot(DA, DB);
	float denom = a * e - b * b;

	float TA = (b * f - c * e) / denom;
	float TB = (b * TA + f) / e;

	CA = PA + DA * TA;
	CB = PB + DB * TB;
}

//--------------------------------------------------------------------------------------------------
void computeSupportEdge(inout vec3 aOut, inout vec3 bOut, const mat3 rot, const vec3 trans, const vec3 e, vec3 n)
{
	n = mul_t(rot, n);
	vec3 absN = abs(n);
	vec3 a, b;

	// x > y
	if (absN.x > absN.y)
	{
		// x > y > z
		if (absN.y > absN.z)
		{
			a = vec3(e.x, e.y, e.z);
			b = vec3(e.x, e.y, -e.z);
		}
		// x > z > y || z > x > y
		else
		{
			a = vec3(e.x, e.y, e.z);
			b = vec3(e.x, -e.y, e.z);
		}
	}

	// y > x
	else
	{
		// y > x > z
		if (absN.x > absN.z)
		{
			a = vec3(e.x, e.y, e.z);
			b = vec3(e.x, e.y, -e.z);
		}
		// z > y > x || y > z > x
		else
		{
			a = vec3(e.x, e.y, e.z);
			b = vec3(-e.x, e.y, e.z);
		}
	}

	float signx = fsign(n.x);
	float signy = fsign(n.y);
	float signz = fsign(n.z);

	a.x *= signx;
	a.y *= signy;
	a.z *= signz;
	b.x *= signx;
	b.y *= signy;
	b.z *= signz;

	aOut = mul(rot, trans, a);
	bOut = mul(rot, trans, b);
}

void OBBtoOBB(inout Manifold m, Box box0, Box box1)
{
	vec3 v = box1.center.xyz - box0.center.xyz;

	vec3 eA = box0.halfLength.xyz;
	vec3 eB = box1.halfLength.xyz;

	mat3 rotA = quat_to_mat3(box0.rot);
	mat3 rotB = quat_to_mat3(box1.rot);

	// B's frame in A's space
	mat3 C = transpose(rotA) * rotB;
	mat3 absC;
	bool parallel = false;
	const float kCosTol = float(1.0e-6);
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			float val = abs(C[i][j]);
			absC[i][j] = val;

			if (val + kCosTol >= float(1.0))
				parallel = true;
		}
	}

	mat3 C_t = transpose(C);
	mat3 absC_t = transpose(absC);

	// Vector from center A to center B in A's space
	vec3 t = transpose(rotA) * v;

	// Query states
	float s;
	float aMax = -FLOAT_MAX;
	float bMax = -FLOAT_MAX;
	float eMax = -FLOAT_MAX;
	int aAxis = ~0;
	int bAxis = ~0;
	int eAxis = ~0;
	vec3 nA;
	vec3 nB;
	vec3 nE;

	// Face axis checks

	// a's x axis
	s = abs(t.x) - (box0.halfLength.x + dot(absC_t[0], box1.halfLength.xyz));
	if (trackFaceAxis(aAxis, aMax, nA, 0, s, rotA[0]))
		return;

	// a's y axis
	s = abs(t.y) - (box0.halfLength.y + dot(absC_t[1], box1.halfLength.xyz));
	if (trackFaceAxis(aAxis, aMax, nA, 1, s, rotA[1]))
		return;

	// a's z axis
	s = abs(t.z) - (box0.halfLength.z + dot(absC_t[2], box1.halfLength.xyz));
	if (trackFaceAxis(aAxis, aMax, nA, 2, s, rotA[2]))
		return;

	// b's x axis
	s = abs(dot(t, C[0])) - (box1.halfLength.x + dot(absC[0], box0.halfLength.xyz));
	if (trackFaceAxis(bAxis, bMax, nB, 3, s, rotB[0]))
		return;

	// b's y axis
	s = abs(dot(t, C[1])) - (box1.halfLength.y + dot(absC[1], box0.halfLength.xyz));
	if (trackFaceAxis(bAxis, bMax, nB, 4, s, rotB[1]))
		return;

	// b's z axis
	s = abs(dot(t, C[2])) - (box1.halfLength.z + dot(absC[2], box0.halfLength.xyz));
	if (trackFaceAxis(bAxis, bMax, nB, 5, s, rotB[2]))
		return;

	if (!parallel)
	{
		// Edge axis checks
		float rA;
		float rB;

		// Cross( a.x, b.x )
		rA = eA.y * absC[0][2] + eA.z * absC[0][1];
		rB = eB.y * absC[2][0] + eB.z * absC[1][0];
		s = abs(t.z * C[0][1] - t.y * C[0][2]) - (rA + rB);
		if (trackEdgeAxis(eAxis, eMax, nE, 6, s, vec3(float(0.0), -C[0][2], C[0][1])))
			return;

		// Cross( a.x, b.y )
		rA = eA.y * absC[1][2] + eA.z * absC[1][1];
		rB = eB.x * absC[2][0] + eB.z * absC[0][0];
		s = abs(t.z * C[1][1] - t.y * C[1][2]) - (rA + rB);
		if (trackEdgeAxis(eAxis, eMax, nE, 7, s, vec3(float(0.0), -C[1][2], C[1][1])))
			return;

		// Cross( a.x, b.z )
		rA = eA.y * absC[2][2] + eA.z * absC[2][1];
		rB = eB.x * absC[1][0] + eB.y * absC[0][0];
		s = abs(t.z * C[2][1] - t.y * C[2][2]) - (rA + rB);
		if (trackEdgeAxis(eAxis, eMax, nE, 8, s, vec3(float(0.0), -C[2][2], C[2][1])))
			return;

		// Cross( a.y, b.x )
		rA = eA.x * absC[0][2] + eA.z * absC[0][0];
		rB = eB.y * absC[2][1] + eB.z * absC[1][1];
		s = abs(t.x * C[0][2] - t.z * C[0][0]) - (rA + rB);
		if (trackEdgeAxis(eAxis, eMax, nE, 9, s, vec3(C[0][2], float(0.0), -C[0][0])))
			return;

		// Cross( a.y, b.y )
		rA = eA.x * absC[1][2] + eA.z * absC[1][0];
		rB = eB.x * absC[2][1] + eB.z * absC[0][1];
		s = abs(t.x * C[1][2] - t.z * C[1][0]) - (rA + rB);
		if (trackEdgeAxis(eAxis, eMax, nE, 10, s, vec3(C[1][2], float(0.0), -C[1][0])))
			return;

		// Cross( a.y, b.z )
		rA = eA.x * absC[2][2] + eA.z * absC[2][0];
		rB = eB.x * absC[1][1] + eB.y * absC[0][1];
		s = abs(t.x * C[2][2] - t.z * C[2][0]) - (rA + rB);
		if (trackEdgeAxis(eAxis, eMax, nE, 11, s, vec3(C[2][2], float(0.0), -C[2][0])))
			return;

		// Cross( a.z, b.x )
		rA = eA.x * absC[0][1] + eA.y * absC[0][0];
		rB = eB.y * absC[2][2] + eB.z * absC[1][2];
		s = abs(t.y * C[0][0] - t.x * C[0][1]) - (rA + rB);
		if (trackEdgeAxis(eAxis, eMax, nE, 12, s, vec3(-C[0][1], C[0][0], float(0.0))))
			return;

		// Cross( a.z, b.y )
		rA = eA.x * absC[1][1] + eA.y * absC[1][0];
		rB = eB.x * absC[2][2] + eB.z * absC[0][2];
		s = abs(t.y * C[1][0] - t.x * C[1][1]) - (rA + rB);
		if (trackEdgeAxis(eAxis, eMax, nE, 13, s, vec3(-C[1][1], C[1][0], float(0.0))))
			return;

		// Cross( a.z, b.z )
		rA = eA.x * absC[2][1] + eA.y * absC[2][0];
		rB = eB.x * absC[1][2] + eB.y * absC[0][2];
		s = abs(t.y * C[2][0] - t.x * C[2][1]) - (rA + rB);
		if (trackEdgeAxis(eAxis, eMax, nE, 14, s, vec3(-C[2][1], C[2][0], float(0.0))))
			return;
	}

	// Artificial axis bias to improve frame coherence
	const float kRelTol = float(0.95);
	const float kAbsTol = float(0.01);
	int axis;
	float sMax;
	vec3 n;
	float faceMax = max(aMax, bMax);
	if (kRelTol * eMax > faceMax + kAbsTol)
	{
		axis = eAxis;
		sMax = eMax;
		n = nE;
	}
	else
	{
		if (kRelTol * bMax > aMax + kAbsTol)
		{
			axis = bAxis;
			sMax = bMax;
			n = nB;
		}
		else
		{
			axis = aAxis;
			sMax = aMax;
			n = nA;
		}
	}

	if (dot(n, v) < float(0.0))
		n = -n;

	if (axis == ~0)
		return;

	Transform atx;
	Transform btx;
	atx.position = box0.center.xyz;
	atx.rotation = rotA;

	btx.position = box1.center.xyz;
	btx.rotation = rotB;

	// if(axis >= 6)
	// 	return;

	if (axis < 6)
	{
		Transform rtx;
		Transform itx;
		vec3 eR;
		vec3 eI;
		bool flip;

		if (axis < 3)
		{
			rtx = atx;
			itx = btx;
			eR = eA;
			eI = eB;
			flip = false;
		}
		else
		{
			rtx = btx;
			itx = atx;
			eR = eB;
			eI = eA;
			flip = true;
			n = -n;
		}

		// Compute reference and incident edge information necessary for clipping
		vec3 incident[4];
		computeIncidentFace(incident, itx, eI, n);
		uint clipEdges[4];
		mat3 basis;
		vec3 e;
		computeReferenceEdgesAndBasis(clipEdges, basis, e, eR, rtx, n, axis);

		// Clip the incident face against the reference face side planes
		vec3 outVert[8];
		float depths[8];
		int outNum;
		outNum = clip(outVert, depths, rtx.position, e, clipEdges, basis, incident);

		if (outNum > 0)
		{
			m.contactCount = outNum;
			m.normal = flip ? -n : n;

			for (int i = 0; i < outNum; ++i)
			{
				m.position[i] = outVert[i];
				m.penetration[i] = depths[i];
			}
		}
		else
			m.contactCount = 0;
	}
	else
	{
		n = rotA * n;

		if (dot(n, v) < float(0.0))
			n = -n;

		vec3 PA, QA;
		vec3 PB, QB;
		computeSupportEdge(PA, QA, rotA, box0.center.xyz, eA, n);
		computeSupportEdge(PB, QB, rotB, box1.center.xyz, eB, -n);

		vec3 CA, CB;
		edgesContact(CA, CB, PA, QA, PB, QB);

		m.normal = n;
		m.contactCount = 1;

		m.penetration[0] = sMax;
		m.position[0] = (CA + CB) * float(0.5);
	}
}

void SpheretoSphere(inout Manifold m, Sphere sphere0, Sphere sphere1)
{
	vec3 c0 = sphere0.center;
	vec3 c1 = sphere1.center;

	float r0 = sphere0.radius;
	float r1 = sphere1.radius;

	vec3 dir = c1 - c0;
	float sMax = length(dir) - r0 - r1;
	if(sMax >= 0)
		return;

	vec3 n = normalize(dir);

	m.normal = n;
	m.penetration[0] = sMax;
	m.position[0] = c0 + (r0 - 0.5 * sMax) * n;
	m.contactCount = 1;
}

void CapsuletoCapsule(inout Manifold m, Capsule capsule0, Capsule capsule1)
{
	float h0 = capsule0.halfLength;
	float h1 = capsule1.halfLength;
	float r0 = capsule0.radius;
	float r1 = capsule1.radius;
	vec3 c0 = capsule0.center.xyz;
	vec3 c1 = capsule1.center.xyz;

	vec3 d0 = quat_rotate(capsule0.rot, vec3(h0, 0, 0));
	vec3 d1 = quat_rotate(capsule1.rot, vec3(h1, 0, 0));

	Segment3D seg0 = makeSegment3D(c0 - d0, c0 + d0);
	Segment3D seg1 = makeSegment3D(c1 - d1, c1 + d1);

	Segment3D prox = proximity(seg0, seg1);

	vec3 dir = prox.v1 - prox.v0;
	float sMax = length(dir) - r0 - r1;

	if(sMax >= 0)
		return;

	vec3 n = normalize(dir);

	m.normal = n;
	m.penetration[0] = sMax;
	m.position[0] = prox.v0 + (r0 - 0.5 * sMax) * n;
	m.contactCount = 1;
}

void SpheretoCapsule(inout Manifold m, Sphere sphere0, Capsule capsule1)
{
	vec3 c0 = sphere0.center;
	float r0 = sphere0.radius;

	float h1 = capsule1.halfLength;
	float r1 = capsule1.radius;
	vec3 c1 = capsule1.center.xyz;

	vec3 d1 = quat_rotate(capsule1.rot, vec3(h1, 0, 0));

	Segment3D seg1 = makeSegment3D(c1 - d1, c1 + d1);

	vec3 vproj = project(c0, seg1);

	vec3 dir = vproj - c0;
	float sMax = length(dir) - r0 - r1;

	if(sMax >= 0)
		return;

	vec3 n = normalize(dir);

	m.normal = n;
	m.penetration[0] = sMax;
	m.position[0] = c0 + (r0 - 0.5 * sMax) * n;
	m.contactCount = 1;
}

void SpheretoBox(inout Manifold m, Sphere sphere0, Box box)
{
	vec3 c0 = sphere0.center;
	float r0 = sphere0.radius;

	OBB3D obb;
	obb.u = quat_rotate(box.rot, vec3(1, 0, 0));
	obb.v = quat_rotate(box.rot, vec3(0, 1, 0));
	obb.w = quat_rotate(box.rot, vec3(0, 0, 1));
	obb.center = box.center.xyz;
	obb.extent = box.halfLength.xyz;

	vec3 vproj = project(c0, obb);
	bool bInside = inside(c0, obb);

	vec3 dir = bInside ? c0 - vproj : vproj - c0;
	float sMax = bInside ? -length(dir) - r0 : length(dir) - r0;

	if(sMax >= 0)
		return;

	vec3 n = normalize(dir);

	m.normal = n;
	m.penetration[0] = sMax;
	m.position[0] = vproj;
	m.contactCount = 1;
}

void CapsuletoBox(inout Manifold m, Capsule capsule0, Box box)
{
	float h0 = capsule0.halfLength;
	float r0 = capsule0.radius;
	vec3 c0 = capsule0.center.xyz;
	vec3 d0 = quat_rotate(capsule0.rot, vec3(h0, 0, 0));

	Segment3D seg0 = makeSegment3D(c0 - d0, c0 + d0);

	OBB3D obb;
	obb.u = quat_rotate(box.rot, vec3(1, 0, 0));
	obb.v = quat_rotate(box.rot, vec3(0, 1, 0));
	obb.w = quat_rotate(box.rot, vec3(0, 0, 1));
	obb.center = box.center.xyz;
	obb.extent = box.halfLength.xyz;

	vec3 proj_seg = project(obb.center, seg0);
	vec3 proj_obb = project(proj_seg, obb);
	
	bool bInside = inside(proj_seg, obb);

	vec3 dir = bInside ? proj_seg - proj_obb : proj_obb - proj_seg;
	float sMax = bInside ? -length(dir) - r0 : length(dir) - r0;

	if(sMax >= 0)
		return;

	vec3 n = normalize(dir);

	m.normal = n;
	m.penetration[0] = sMax;
	m.position[0] = proj_obb;
	m.contactCount = 1;
}

#endif