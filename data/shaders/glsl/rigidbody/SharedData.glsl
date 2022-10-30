#ifndef COMMON_H
#define COMMON_H

#include "../math/Primitive3D.glsl"

#define GROUND_LELEL 0.0

#define Static 		0
#define Kinematic	1
#define Dynamic		2

#define INVALID_ID 	-1

#define CT_BOUNDARY 		0
#define CT_NONPENETRATION 	1
#define CT_FRICTION 		2

#define CM_AllObjects  0xFFFFFFFF
#define CM_BoxExcluded 0xFFFFFFFE
#define CM_TetExcluded 0xFFFFFFFD
#define CM_SegmentExcluded 0xFFFFFFFA
#define CM_SphereExcluded 0xFFFFFFF7
#define CM_BoxOnly 0x00000001
#define CM_TetOnly 0x00000002
#define CM_SegmentOnly 0x00000004
#define CM_SphereOnly 0x00000008
#define CM_Disabled 0x00000000

#define ST_Box  0x80000001
#define ST_Tet  2
#define ST_Segement  4
#define ST_Sphere  8
#define ST_Other  0x80000000

#define BETA 0.25

#define SPLIT_MASS 0.2

struct ContactInfo
{
	uint contactNum;
	uint constraintNum;
};

struct Box
{
	vec4 center;
	vec4 halfLength;
	vec4 rot;
};

struct Capsule
{
	vec4 rot;
	vec4 center;
	float halfLength;
	float radius;
};

struct Sphere
{
	vec4 rot;
	vec3 center;
	float radius;
};

struct AlignedBox3D
{
	vec3 v0;
	vec3 v1;
};

#define CollisionType uint
#define ShapeType uint

struct BodyDef
{
	vec4 quat;

	/// The linear velocity of the body's origin in world co-ordinates.
	vec4 linearVelocity;

	/// The angular velocity of the body.
	vec4 angularVelocity;

	/// The impulse applied to the body's origin in world co-ordinates.
	vec4 extForce;

	/// The moment of impulse applied to the body.
	vec4 extTorque;

	/// The (constant) acceleration applied to the body's origin in world co-ordinates.
	vec4 gravity;

	/// The world position of the body.
	vec4 position;

	mat3 inertia;

	float mass;

	float friction;

	float restitution;

	uint type;

	uint nJoints;
	CollisionType cType;
	ShapeType sType;

	int padding[2];
};

struct ContactPair
{
	vec4 pos0;
	vec4 pos1;

	vec4 normal0;
	vec4 normal1;

	int id0;
	int id1;

	uint cType;
	float distance;
};

struct ConstraintPair
{
	vec4 r0;
	vec4 r00;
	vec4 r1;
	vec4 r11;

	vec4 n00;
	vec4 n10;

	vec4 n01;
	vec4 n11;

	vec4 lowerPos;
	vec4 upperPos;

	vec4 lowerAng;
	vec4 upperAng;

	vec4 angle;
	vec4 frame0;
	vec4 frame1;

	int id0;
	int id1;

	uint posDOF;
	uint rotDOF;

	float distance;
};

struct RigidBodySolverState
{
	vec4 gravity;
	float dt;
	float erp;
};

struct ElementOffset
{
	uint box_bound;
	uint capsule_bound;
	uint sphere_bound;
};

#endif
