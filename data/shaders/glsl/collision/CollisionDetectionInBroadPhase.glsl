#ifndef COLLISIONDETECTIONINBROADPHASE_H
#define COLLISIONDETECTIONINBROADPHASE_H
#include "../rigidbody/SharedData.glsl"
#include "../math/Quat.glsl"

bool checkCollision(AlignedBox3D b0, AlignedBox3D b1)
{
    bool not_overlap = false;
    not_overlap = not_overlap || b0.v0.x > b1.v1.x || b0.v1.x < b1.v0.x;
    not_overlap = not_overlap || b0.v0.y > b1.v1.y || b0.v1.y < b1.v0.y;
    not_overlap = not_overlap || b0.v0.z > b1.v1.z || b0.v1.z < b1.v0.z;

    return !not_overlap;
}

bool filterCollision(AlignedBox3D b0, AlignedBox3D b1, CollisionType cType0, CollisionType cType1, ShapeType sType0, ShapeType sType1)
{
	bool canCollide = (cType0 & sType1) > 0 && (cType1 & sType0) > 0;
	if(!canCollide)
		return false;

    bool not_overlap = false;
    not_overlap = not_overlap || b0.v0.x > b1.v1.x || b0.v1.x < b1.v0.x;
    not_overlap = not_overlap || b0.v0.y > b1.v1.y || b0.v1.y < b1.v0.y;
    not_overlap = not_overlap || b0.v0.z > b1.v1.z || b0.v1.z < b1.v0.z;

    return !not_overlap;
}

#endif