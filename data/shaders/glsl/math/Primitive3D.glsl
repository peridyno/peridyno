#ifndef PRIMITIVE3D_H
#define PRIMITIVE3D_H

#define EPSILON 0.000001
#define FLOAT_MAX 1000000

struct Plane3D
{
    vec3 origin;
    vec3 normal;
};

struct IPoint
{
    vec4 point;
    uint id;

    int pad[3];
};

struct Line3D
{
    vec3 origin;
    vec3 direction;
};

struct Ray3D
{
    vec3 origin;
    vec3 direction;
};

struct Segment3D
{
    vec3 v0;
    vec3 v1;
};

struct AABB3D
{
    vec3 v0;
    vec3 v1;
};

struct Sphere3D
{
    float radius;
	vec3 center;
};

struct OBB3D
{
    vec3 center;

    /**
    * @brief three unit vectors u, v and w forming a right-handed orthornormal basis
    *
    */
    vec3 u;
    vec3 v;
    vec3 w;

    /**
    * @brief half the dimension in each of the u, v, and w directions
    */
    vec3 extent;
};

// vec3 clamp(vec3 v, vec3 lo, vec3 hi)
// {
//     vec3 ret;
//     ret[0] = (v[0] < lo[0]) ? lo[0] : (hi[0] < v[0]) ? hi[0] : v[0];
//     ret[1] = (v[1] < lo[1]) ? lo[1] : (hi[1] < v[1]) ? hi[1] : v[1];
//     ret[2] = (v[2] < lo[2]) ? lo[2] : (hi[2] < v[2]) ? hi[2] : v[2];

//     return ret;
// }

bool inside(vec3 p, AABB3D box)
{
    vec3 offset = p - box.v0;
    vec3 extent = box.v1 - box.v0;

    bool bInside = true;
    bInside = bInside && offset.x < extent.x && offset.x > 0;
    bInside = bInside && offset.y < extent.y && offset.y > 0;
    bInside = bInside && offset.z < extent.z && offset.z > 0;

    return bInside;
}

bool inside(vec3 p, OBB3D obb)
{
    vec3 offset = p - obb.center;
    vec3 pPrime = vec3(dot(offset, obb.u), dot(offset, obb.v), dot(offset, obb.w));

    bool bInside = true;
    bInside = bInside && pPrime.x < obb.extent.x && pPrime.x >  -obb.extent.x;
    bInside = bInside && pPrime.y < obb.extent.y && pPrime.y >  -obb.extent.y;
    bInside = bInside && pPrime.z < obb.extent.z && pPrime.z >  -obb.extent.z;
    return bInside;
}

Segment3D makeSegment3D(vec3 v0, vec3 v1)
{
    Segment3D seg;
    seg.v0 = v0;
    seg.v1 = v1;

    return seg;
}

Segment3D opposite(Segment3D seg)
{
    Segment3D ret;
    ret.v0 = seg.v1;
    ret.v1 = seg.v0;

    return ret;
}

vec3 project(vec3 p, Segment3D seg)
{
    vec3 l = p - seg.v0;
    vec3 dir = seg.v1 - seg.v0;
    float squareLength = dot(dir, dir);
    if (squareLength < EPSILON*EPSILON)
    {
        return seg.v0;
    }

    float t = dot(l, dir) / squareLength;

    vec3 q = seg.v0 + t * dir;
    q = t < 0 ? seg.v0 : q;
    q = t > 1 ? seg.v1 : q;
    //printf("T: %.3lf\n", t);
    return q;
}

vec3 project(vec3 p, AABB3D abox)
{
    bool bInside = inside(p, abox);

    if (!bInside)
    {
        return clamp(p, abox.v0, abox.v1);
    }

    //compute the distance to six faces
    vec3 q;
    float minDist = FLOAT_MAX;
    vec3 offset0 = abs(p - abox.v0);
    if (offset0.x < minDist)
    {
        q = vec3(abox.v0.x, p.y, p.z);
        minDist = offset0.x;
    }

    if (offset0.y < minDist)
    {
        q = vec3(p.x, abox.v0.y, p.z);
        minDist = offset0.y;
    }

    if (offset0.z < minDist)
    {
        q = vec3(p.x, p.y, abox.v0.z);
        minDist = offset0.z;
    }


    vec3 offset1 = abs(p - abox.v1);
    if (offset1.x < minDist)
    {
        q = vec3(abox.v1.x, p.y, p.z);
        minDist = offset1.x;
    }

    if (offset1.y < minDist)
    {
        q = vec3(p.x, abox.v1.y, p.z);
        minDist = offset1.y;
    }

    if (offset1[2] < minDist)
    {
        q = vec3(p.x, p.y, abox.v1.z);
        minDist = offset1.z;
    }

    return q;
}

vec3 project(vec3 p, OBB3D obb)
{
    vec3 offset = p - obb.center;
    vec3 pPrime = vec3(dot(offset, obb.u), dot(offset, obb.v), dot(offset, obb.w));

    AABB3D abox;
    abox.v0 = -obb.extent;
    abox.v1 = obb.extent;
    vec3 qPrime = project(pPrime, abox);

    return obb.center + qPrime.x * obb.u + qPrime.y * obb.v + qPrime.z * obb.w;
}

Segment3D proximity(Segment3D seg0, Segment3D seg1)
{
    vec3 u = seg0.v0 - seg1.v0;
    vec3 dir0 = seg0.v1 - seg0.v0;
    vec3 dir1 = seg1.v1 - seg1.v0;
    float a = dot(dir0, dir0);
    float b = dot(dir0, dir1);
    float c = dot(dir1, dir1);
    float d = dot(u, dir0);
    float e = dot(u, dir1);
    float det = a * c - b * b;

    // Check for (near) parallelism
    if (det < EPSILON) {
        // Arbitrary choice
        vec3 p0 = a < b ? seg0.v0 : seg1.v0;
        vec3 p1 = a < b ? seg0.v1 : seg1.v1;
        Segment3D longerSeg = a < b ? seg1 : seg0;
        bool bOpposite = a < b ? false : true;
        vec3 q0 = project(p0, longerSeg);
        vec3 q1 = project(p1, longerSeg);
        Segment3D ret = length(p0 - q0) < length(p1 - q1) ? makeSegment3D(p0, q0) : makeSegment3D(p1, q1);
        return bOpposite ? opposite(ret) : ret;
    }

    // Find parameter values of closest points
    // on each segment��s infinite line. Denominator
    // assumed at this point to be ����det����,
    // which is always positive. We can check
    // value of numerators to see if we��re outside
    // the [0, 1] x [0, 1] domain.
    float sNum = b * e - c * d;
    float tNum = a * e - b * d;

    float sDenom = det;
    float tDenom = det;

    if (sNum < 0) {
        sNum = 0;
        tNum = e;
        tDenom = c;
    }
    else if (sNum > det) {
        sNum = det;
        tNum = e + b;
        tDenom = c;
    }

    // Check t
    if (tNum < 0) {
        tNum = 0;
        if (-d < 0) {
            sNum = 0;
        }
        else if (-d > a) {
            sNum = sDenom;
        }
        else {
            sNum = -d;
            sDenom = a;
        }
    }
    else if (tNum > tDenom) {
        tNum = tDenom;
        if ((-d + b) < 0) {
            sNum = 0;
        }
        else if ((-d + b) > a) {
            sNum = sDenom;
        }
        else {
            sNum = -d + b;
            sDenom = a;
        }
    }

    float s = sNum / sDenom;
    float t = tNum / tDenom;

    return makeSegment3D(seg0.v0 + (s * dir0), seg1.v0 + (t * dir1));
}

bool clip(float denom, float numer, inout float t0, inout float t1)
{
    if (denom > EPSILON)
    {
        if (numer > denom * t1)
        {
            return false;
        }
        if (numer > denom * t0)
        {
            t0 = numer / denom;
        }
        return true;
    }
    else if (denom < -EPSILON)
    {
        if (numer > denom * t0)
        {
            return false;
        }
        if (numer > denom * t1)
        {
            t1 = numer / denom;
        }
        return true;
    }
    else
    {
        return numer <= -EPSILON;
    }
}

int intersect(inout Segment3D interSeg, inout float t0, inout float t1, Line3D line, AABB3D abox)
{
    if (dot(line.direction, line.direction) < EPSILON)
    {
        return 0;
    }

    t0 = -FLOAT_MAX;
    t1 = FLOAT_MAX;

    vec3 boxCenter = 0.5 * (abox.v0 + abox.v1);
    vec3 boxExtent = 0.5 * (abox.v1 - abox.v0);

    vec3 offset = line.origin - boxCenter;
    vec3 lineDir = normalize(line.direction);

    if (clip(+lineDir.x, -offset.x - boxExtent.x, t0, t1) &&
        clip(-lineDir.x, +offset.x - boxExtent.x, t0, t1) &&
        clip(+lineDir.y, -offset.y - boxExtent.y, t0, t1) &&
        clip(-lineDir.y, +offset.y - boxExtent.y, t0, t1) &&
        clip(+lineDir.z, -offset.z - boxExtent.z, t0, t1) &&
        clip(-lineDir.z, +offset.z - boxExtent.z, t0, t1))
    {
        if (t1 > t0)
        {
            interSeg.v0 = line.origin + t0 * lineDir;
            interSeg.v1 = line.origin + t1 * lineDir;
            return 2;
        }
        else
        {
            interSeg.v0 = line.origin + t0 * lineDir;
            interSeg.v1 = interSeg.v0;
            return 1;
        }
    }

    return 0;
}

int intersect(inout Segment3D interSeg, Ray3D ray, AABB3D abox)
{
    Line3D line;
    line.origin = ray.origin;
    line.direction = ray.direction;
    float t0, t1;
    int interNum = intersect(interSeg, t0, t1, line, abox);
    if (interNum == 0)
    {
        return 0;
    }

    if (t0 > 0)
    {
        interSeg.v0 = ray.origin + t0 * ray.direction;
        interSeg.v1 = ray.origin + t1 * ray.direction;
        return 2;
    }
    else if (t1 > 0)
    {
        interSeg.v0 = ray.origin;
        interSeg.v1 = ray.origin + t1 * ray.direction;
        return 1;
    }
    else
    {
        return 0;
    }
}

int intersect(inout Segment3D interSeg, Ray3D ray, Sphere3D sphere)
{
    vec3 diff = ray.origin - sphere.center;
    float a0 = dot(diff, diff) - sphere.radius * sphere.radius;
    float a1 = dot(ray.direction, diff);

    // Intersection occurs when Q(t) has real roots.
    float discr = a1 * a1 - a0;
    if (discr > 0)
    {
        float root = sqrt(discr);

        if (-a1 + root < 0)
        {
            return 0;
        }
        else if (-a1 + root < 0)
        {
            interSeg.v0 = ray.origin + (-a1 + root) * ray.direction;
            return 1;
        }
        else
        {
            interSeg.v0 = ray.origin + (-a1 - root) * ray.direction;
            interSeg.v1 = ray.origin + (-a1 + root) * ray.direction;
            return 2;
        }
    }
    else if (discr < 0)
    {
        return 0;
    }
    else
    {
        if (a1 > 0)
        {
            return 0;
        }
        interSeg.v0 = ray.origin - a1 * ray.direction;
        return 1;
    }
}

#endif