
#define Real float
#define Edge ivec2
#define Triangle ivec3
#define Coord3D vec3
#define Quat vec4

#define REAL_EPSILON 1e-5
#define  REAL_EPSILON_SQUARED 1e-10

struct Ray3D
{
    Coord3D origin;
    Coord3D direction;
};

struct Segment3D {
    Coord3D v0;
    Coord3D v1;
};

struct Sphere3D {
	Coord3D center;
	Quat rotation;
	Real radius;
};

struct Plane3D {
    Coord3D origin;
    Coord3D normal;
};

struct Point3D {
    Coord3D origin;
};

struct Triangle3D {
    Coord3D v[3];
};


Sphere3D new_sphere3D(Coord3D center, Real radius) {
    return Sphere3D(center, Quat(0), max(radius, 0));
}

Triangle3D new_triangle3D(Coord3D v1, Coord3D v2, Coord3D v3) {
    Triangle3D t;
    t.v[0] = v1;
    t.v[1] = v2;
    t.v[2] = v3;
    return t;
}

Coord3D triangle_normal(Triangle3D self) {
    Coord3D n = cross((self.v[1] - self.v[0]), self.v[2] - self.v[0]);
    if (length(n) > REAL_EPSILON_SQUARED) {
        n = normalize(n);
    }
    return n;
}

Coord3D segment_direction(Segment3D s) {
    return s.v1 - s.v0;
}

Point3D point_project(Point3D p, Ray3D ray) {
    Coord3D u = p.origin - ray.origin;

    Real tNum = dot(u, ray.direction);
    Real a = dot(ray.direction, ray.direction);
    Real t = a < REAL_EPSILON_SQUARED ? 0 : tNum / a;

    t = t < 0 ? 0 : t;

    return Point3D(ray.origin + t * ray.direction);
}

Point3D point_project(Point3D p, Segment3D segment) {
    Coord3D l = p.origin - segment.v0;
    Coord3D dir = segment.v1 - segment.v0;
    if (dot(dir, dir) < REAL_EPSILON_SQUARED)
    {
        return Point3D(segment.v0);
    }

    Real t = dot(l, dir) / dot(dir, dir);

    Coord3D q = segment.v0 + t * dir;
    q = t < 0 ? segment.v0 : q;
    q = t > 1 ? segment.v1 : q;

    return Point3D(q);
}

Real point_distance(Point3D p, Segment3D segment) {
    return length(p.origin - point_project(p, segment).origin);
}

Real segment_length(Segment3D s) {
    return length(s.v1 - s.v0);
}

Real length2(Coord3D c) {
    return dot(c, c);
}

Real length2(Segment3D s) {
    return length2(s.v1 - s.v0);
}

Segment3D point_sub(Point3D p0, Point3D p1) {
    return Segment3D(p1.origin, p0.origin);
}

Segment3D ray_promixy(Ray3D ray, Segment3D segment) {
    Coord3D u = ray.origin - segment.v0;
    Coord3D dir1 = segment.v1 - segment.v0;
    Real a = dot(ray.direction, ray.direction);
    Real b = dot(ray.direction, dir1);
    Real c = dot(dir1, dir1);
    Real d = dot(u, ray.direction);
    Real e = dot(u, dir1);
    Real det = a * c - b * b;

    if (det < REAL_EPSILON)
    {
        if (a < REAL_EPSILON_SQUARED)
        {
            Point3D p0 = Point3D(ray.origin);
            return point_sub(point_project(p0, segment), p0);
        }
        else
        {
            Point3D p1 = Point3D(segment.v0);
            Point3D p2 = Point3D(segment.v1);

            Point3D q1 = point_project(p1, ray);
            Point3D q2 = point_project(p1, ray);

            return length2(point_sub(p1, q1)) < length2(point_sub(p2, q2)) ? point_sub(p1, q1) : point_sub(p2, q2);
        }
    }

    Real sNum = b * e - c * d;
    Real tNum = a * e - b * d;

    Real sDenom = det;
    Real tDenom = det;

    if (sNum < 0) {
        sNum = 0;
        tNum = e;
        tDenom = c;
    }

    // Check t
    if (tNum < 0) {
        tNum = 0;
        if (-d < 0) {
            sNum = 0;
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
        else {
            sNum = -d + b;
            sDenom = a;
        }
    }

    Real s = sNum / sDenom;
    Real t = tNum / tDenom;

    return Segment3D(ray.origin + (s * ray.direction), segment.v0 + (t * segment_direction(segment)));
}

Real ray_distance(Ray3D ray, Segment3D segment) {
    return segment_length(ray_promixy(ray, segment));
}

int ray_intersect(Ray3D ray, Sphere3D sphere, inout Segment3D interSeg)
{
    Coord3D diff = ray.origin - sphere.center;
    Real a0 = dot(diff, diff) - sphere.radius * sphere.radius;
    Real a1 = dot(ray.direction, diff);

    // Intersection occurs when Q(t) has real roots.
    Real discr = a1 * a1 - a0;
    if (discr > 0.0)
    {
        Real root = sqrt(discr);


        if (-a1 + root < 0.0)
        {
            return 0;
        }
        else if (-a1 - root < 0.0)
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
    else if (discr < 0.0)
    {
        return 0;
    }
    else
    {
        if (a1 > 0.0)
        {
            return 0;
        }
        interSeg.v0 = ray.origin - a1 * ray.direction;
        return 1;
    }
}

int ray_intersect(in Ray3D ray, in Triangle3D triangle, inout Point3D interPt) {
    Coord3D diff = ray.origin - triangle.v[0];
    Coord3D e0 = triangle.v[1] - triangle.v[0];
    Coord3D e1 = triangle.v[2] - triangle.v[0];
    Coord3D normal = cross(e0, e1);

    Real DdN = dot(ray.direction, normal);
    Real sign;
    if (DdN >= REAL_EPSILON)
    {
        sign = Real(1);
    }
    else if (DdN <= -REAL_EPSILON)
    {
        sign = Real(-1);
        DdN = -DdN;
    }
    else
    {
        return 0;
    }

    Real DdQxE1 = sign * dot(ray.direction, cross(diff, e1));
    if (DdQxE1 >= Real(0))
    {
        Real DdE0xQ = sign * dot(ray.direction, cross(e0, diff));
        if (DdE0xQ >= Real(0))
        {
            if (DdQxE1 + DdE0xQ <= DdN)
            {
                // Line intersects triangle.
                Real QdN = -sign * dot(diff, normal);
                Real inv = Real(1) / DdN;

                Real t = QdN * inv;

                if (t < Real(0))
                {
                    return 0;
                }

                interPt.origin = ray.origin + t * ray.direction;
                return 1;
            }
            // else: b1+b2 > 1, no intersection
        }
        // else: b2 < 0, no intersection
    }
    // else: b1 < 0, no intersection

    return 0;
}

bool segment_intersect(in Segment3D seg, in Plane3D plane, inout Point3D interPt) {
    Coord3D dir = segment_direction(seg);
    Real DdN = dot(dir, plane.normal);
    if (abs(DdN) < REAL_EPSILON)
    {
        return false;
    }

    Coord3D offset = seg.v0 - plane.origin;
    Real t = dot(-offset, plane.normal) / DdN;

    if (t < Real(0) || t > Real(1))
    {
        return false;
    }

    interPt.origin = seg.v0 + t * dir;
    return true;
}