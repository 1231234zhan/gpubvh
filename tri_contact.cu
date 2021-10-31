
#include "tri_contact.cuh"

#ifndef M_PI
#define M_PI 3.14159f
#endif

__device__ inline vec3f operator*(double t, const vec3f& v)
{
    return vec3f(v.x * t, v.y * t, v.z * t);
}
__device__ inline vec3f interp(const vec3f& a, const vec3f& b, double t)
{
    return a * (1 - t) + b * t;
}
__device__ inline vec3f vinterp(const vec3f& a, const vec3f& b, double t)
{
    return a * t + b * (1 - t);
}
__device__ inline vec3f interp(const vec3f& a, const vec3f& b, const vec3f& c, double u, double v, double w)
{
    return a * u + b * v + c * w;
}
__device__ inline double clamp(double f, double a, double b)
{
    return fmax(a, fmin(f, b));
}
__device__ inline double vdistance(const vec3f& a, const vec3f& b)
{
    return (a - b).length();
}

// inline std::ostream& operator<<(std::ostream&os, const vec3f &v) {
// 	os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
// 	return os;
// }

__device__ inline void
vmin(vec3f& a, const vec3f& b)
{
    a.set_value(
        fmin(a[0], b[0]),
        fmin(a[1], b[1]),
        fmin(a[2], b[2]));
}

__device__ inline void
vmax(vec3f& a, const vec3f& b)
{
    a.set_value(
        fmax(a[0], b[0]),
        fmax(a[1], b[1]),
        fmax(a[2], b[2]));
}

__device__ inline vec3f lerp(const vec3f& a, const vec3f& b, float t)
{
    return a + t * (b - a);
}

__device__ inline double fmax(double a, double b, double c)
{
    double t = a;
    if (b > t)
        t = b;
    if (c > t)
        t = c;
    return t;
}

__device__ inline double fmin(double a, double b, double c)
{
    double t = a;
    if (b < t)
        t = b;
    if (c < t)
        t = c;
    return t;
}

__device__ inline int project3(const vec3f& ax,
    const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
    double P1 = ax.dot(p1);
    double P2 = ax.dot(p2);
    double P3 = ax.dot(p3);

    double mx1 = fmax(P1, P2, P3);
    double mn1 = fmin(P1, P2, P3);

    if (mn1 > 0)
        return 0;
    if (0 > mx1)
        return 0;
    return 1;
}

__device__ inline int project6(vec3f& ax,
    vec3f& p1, vec3f& p2, vec3f& p3,
    vec3f& q1, vec3f& q2, vec3f& q3)
{
    double P1 = ax.dot(p1);
    double P2 = ax.dot(p2);
    double P3 = ax.dot(p3);
    double Q1 = ax.dot(q1);
    double Q2 = ax.dot(q2);
    double Q3 = ax.dot(q3);

    double mx1 = fmax(P1, P2, P3);
    double mn1 = fmin(P1, P2, P3);
    double mx2 = fmax(Q1, Q2, Q3);
    double mn2 = fmin(Q1, Q2, Q3);

    if (mn1 > mx2)
        return 0;
    if (mn2 > mx1)
        return 0;
    return 1;
}

__device__ bool
tri_contact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3)
{
    vec3f p1;
    vec3f p2 = P2 - P1;
    vec3f p3 = P3 - P1;
    vec3f q1 = Q1 - P1;
    vec3f q2 = Q2 - P1;
    vec3f q3 = Q3 - P1;

    vec3f e1 = p2 - p1;
    vec3f e2 = p3 - p2;
    vec3f e3 = p1 - p3;

    vec3f f1 = q2 - q1;
    vec3f f2 = q3 - q2;
    vec3f f3 = q1 - q3;

    vec3f n1 = e1.cross(e2);
    vec3f m1 = f1.cross(f2);

    vec3f g1 = e1.cross(n1);
    vec3f g2 = e2.cross(n1);
    vec3f g3 = e3.cross(n1);

    vec3f h1 = f1.cross(m1);
    vec3f h2 = f2.cross(m1);
    vec3f h3 = f3.cross(m1);

    vec3f ef11 = e1.cross(f1);
    vec3f ef12 = e1.cross(f2);
    vec3f ef13 = e1.cross(f3);
    vec3f ef21 = e2.cross(f1);
    vec3f ef22 = e2.cross(f2);
    vec3f ef23 = e2.cross(f3);
    vec3f ef31 = e3.cross(f1);
    vec3f ef32 = e3.cross(f2);
    vec3f ef33 = e3.cross(f3);

    // now begin the series of tests
    if (!project3(n1, q1, q2, q3))
        return false;
    if (!project3(m1, -q1, p2 - q1, p3 - q1))
        return false;

    if (!project6(ef11, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef12, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef13, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef21, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef22, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef23, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef31, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef32, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(ef33, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g2, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(g3, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h1, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h2, p1, p2, p3, q1, q2, q3))
        return false;
    if (!project6(h3, p1, p2, p3, q1, q2, q3))
        return false;

    return true;
}