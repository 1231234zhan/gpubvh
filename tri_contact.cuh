
#ifndef TRI_CONTACT_CUH
#define TRI_CONTACT_CUH

#define GLH_ZERO double(0.0)
#define GLH_EPSILON double(10e-6)
#define GLH_EPSILON_2 double(10e-12)
#define equivalent(a, b) (((a < b + GLH_EPSILON) && (a > b - GLH_EPSILON)) ? true : false)

__device__ inline double lerp(double a, double b, float t)
{
    return a + t * (b - a);
}
__device__ inline double fmax(double a, double b)
{
    return (a > b) ? a : b;
}
__device__ inline double fmin(double a, double b)
{
    return (a < b) ? a : b;
}
__device__ inline bool isEqual(double a, double b, double tol = GLH_EPSILON)
{
    return fabs(a - b) < tol;
}

class vec3f {
public:
    union {
        struct {
            double x, y, z;
        };
        struct {
            double v[3];
        };
    };

    __device__ inline vec3f()
    {
        x = 0;
        y = 0;
        z = 0;
    }

    __device__ inline vec3f(const vec3f& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    __device__ inline vec3f(const double* v)
    {
        x = v[0];
        y = v[1];
        z = v[2];
    }

    __device__ inline vec3f(double x, double y, double z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    __device__ inline double operator[](int i) const { return v[i]; }
    __device__ inline double& operator[](int i) { return v[i]; }

    __device__ inline vec3f& operator+=(const vec3f& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __device__ inline vec3f& operator-=(const vec3f& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    __device__ inline vec3f& operator*=(double t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    __device__ inline vec3f& operator/=(double t)
    {
        x /= t;
        y /= t;
        z /= t;
        return *this;
    }

    __device__ inline void negate()
    {
        x = -x;
        y = -y;
        z = -z;
    }

    __device__ inline vec3f operator-() const
    {
        return vec3f(-x, -y, -z);
    }

    __device__ inline vec3f operator+(const vec3f& v) const
    {
        return vec3f(x + v.x, y + v.y, z + v.z);
    }

    __device__ inline vec3f operator-(const vec3f& v) const
    {
        return vec3f(x - v.x, y - v.y, z - v.z);
    }

    __device__ inline vec3f operator*(double t) const
    {
        return vec3f(x * t, y * t, z * t);
    }

    __device__ inline vec3f operator/(double t) const
    {
        return vec3f(x / t, y / t, z / t);
    }

    // cross product
    __device__ inline const vec3f cross(const vec3f& vec) const
    {
        return vec3f(y * vec.z - z * vec.y, z * vec.x - x * vec.z, x * vec.y - y * vec.x);
    }

    __device__ inline double dot(const vec3f& vec) const
    {
        return x * vec.x + y * vec.y + z * vec.z;
    }

    __device__ inline void normalize()
    {
        double sum = x * x + y * y + z * z;
        if (sum > GLH_EPSILON_2) {
            double base = double(1.0 / sqrt(sum));
            x *= base;
            y *= base;
            z *= base;
        }
    }

    __device__ inline double length() const
    {
        return double(sqrt(x * x + y * y + z * z));
    }

    __device__ inline vec3f getUnit() const
    {
        return (*this) / length();
    }

    __device__ inline bool isUnit() const
    {
        return isEqual(squareLength(), 1.f);
    }

    //! max(|x|,|y|,|z|)
    __device__ inline double infinityNorm() const
    {
        return fmax(fmax(fabs(x), fabs(y)), fabs(z));
    }

    __device__ inline vec3f& set_value(const double& vx, const double& vy, const double& vz)
    {
        x = vx;
        y = vy;
        z = vz;
        return *this;
    }

    __device__ inline bool equal_abs(const vec3f& other)
    {
        return x == other.x && y == other.y && z == other.z;
    }

    __device__ inline double squareLength() const
    {
        return x * x + y * y + z * z;
    }

    __device__ static vec3f zero()
    {
        return vec3f(0.f, 0.f, 0.f);
    }

    //! Named constructor: retrieve vector for nth axis
    __device__ static vec3f axis(int n)
    {
        // assert(n < 3);
        switch (n) {
        case 0: {
            return xAxis();
        }
        case 1: {
            return yAxis();
        }
        case 2: {
            return zAxis();
        }
        }
        return vec3f();
    }

    //! Named constructor: retrieve vector for x axis
    __device__ static vec3f xAxis() { return vec3f(1.f, 0.f, 0.f); }
    //! Named constructor: retrieve vector for y axis
    __device__ static vec3f yAxis() { return vec3f(0.f, 1.f, 0.f); }
    //! Named constructor: retrieve vector for z axis
    __device__ static vec3f zAxis() { return vec3f(0.f, 0.f, 1.f); }
};

__device__ bool tri_contact(vec3f&, vec3f&, vec3f&, vec3f&, vec3f&, vec3f&);

#endif