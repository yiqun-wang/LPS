/************************************************************
* This file is MODIFIED from a part of the DGPC library to
* help it support to read off model files.
*
* The library computes Discrete Geodesic Polar Coordinates
* on a polygonal mesh.
*
* Author: Hanyu Wang(王涵玉)
* DGPC file's authors: Eivind Lyche Melvær and Martin Reimers
************************************************************/

#ifndef VECTOR3_H
#define VECTOR3_H
#include <iostream>
#include <string>
#include <cmath>

namespace GIGen {
    template<class real_t>
    class Vector3 {
        real_t p_[3];
        void set(real_t x, real_t y, real_t z) {
            p_[0] = x;
            p_[1] = y;
            p_[2] = z;
        }
    public:
        typedef real_t value_type;
        typedef Vector3<real_t> vector_type;

        static const size_t size_ = 3;

        Vector3() {
            p_[0] = p_[1] = p_[2] = std::numeric_limits<real_t>::max();
        }

        Vector3(real_t x, real_t y, real_t z) {
            set(x, y, z);
        }

        Vector3(const real_t v[]) {
            set(v[0], v[1], v[2]);
        }

        Vector3(const Vector3<real_t>& p) {
            set(p.x(), p.y(), p.z());
        }

        const real_t& x() const { return p_[0]; };
        real_t& x() { return p_[0]; };
        const real_t& y() const { return p_[1]; };
        real_t& y() { return p_[1]; };
        const real_t& z() const { return p_[2]; };
        real_t& z() { return p_[2]; };

        const real_t& operator [] (int i) const { return p_[i]; };
        real_t& operator [] (int i) { return p_[i]; };

        Vector3<real_t>& operator= (const Vector3<real_t>& p) {
            set(p.x(), p.y(), p.z());
            return *this;
        }

        Vector3<real_t> operator- (const Vector3<real_t>& v) const {
            return Vector3<real_t>(x() - v.x(), y() - v.y(), z() - v.z());
        }

        Vector3<real_t> operator+ (const Vector3<real_t>& v) const {
            return Vector3<real_t>(x() + v.x(), y() + v.y(), z() + v.z());
        }

        real_t operator* (const Vector3<real_t>& v) const {
            return x()*v.x() + y()*v.y() + z()*v.z();
        }

        Vector3<real_t> operator* (real_t d) const {
            return Vector3<real_t>(x()*d, y()*d, z()*d);
        }

        // Hanyu's code
        Vector3<real_t> operator/ (real_t d) const
        {
            return Vector3<real_t>(x() / d, y() / d, z() / d);
        }

        real_t dist(const Vector3<real_t>& v) const {
            return std::sqrt(dist2(v));
        }

        real_t dist2(const Vector3<real_t>& v) const {
            real_t dx = x() - v.x();
            real_t dy = y() - v.y();
            real_t dz = z() - v.z();
            return dx*dx + dy*dy + dz*dz;
        }

        real_t length() const {
            return std::sqrt(length2());
        }

        real_t length2() const {
            return x()*x() + y()*y() + z()*z();
        }

        Vector3<real_t> crossProd(const Vector3<real_t>& v) const {
            return Vector3(y()*v.z() - z()*v.y(),
                z()*v.x() - x()*v.z(),
                x()*v.y() - y()*v.x());
        }

        Vector3<real_t>& normalize() {
            const real_t len2 = length2();
            if (len2) {
                const real_t len = std::sqrt(len2);
                p_[0] /= len;
                p_[1] /= len;
                p_[2] /= len;
            }
            return *this;
        }

        //// Hanyu's code
        //template<typename real_t_>
        //friend ostream& operator<< (ostream& out, const Vector3<real_t_>& s);

        // Hanyu's code
        std::string to_string() const
        {
            std::ostringstream oss;
            oss << "x: " << p_[0] << ", y: " << p_[1] << ", z: " << p_[2];
            return oss.str();
        }

    };

    template<class real_t>
    Vector3<real_t> operator* (real_t d, const Vector3<real_t>& v)
    {
        return v*d;
    }


    //// Hanyu's code
    //template<typename real_t_>
    //ostream& operator<< (ostream& out, const Vector3<real_t_>& s)
    //{
    //	out << "x: " << s[0] << ", y: " << s[1] << ", z: " << s[2];
    //	return out;
    //}


    // Hanyu's code
    template<typename real_t>
    class Rot_mat
    {
    private:
        real_t r_m[3][3];

    public:

        Rot_mat(const real_t &rot_x, const real_t &rot_y, const real_t &rot_z, const real_t &rot_theta)
        {
            double &&vec_norm = std::sqrt(rot_x * rot_x + rot_y * rot_y + rot_z * rot_z);

            real_t th = rot_theta;

            real_t &&x = rot_x / vec_norm;
            real_t &&y = rot_y / vec_norm;
            real_t &&z = rot_z / vec_norm;

            this->r_m[0][0] = std::cos(th) + (1 - std::cos(th)) * x * x;
            this->r_m[0][1] = (1 - std::cos(th)) * x * y - std::sin(th) * z;
            this->r_m[0][2] = (1 - std::cos(th)) * x * z + std::sin(th) * y;

            this->r_m[1][0] = (1 - std::cos(th)) * y * z + std::sin(th) * z;
            this->r_m[1][1] = std::cos(th) + (1 - std::cos(th)) * y * y;
            this->r_m[1][2] = (1 - std::cos(th)) * y * z - std::sin(th) * x;

            this->r_m[2][0] = (1 - std::cos(th)) * z * x - std::sin(th) * y;
            this->r_m[2][1] = (1 - std::cos(th)) * z * y + std::sin(th) * x;
            this->r_m[2][2] = std::cos(th) + (1 - std::cos(th)) * z * z;
        }

        Rot_mat(const Vector3<real_t> &rot_axis, const real_t &rot_theta) :
            Rot_mat(rot_axis.x(), rot_axis.y(), rot_axis.z(), rot_theta) {}

        Vector3<real_t> operator* (const Vector3<real_t>& v) const
        {
            return Vector3<real_t>
                (
                    this->r_m[0][0] * v.x() + this->r_m[0][1] * v.y() + this->r_m[0][2] * v.z(),
                    this->r_m[1][0] * v.x() + this->r_m[1][1] * v.y() + this->r_m[1][2] * v.z(),
                    this->r_m[2][0] * v.x() + this->r_m[2][1] * v.y() + this->r_m[2][2] * v.z()
                    );
        }

    };


} //End namespace GIGen

#endif //VECTOR3_H
