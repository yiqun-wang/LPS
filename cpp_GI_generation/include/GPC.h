/************************************************************
* This file is MODIFIED from a part of the 3D descriptor 
* learning framework by Hanyu Wang(王涵玉)
* https://github.com/jianweiguo/local3Ddescriptorlearning
*
* Author: Yiqun Wang(王逸群)
* https://github.com/yiqun-wang/LPS
************************************************************/

#ifndef GPC_H
#define GPC_H

#include <iostream>
#include <iomanip>
#include "Generator.h"
#include "Mesh_C.h"
#include "Vector3.h"
#include "nanoflann.hpp"
#include <cmath>
#include <limits>
#include <iterator>
#include <memory>
#include <map>
#include <vector>
#include <set>

namespace GIGen
{
    using Point = Vector3<double>;

    // Basic 2D point, default initialized by rectangular coordinate.
    class Point_2
    {
    private:
        double x_, y_, r_, theta_;

    public:
        Point_2(const double &coor_0, const double &coor_1, const bool &use_polar_coor = false) // (x, y) or (r, theta)
        {
            if (!use_polar_coor)
            {
                x_ = coor_0;
                y_ = coor_1;
                r_ = std::sqrt(coor_0 * coor_0 + coor_1 * coor_1);
                theta_ = std::atan2(coor_1, coor_0);
            }
            else
            {
                x_ = coor_0 * std::cos(coor_1);
                y_ = coor_0 * std::sin(coor_1);
                r_ = coor_0;
                theta_ = coor_1;
            }
        }

        Point_2() {}

        double x() const
        {
            return x_;
        }

        double y() const
        {
            return y_;
        }

        double r() const
        {
            return r_;
        }

        double theta() const
        {
            return theta_;
        }
    };


    // 2D Point with attributes, default initialized by polar coordinate.
    class Attr_Point_2 : public Point_2
    {
    private:
        const unsigned int idx_;

        std::map<std::string, double> scalar_attr_;
        std::map<std::string, Point> vec3_attr_;
        std::map<std::string, std::vector<double>> stdvec_attr_;
    public:
        Attr_Point_2(const unsigned int &idx_, const double &coor_0, const double &coor_1, const bool &use_polar_coor = true) :
            idx_(idx_), Point_2(coor_0, coor_1, use_polar_coor) {}


        unsigned int idx() const
        {
            return this->idx_;
        }

        auto scalar_attr() const
        {
            return this->scalar_attr_;
        }

        auto vec3_attr() const
        {
            return this->vec3_attr_;
        }

        auto stdvec_attr() const
        {
            return this->stdvec_attr_;
        }

        //int attr_vec_length() const
        //{
        //    return scalar_attr_.size() + vec3_attr_.size() * 3;
        //}

        void get_attr(const std::string &attr_name, double &attr_val) const
        {
            attr_val = this->scalar_attr_.at(attr_name);
        }

        void get_attr(const std::string &attr_name, Point &attr_val) const
        {
            attr_val = this->vec3_attr_.at(attr_name);
        }

        void get_attr(const std::string &attr_name, std::vector<double> &attr_val) const
        {
            attr_val = this->stdvec_attr_.at(attr_name);
        }

        void set_attr(const std::string &attr_name, const double &attr_val)
        {
            this->scalar_attr_[attr_name] = attr_val;
        }

        void set_attr(const std::string &attr_name, const Point &attr_val)
        {
            this->vec3_attr_[attr_name] = attr_val;
        }

        void set_attr(const std::string &attr_name, const std::vector<double> &attr_val)
        {
            this->stdvec_attr_[attr_name] = attr_val;
        }

        std::string to_string_all() const
        {
            std::ostringstream oss;
            oss << "idx: " << std::setw(5) << this->idx_ << std::setprecision(6)
                << "\tr: " << this->r() << "\ttheta: " << this->theta()
                << "\tx: " << this->x() << "\ty: " << this->y();

            oss << std::endl;

            for (auto &p : this->scalar_attr_)
            {
                oss << "\t" << p.first << ": " << p.second;
            }

            oss << std::endl;

            for (auto &p : this->vec3_attr_)
            {
                oss << "\t" << p.first << ": " << p.second.to_string() << std::endl;
            }

            for (auto &p : this->stdvec_attr_)
            {
                oss << "\t" << p.first << ": " << std::endl << "\t";
                for (auto &item : p.second)
                {
                    oss << item << ", ";
                }
            }

            oss << std::endl << "--------------------------------------------------------------------";

            return oss.str();
        }

        // Overload operator== for Attr_Point_2.
        bool operator==(const Attr_Point_2 &p) const
        {
            return this->idx() == p.idx() && this->x() == p.r() && this->y() == p.theta();
        }


    };


    //Overload operator<< for Attr_Point_2
    ostream& operator<<(ostream& out, const Attr_Point_2 &p)
    {
        std::cout.setf(ios::fixed);
        out << "idx: " << std::setw(5) << p.idx() << std::setprecision(6)
            << "\tr: " << p.r() << "\ttheta: " << p.theta()
            << "\tx: " << p.x() << "\ty: " << p.y();
        return out;
    }


    //// Hash functor for Attr_Point_2
    //struct hash_Attr_Point
    //{
    //	size_t operator()(const Attr_Point_2 &p) const
    //	{
    //		return std::hash<int>()(p.idx()) ^ std::hash<double>()(p.r()) ^ std::hash<double>()(p.theta());
    //	}
    //};



    class Attr_Point_Set : public std::vector<Attr_Point_2>
    {
    private:
        std::map<unsigned int, unsigned int> mesh_idx2aps_idx;


    public:
        auto emplace_back(Attr_Point_2 &ap)
        {
            this->mesh_idx2aps_idx[ap.idx()] = std::vector<Attr_Point_2>::size();
            return std::vector<Attr_Point_2>::emplace_back(ap);
        }

        auto &access_by_mesh_idx(const unsigned int &idx)
        {
            return (*this)[mesh_idx2aps_idx[idx]];
        }

        auto &const_access_by_mesh_idx(const unsigned int &idx) const
        {
            return (*this)[mesh_idx2aps_idx.at(idx)];
        }

        auto count(const unsigned int &key) const
        {
            return mesh_idx2aps_idx.count(key);
        }

        auto clear()
        {
            mesh_idx2aps_idx.clear();
            return std::vector<Attr_Point_2>::clear();
        }

    };


    // 2D PointCloud for kd_tree
    template <typename T>
    struct PointCloud_2
    {
        struct PC_2_Point
        {
            T  x, y;
        };

        std::vector<PC_2_Point>  pts;
        std::vector<std::vector<unsigned int>> tri_vertices_idx;

        // Must return the number of data points
        inline unsigned int kdtree_get_point_count() const { return pts.size(); }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline T kdtree_get_pt(const unsigned int idx, int dim) const
        {
            if (dim == 0) return pts[idx].x;
            else return pts[idx].y;
        }

        inline std::vector<unsigned int> kdtree_get_tri_vertices(const unsigned int &idx) const
        {
            return tri_vertices_idx.at(idx);
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

        inline void add_point(const T &x, const T &y, const unsigned int &p0_idx, const unsigned int &p1_idx, const unsigned int &p2_idx)
        {
            pts.emplace_back(PC_2_Point{ x, y });
            tri_vertices_idx.emplace_back(std::vector<unsigned int>{p0_idx, p1_idx, p2_idx});
        }

    };



    using namespace nanoflann;

    // GPC computation class.
    template <typename Mesh>
    class GPC
    {

    private:
        Mesh &mesh;
        const unsigned int source_idx;
        const double computation_max_radius;
		const std::vector<double> computation_radius_list;

        Generator<Mesh> gpc_gen;
        Attr_Point_Set all_computed_points; // Info of parameterized 2D points in the local patch. 
        PointCloud_2<double> centroids; // Centroids 
        Point mean_normal, mean_u_max;
        double mean_c_max, mean_c_min;


        // construct a kd-tree index:
        typedef KDTreeSingleIndexAdaptor<
            L2_Simple_Adaptor<double, PointCloud_2<double> >,
            PointCloud_2<double>,
            2 /* dim */
        > my_kd_tree_t;

        //my_kd_tree_t *index;

        std::shared_ptr<my_kd_tree_t> index;


        // Compute the mean normal and mean u_max of all points in this GPC.
        void compute_mean_vals()
        {
            if (this->all_computed_points.size())
            {

                Point normal_accumelater(0, 0, 0), u_max_accumelater(0, 0, 0);

                for (auto &p : this->all_computed_points)
                {
                    Point normal_p, u_max_p;

                    p.get_attr("normal", normal_p);
                    normal_accumelater = normal_accumelater + normal_p;

                    p.get_attr("u_max", u_max_p);
                    u_max_accumelater = u_max_accumelater + u_max_p;
                }

                this->mean_normal = normal_accumelater / this->all_computed_points.size();
                this->mean_u_max = u_max_accumelater / this->all_computed_points.size();

                this->mean_normal = this->mean_normal / this->mean_normal.length();
                this->mean_u_max = this->mean_u_max / this->mean_u_max.length();
            }
            else
            {
                this->mean_normal = this->mean_u_max = Point(0, 0, 0);
            }
        }

        void compute_mean_curvature()
        {
            if (this->all_computed_points.size())
            {

                double c_max_accumelater(0), c_min_accumelater(0);

                for (auto &p : this->all_computed_points)
                {
                    double c_max_p, c_min_p;

                    p.get_attr("c_max", c_max_p);
                    c_max_accumelater += c_max_p;

                    p.get_attr("c_min", c_min_p);
                    c_min_accumelater += c_min_p;
                }

                this->mean_c_max = c_max_accumelater / this->all_computed_points.size();
                this->mean_c_min = c_min_accumelater / this->all_computed_points.size();

            }
            else
            {
                this->mean_c_max = this->mean_c_min = 0;
            }
        }

        inline auto append_centroid(const unsigned int &p0_idx, const unsigned int &p1_idx, const unsigned int &p2_idx)
        {
            auto p0 = all_computed_points.access_by_mesh_idx(p0_idx);
            auto p1 = all_computed_points.access_by_mesh_idx(p1_idx);
            auto p2 = all_computed_points.access_by_mesh_idx(p2_idx);

            auto &&centroid_x = (p0.x() + p1.x() + p2.x()) / 3;
            auto &&centroid_y = (p0.y() + p1.y() + p2.y()) / 3;

            centroids.add_point(centroid_x, centroid_y, p0_idx, p1_idx, p2_idx);
        }


    public:
        GPC(Mesh &mesh_, const unsigned int &source_idx_, const double &computation_max_radius_, const std::vector<double> &computation_radius_list_) :
            mesh(mesh_), source_idx(source_idx_), computation_max_radius(computation_max_radius_), computation_radius_list(computation_radius_list_), gpc_gen(mesh)
        {
			
        }


        //~GPC()
        //{
        //    delete index;
        //}

        void test()
        {
            std::cout << "i      r      theta" << std::endl;
            std::cout << "-------------------" << std::endl;

            auto INF = (std::numeric_limits<typename double>::max)();

            for (int i = 0; i < mesh.n_vertices(); i++)
            {
                const double r = gpc_gen.getDistance(i);
                if (r < INF)
                {
                    const double theta = gpc_gen.getAngle(i);
                    std::cout << i << "    " << r << "    " << theta << std::endl;
                }
            }
        }

		void compute_cf(int number, string base)  
		{
			auto &vphs = this->mesh.get_vphs();

			int mul = 1;
			Mesh patch = mesh; 
			OpenMesh::VertexHandle vertex(source_idx);
			gpc_gen.setStopDist(computation_max_radius * 2.0); 
			gpc_gen.setNodeSource(source_idx);
			gpc_gen.run();
			for (size_t j = 0; j < mesh.n_vertices(); j++)
			{
				OpenMesh::VertexHandle vertex_del(j);
				const double r = gpc_gen.getDistance(j);
				if (r > computation_max_radius * 1.5 )	
				{
					patch.delete_vertex(vertex_del, true);
					for (Mesh::VertexFaceIter vf_it = patch.vf_iter(vertex_del); vf_it.is_valid(); ++vf_it)
					{
						patch.delete_face(*vf_it, true);
					}
				}
			}
			patch.garbage_collection();		
			std::cout << patch.n_vertices() << "   ";
			
			mwArray cf = patch.compute_vertex_cf(mul);

			size_t cf_len = 3 * mesh.get_le_len() + 1;
			mesh.property(vphs.cf, vertex).resize(2 * cf_len); 
			for (size_t i = 0; i < mul * cf_len; i++)
			{
				mesh.property(vphs.cf, vertex)[i] = cf(1, i + 1);
			}

			//////////////////add
			double rlarge = computation_max_radius * 2.0 
			double rsmall = computation_max_radius;
			patch = mesh;
			for (size_t j = 0; j < mesh.n_vertices(); j++)
			{
				OpenMesh::VertexHandle vertex_del(j);
				const double r = gpc_gen.getDistance(j);
				if (r > rlarge)
				{
					patch.delete_vertex(vertex_del, true);
					for (Mesh::VertexFaceIter vf_it = patch.vf_iter(vertex_del); vf_it.is_valid(); ++vf_it)
					{
						patch.delete_face(*vf_it, true);
					}
				}
			}
			patch.garbage_collection();
			std::cout << patch.n_vertices() << "   ";
			mwArray cf2 = patch.compute_vertex_cf(mul);

			patch = mesh;
			for (size_t i = 0; i < mul * cf_len; i++)
			{
				mesh.property(vphs.cf, vertex)[i + cf_len] = cf2(1, i + 1);
			}

		}

		void compute_cf_45(int number, string base)
		{
			auto &vphs = this->mesh.get_vphs();

			int mul = 1;
			Mesh patch = mesh; //add
			OpenMesh::VertexHandle vertex(source_idx);
			gpc_gen.setStopDist(computation_max_radius * 3.0); 
			gpc_gen.setNodeSource(source_idx);
			gpc_gen.run();
			for (size_t j = 0; j < mesh.n_vertices(); j++)
			{
				OpenMesh::VertexHandle vertex_del(j);
				const double r = gpc_gen.getDistance(j);
				if (r > computation_max_radius * 3.0)	
				{
					patch.delete_vertex(vertex_del, true);
					for (Mesh::VertexFaceIter vf_it = patch.vf_iter(vertex_del); vf_it.is_valid(); ++vf_it)
					{
						patch.delete_face(*vf_it, true);
					}
				}
			}
			patch.garbage_collection();
			// write mesh to output.obj
			if (number == 4478)
			{
				try {
					if (!OpenMesh::IO::write_mesh(patch, "outputxyz.off")) { //"+ std::to_string(number)+"  " + base + "
						std::cerr << "Cannot write mesh to file 'output.off'" << std::endl;
					}
				}
				catch (std::exception& x)
				{
					std::cerr << x.what() << std::endl;
				}
			}
			std::cout << patch.n_vertices() << "   ";
			mwArray cf = patch.compute_vertex_cf(mul);

			size_t cf_len = 3 * mesh.get_le_len() +1;
			mesh.property(vphs.cf, vertex).resize(3 * cf_len); 
			for (size_t i = 0; i < mul * cf_len; i++)
			{
				mesh.property(vphs.cf, vertex)[i] = cf(1, i + 1);
			}

			////////////////add
			double rlarge = computation_max_radius * 2.0;  
			double rsmall = computation_max_radius;
			patch = mesh;
			for (size_t j = 0; j < mesh.n_vertices(); j++)
			{
				OpenMesh::VertexHandle vertex_del(j);
				const double r = gpc_gen.getDistance(j);
				if (r > rlarge)
				{
					patch.delete_vertex(vertex_del, true);
					for (Mesh::VertexFaceIter vf_it = patch.vf_iter(vertex_del); vf_it.is_valid(); ++vf_it)
					{
						patch.delete_face(*vf_it, true);
					}
				}
			}
			patch.garbage_collection();
			std::cout << patch.n_vertices() << "   ";
			mwArray cf2 = patch.compute_vertex_cf(mul);

			patch = mesh;
			for (size_t i = 0; i < mul * cf_len; i++)
			{
				mesh.property(vphs.cf, vertex)[i + cf_len] = cf2(1, i + 1);
			}

			////////////////////////////////////
			for (size_t j = 0; j < mesh.n_vertices(); j++)
			{
				OpenMesh::VertexHandle vertex_del(j);
				const double r = gpc_gen.getDistance(j);
				if (r > rsmall)
				{
					patch.delete_vertex(vertex_del, true);
					for (Mesh::VertexFaceIter vf_it = patch.vf_iter(vertex_del); vf_it.is_valid(); ++vf_it)
					{
						patch.delete_face(*vf_it, true);
					}
				}
			}
			patch.garbage_collection();
			std::cout << patch.n_vertices() << "   ";
			mwArray cf3 = patch.compute_vertex_cf(mul);
			for (size_t i = 0; i < mul * cf_len; i++)
			{
				mesh.property(vphs.cf, vertex)[i + 2 * cf_len] = cf3(1, i + 1);
			}
		}

		void compute_cf_90(int number, string base)
		{
			auto &vphs = this->mesh.get_vphs();

			int mul = 3;
			Mesh patch = mesh; //add
			OpenMesh::VertexHandle vertex(source_idx);
			gpc_gen.setStopDist(computation_max_radius * 3.0); 
			gpc_gen.setNodeSource(source_idx);
			gpc_gen.run();
			for (size_t j = 0; j < mesh.n_vertices(); j++)
			{
				OpenMesh::VertexHandle vertex_del(j);
				const double r = gpc_gen.getDistance(j);
				if (r > computation_max_radius * 3.0)	
				{
					patch.delete_vertex(vertex_del, true);
					for (Mesh::VertexFaceIter vf_it = patch.vf_iter(vertex_del); vf_it.is_valid(); ++vf_it)
					{
						patch.delete_face(*vf_it, true);
					}
				}
			}
			patch.garbage_collection();
			std::cout << patch.n_vertices() << "   ";
			mwArray cf = patch.compute_vertex_cf(mul);


			size_t cf_len = 3 * mesh.get_le_len();
			mesh.property(vphs.cf, vertex).resize((1 + 2 + 3) * cf_len); 
			for (size_t i = 0; i < mul * cf_len; i++)
			{
				mesh.property(vphs.cf, vertex)[i] = cf(1, i + 1);
			}
			mul--;

			double rlarge = computation_max_radius * 2.0;  
			double rsmall = computation_max_radius;
			patch = mesh;
			for (size_t j = 0; j < mesh.n_vertices(); j++)
			{
				OpenMesh::VertexHandle vertex_del(j);
				const double r = gpc_gen.getDistance(j);
				if (r > rlarge)
				{
					patch.delete_vertex(vertex_del, true);
					for (Mesh::VertexFaceIter vf_it = patch.vf_iter(vertex_del); vf_it.is_valid(); ++vf_it)
					{
						patch.delete_face(*vf_it, true);
					}
				}
			}
			patch.garbage_collection();
			std::cout << patch.n_vertices() << "   ";
			mwArray cf2 = patch.compute_vertex_cf(mul);

			patch = mesh;
			for (size_t i = 0; i < mul * cf_len; i++)
			{
				mesh.property(vphs.cf, vertex)[i + (mul + 1)*cf_len] = cf2(1, i + 1);
			}
			mul--;
			////////////////////////////////////
			for (size_t j = 0; j < mesh.n_vertices(); j++)
			{
				OpenMesh::VertexHandle vertex_del(j);
				const double r = gpc_gen.getDistance(j);
				if (r > rsmall)
				{
					patch.delete_vertex(vertex_del, true);
					for (Mesh::VertexFaceIter vf_it = patch.vf_iter(vertex_del); vf_it.is_valid(); ++vf_it)
					{
						patch.delete_face(*vf_it, true);
					}
				}
			}
			patch.garbage_collection();
			if (number == 456)
			{
				try {
					if (!OpenMesh::IO::write_mesh(patch, "outputxyz3.off")) { 
						std::cerr << "Cannot write mesh to file 'output.off'" << std::endl;
					}
				}
				catch (std::exception& x)
				{
					std::cerr << x.what() << std::endl;
				}
			}
			std::cout << patch.n_vertices() << "   ";
			mwArray cf3 = patch.compute_vertex_cf(mul);
			for (size_t i = 0; i < mul * cf_len; i++)
			{
				mesh.property(vphs.cf, vertex)[i + (2 * mul + 3)*cf_len] = cf3(1, i + 1);
			}
		}

		void compute_GPC()
		{
			auto &vphs = this->mesh.get_vphs();

			gpc_gen.setStopDist(computation_max_radius);
			gpc_gen.setNodeSource(source_idx);
			gpc_gen.run();

			for (size_t i = 0; i < mesh.n_vertices(); i++)
			{
				const double r = gpc_gen.getDistance(i);
				OpenMesh::VertexHandle vertex(i);
				//if (r < INF)
				if (r < computation_max_radius)
				{
					const double theta = gpc_gen.getAngle(i);

					Attr_Point_2 ap(i, r, theta);
					ap.set_attr("u_max", mesh.property(vphs.u_max, vertex));
					ap.set_attr("u_min", mesh.property(vphs.u_min, vertex));
					ap.set_attr("c_max", mesh.property(vphs.c_max, vertex));
					ap.set_attr("c_min", mesh.property(vphs.c_min, vertex));
					ap.set_attr("normal", mesh.property(vphs.normal, vertex));
					ap.set_attr("hks", mesh.property(vphs.hks, vertex));
					ap.set_attr("resp", mesh.property(vphs.resp, vertex));
					ap.set_attr("le", mesh.property(vphs.le, vertex));
					ap.set_attr("cf", mesh.property(vphs.cf, vertex));
					ap.set_attr("point", mesh.property(vphs.point, vertex));

					all_computed_points.emplace_back(ap);
				}
			}

			this->compute_mean_vals();

			this->auto_rotate();

			this->compute_mean_vals();

			this->compute_mean_curvature();

			vector<unsigned int> tmp;
			tmp.reserve(3);

			for (auto &face : mesh.faces())
			{
				tmp.clear();
				bool flag = true;

				for (auto fv_iter = mesh.fv_begin(face); fv_iter.is_valid(); fv_iter++) // Select the faces in the local patch.
				{
					//cout << fv_iter->idx() << endl;
					if (!all_computed_points.count(fv_iter->idx()))
					{
						flag = false;
						break;
					}
					tmp.emplace_back(fv_iter->idx());
				}

				if (flag)  // It's the triangle in the local patch.
				{
					this->append_centroid(tmp[0], tmp[1], tmp[2]);
				}
			}


			this->index = std::make_shared<my_kd_tree_t>(2 /*dim*/, centroids, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
			this->index->buildIndex();
		}

        // Rotate the GPC patch according to rotation_mat and recompute mean vals.
        void applied_rotation(Rot_mat<double> rotation_mat)
        {
            for (auto &p : this->all_computed_points)
            {
                Point normal_p, u_max_p;

                p.get_attr("normal", normal_p);
                p.set_attr("normal", rotation_mat * normal_p);

                p.get_attr("u_max", u_max_p);
                p.set_attr("u_max", rotation_mat * u_max_p);

                // Ignore u_min.
            }
            this->compute_mean_vals();
        }


        // Automatically rotate the GPC patch to align mean_normal to (0, 0, 1), mean_u_max to x-z plane(norm: (0, 1, 0)).
        void auto_rotate()
        {
            Point z_axis(0, 0, 1), x_axis(1, 0, 0);
            Point rot_axis;
            double theta;


            rot_axis = this->mean_normal.crossProd(z_axis);
            theta = std::acos(this->mean_normal * z_axis / this->mean_normal.length());
            this->applied_rotation(Rot_mat<double>(rot_axis, theta));

            Point curr_mean_u_max = this->mean_u_max;
            Point curr_mean_u_max_xy_proj(curr_mean_u_max.x(), curr_mean_u_max.y(), 0);
            Point target_vec_xy_proj(1, 0, 0);
            rot_axis = z_axis;
            theta = std::acos(curr_mean_u_max_xy_proj * target_vec_xy_proj / curr_mean_u_max_xy_proj.length());
            this->applied_rotation(Rot_mat<double>(rot_axis, theta));
        }


        // Warning: r_set will be cleared firstly.
        void get_patch_vertex_set(Attr_Point_Set &r_set, const double &max_radius) const
        {
            if (!r_set.empty())
                r_set.clear();

            for (const auto &p : this->all_computed_points)
            {
                if (p.r() <= max_radius)
                {
                    Attr_Point_2 p_in = p;
                    r_set.emplace_back(p_in);
                }
            }
        }


        const Attr_Point_2 access_vertex_by_mesh_idx(const unsigned int &idx) const
        {
            const auto &r_val = this->all_computed_points.const_access_by_mesh_idx(idx);
            return r_val;
        }


        //inline std::vector<unsigned int>& find_triangle_vertices(const Point_2 &p) const
        //{
        //	return find_triangle_vertices(p.x(), p.y());
        //}


        // This function is modified from nanoflann/pointcloud_example.cpp
        std::vector<unsigned int> find_triangle_vertices(const Point_2 &p) const
        {
            //using namespace nanoflann;

            double query_pt[2] = { p.x(), p.y() };

            //// construct a kd-tree index:
            //typedef KDTreeSingleIndexAdaptor<
            //    L2_Simple_Adaptor<double, PointCloud_2<double> >,
            //    PointCloud_2<double>,
            //    2 /* dim */
            //> my_kd_tree_t;

            //my_kd_tree_t index(2 /*dim*/, centroids, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
            //index.buildIndex();

            const size_t num_results = 1;
            size_t ret_index;
            double out_dist_sqr;
            nanoflann::KNNResultSet<double> resultSet(num_results);
            resultSet.init(&ret_index, &out_dist_sqr);
            this->index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

            return centroids.kdtree_get_tri_vertices(ret_index);  // Here we simply assume that the given point (x, y) belongs to the triangle whose centroid is cloest to it.
        }



        Point get_mean_normal() const
        {
            return this->mean_normal;
        }


        Point get_mean_u_max() const
        {
            return this->mean_u_max;
        }

        size_t point_num() const
        {
            return this->all_computed_points.size();
        }

    };

}; // End namespace GIGen

#endif // !GPC_H
